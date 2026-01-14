#!/usr/bin/env python3
"""
Production-Ready Maze Captcha with Database Persistence
"""

import cv2
import numpy as np
import random
import json
import hashlib
import time
import math
from flask import Flask, render_template, request, jsonify, session, g
import base64
from datetime import datetime, timedelta
from collections import defaultdict

# Import our modules
from rate_limiter import apply_rate_limiting, add_admin_endpoints, rate_limiter
from database_sessions import db_session_manager

app = Flask(__name__)
app.secret_key = 'maze_captcha_production_database'

# Configure session
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30)
)

# Analytics with database persistence
analytics = {
    'total_attempts': 0,
    'successful_verifications': 0,
    'bot_detected': 0,
    'human_detected': 0,
    'human_patterns': [],
    'learned_behaviors': {
        'avg_velocity_variance': 0,
        'avg_solve_time': 0,
        'avg_direction_changes': 0,
        'sample_count': 0
    },
    'difficulty_stats': defaultdict(lambda: {'attempts': 0, 'success': 0})
}

def generate_maze(size=15):
    rows, cols = size, size
    maze = np.zeros((rows, cols), dtype=np.uint8)
    
    def walk(r, c):
        maze[r, c] = 1
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dr, dc in directions:
            nr, nc = r + dr*2, c + dc*2
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
                maze[r + dr, c + dc] = 1
                walk(nr, nc)
    
    walk(1, 1)
    return maze

def solve_maze(maze, start, end):
    rows, cols = maze.shape
    queue = [(start, [])]
    visited = set()
    
    while queue:
        (r, c), path = queue.pop(0)
        
        if (r, c) == end:
            return path + [(r, c)]
        
        if (r, c) in visited:
            continue
            
        visited.add((r, c))
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and 
                maze[nr, nc] == 1 and (nr, nc) not in visited):
                queue.append(((nr, nc), path + [(r, c)]))
    
    return []

def maze_to_image(maze, start, end, cell_size=20):
    rows, cols = maze.shape
    
    # Create black and white image
    img = np.zeros((rows * cell_size, cols * cell_size), dtype=np.uint8)
    
    # Draw maze walls (black) and paths (white)
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = 255
    
    # Convert to 3-channel for colors
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Mark start point in green
    start_r, start_c = start
    cv2.rectangle(img_color,
                  (start_c * cell_size, start_r * cell_size),
                  ((start_c + 1) * cell_size, (start_r + 1) * cell_size),
                  (0, 255, 0), -1)
    
    # Mark end point in blue
    end_r, end_c = end
    cv2.rectangle(img_color,
                  (end_c * cell_size, end_r * cell_size),
                  ((end_c + 1) * cell_size, (end_r + 1) * cell_size),
                  (255, 0, 0), -1)
    
    return img_color

def analyze_and_learn(mouse_data, solve_time, path, session_id=None):
    velocities = []
    for i in range(1, len(mouse_data)):
        dx = mouse_data[i]['x'] - mouse_data[i-1]['x']
        dy = mouse_data[i]['y'] - mouse_data[i-1]['y']
        dt = mouse_data[i]['timestamp'] - mouse_data[i-1]['timestamp']
        if dt > 0:
            velocity = math.sqrt(dx**2 + dy**2) / dt
            velocities.append(velocity)
    
    velocity_variance = np.var(velocities) if velocities else 0
    
    direction_changes = 0
    for i in range(2, len(path)):
        prev_dir = (path[i-1][0] - path[i-2][0], path[i-1][1] - path[i-2][1])
        curr_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        if prev_dir != curr_dir:
            direction_changes += 1
    
    # Load existing human patterns from database
    db_patterns = db_session_manager.get_human_patterns(limit=100)
    
    is_human = True
    reasons = []
    confidence = 0.8
    
    if velocity_variance > 50:
        is_human = False
        reasons.append(f"Too much velocity variance: {velocity_variance:.1f}")
        confidence = 0.2
    
    if solve_time < 1.0:
        is_human = False
        reasons.append(f"Solved too quickly: {solve_time:.1f}s")
        confidence = 0.3
    
    if direction_changes < len(path) * 0.1:
        is_human = False
        reasons.append(f"Too few direction changes: {direction_changes}")
        confidence = 0.4
    
    if len(path) < 10:
        is_human = False
        reasons.append(f"Path too short: {len(path)} points")
        confidence = 0.3
    
    # Learn from human behavior
    if is_human and len(mouse_data) > 10:
        pattern_data = {
            'velocity_variance': velocity_variance,
            'solve_time': solve_time,
            'direction_changes': direction_changes,
            'path_length': len(path),
            'mouse_data_count': len(mouse_data),
            'timestamp': time.time()
        }
        
        # Store in database
        db_session_manager.store_human_pattern(pattern_data, session_id)
        
        # Update in-memory analytics
        analytics['human_patterns'].append(pattern_data)
        
        # Update learned behaviors
        sample_count = analytics['learned_behaviors']['sample_count'] + len(db_patterns)
        if sample_count > 0:
            analytics['learned_behaviors']['avg_velocity_variance'] = (
                (analytics['learned_behaviors']['avg_velocity_variance'] * (sample_count - 1) + velocity_variance) / sample_count
            )
            analytics['learned_behaviors']['avg_solve_time'] = (
                (analytics['learned_behaviors']['avg_solve_time'] * (sample_count - 1) + solve_time) / sample_count
            )
            analytics['learned_behaviors']['avg_direction_changes'] = (
                (analytics['learned_behaviors']['avg_direction_changes'] * (sample_count - 1) + direction_changes) / sample_count
            )
            analytics['learned_behaviors']['sample_count'] = sample_count
    
    return {
        'is_human': is_human,
        'confidence': confidence,
        'reasons': reasons,
        'velocity_variance': velocity_variance,
        'solve_time': solve_time,
        'direction_changes': direction_changes,
        'path_length': len(path),
        'learned_from_humans': analytics['learned_behaviors']['sample_count']
    }

@app.route('/')
def index():
    return render_template('production_index.html')

@app.route('/api/captcha', methods=['GET'])
def generate_captcha():
    difficulty = request.args.get('difficulty', 'medium')
    difficulty_sizes = {'easy': 11, 'medium': 15, 'hard': 21, 'expert': 31}
    size = difficulty_sizes.get(difficulty, 15)
    
    maze = generate_maze(size)
    start = (1, 1)
    end = (size - 2, size - 2)
    solution = solve_maze(maze, start, end)
    
    # Create unique captcha ID
    captcha_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:12]
    
    # Generate image
    maze_image = maze_to_image(maze, start, end)
    _, buffer = cv2.imencode('.png', maze_image)
    maze_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare captcha data
    captcha_data = {
        'maze': maze.tolist(),
        'start': start,
        'end': end,
        'solution': solution,
        'difficulty': difficulty,
        'created_at': time.time()
    }
    
    # Store in database
    session_id = session.get('session_id', 'anonymous')
    db_session_manager.store_captcha(captcha_id, session_id, captcha_data)
    
    # Also store in session for backward compatibility
    session[captcha_id] = captcha_data
    
    # Log analytics
    db_session_manager.log_analytics(session_id, 'captcha_generated', {
        'difficulty': difficulty,
        'captcha_id': captcha_id
    }, getattr(g, 'client_ip', None))
    
    return jsonify({
        'captcha_id': captcha_id,
        'difficulty': difficulty,
        'start': start,
        'end': end,
        'maze_image': f"data:image/png;base64,{maze_b64}",
        'timestamp': time.time(),
        'learned_patterns': len(analytics['human_patterns'])
    })

@app.route('/api/track', methods=['POST'])
def track_mouse():
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    mouse_data = data.get('mouse_data', [])
    
    try:
        # Try database first, fallback to session
        captcha_data = db_session_manager.get_captcha(captcha_id)
        if not captcha_data:
            captcha_data = session.get(captcha_id)
        
        if not captcha_data:
            return jsonify({'success': False, 'message': 'Invalid captcha'})
        
        captcha_data['mouse_data'] = mouse_data
        captcha_data['start_time'] = time.time()
        
        # Update database
        session_id = session.get('session_id', 'anonymous')
        db_session_manager.store_captcha(captcha_id, session_id, captcha_data)
        
        # Update session
        session[captcha_id] = captcha_data
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bot-simulate', methods=['POST'])
def bot_simulate():
    data = request.get_json()
    difficulty = data.get('difficulty', 'medium')
    bot_mode = data.get('mode', 'good')
    
    difficulty_sizes = {'easy': 11, 'medium': 15, 'hard': 21, 'expert': 31}
    size = difficulty_sizes.get(difficulty, 15)
    
    maze = generate_maze(size)
    start = (1, 1)
    end = (size - 2, size - 2)
    solution = solve_maze(maze, start, end)
    
    # Generate image
    maze_image = maze_to_image(maze, start, end)
    _, buffer = cv2.imencode('.png', maze_image)
    maze_b64 = base64.b64encode(buffer).decode('utf-8')
    
    captcha_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:12]
    
    captcha_data = {
        'maze': maze.tolist(),
        'start': start,
        'end': end,
        'solution': solution,
        'difficulty': difficulty,
        'created_at': time.time()
    }
    
    # Store captcha
    session_id = session.get('session_id', 'anonymous')
    db_session_manager.store_captcha(captcha_id, session_id, captcha_data)
    session[captcha_id] = captcha_data
    
    # Simulate bot path
    cell_size = 20
    user_path = []
    mouse_data = []
    
    for i, (r, c) in enumerate(solution):
        pixel_x = c * cell_size + cell_size // 2
        pixel_y = r * cell_size + cell_size // 2
        user_path.append({'x': pixel_x, 'y': pixel_y})
        
        mouse_data.append({
            'x': pixel_x + (random.random() - 0.5) * 5,
            'y': pixel_y + (random.random() - 0.5) * 5,
            'timestamp': time.time() + i * 0.1
        })
    
    solve_time = len(solution) * 0.1
    if bot_mode == 'perfect':
        solve_time *= 0.8
    elif bot_mode == 'suspicious':
        solve_time *= 0.3
    
    analysis = analyze_and_learn(mouse_data, solve_time, 
                                [(p['x'], p['y']) for p in user_path], session_id)
    
    analytics['bot_detected'] += 1
    
    # Log bot simulation
    db_session_manager.log_analytics(session_id, 'bot_simulation', {
        'difficulty': difficulty,
        'mode': bot_mode,
        'detected_as_human': analysis['is_human'],
        'confidence': analysis['confidence']
    }, getattr(g, 'client_ip', None))
    
    return jsonify({
        'captcha_id': captcha_id,
        'difficulty': difficulty,
        'start': start,
        'end': end,
        'maze_image': f"data:image/png;base64,{maze_b64}",
        'bot_path': user_path,
        'analysis': analysis,
        'solve_time': solve_time,
        'learned_from_humans': analytics['learned_behaviors']['sample_count'],
        'mimicking_human': bot_mode == 'good'
    })

@app.route('/api/verify', methods=['POST'])
def verify_solution():
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    user_path = data.get('path', [])
    
    try:
        # Try database first, fallback to session
        captcha_data = db_session_manager.get_captcha(captcha_id)
        if not captcha_data:
            captcha_data = session.get(captcha_id)
        
        if not captcha_data:
            return jsonify({'success': False, 'message': 'Invalid captcha', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        solve_time = time.time() - captcha_data.get('start_time', captcha_data['created_at'])
        
        # Basic validation
        try:
            path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
        except:
            return jsonify({'success': False, 'message': 'Invalid path format', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        if len(path_tuples) == 0:
            return jsonify({'success': False, 'message': 'Empty path', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Check start and end
        if path_tuples[0] != tuple(captcha_data['start']):
            return jsonify({'success': False, 'message': 'Path must start at green square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        if path_tuples[-1] != tuple(captcha_data['end']):
            return jsonify({'success': False, 'message': 'Path must end at blue square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Check path validity
        maze = np.array(captcha_data['maze'])
        for r, c in path_tuples:
            if maze[r, c] != 1:
                return jsonify({'success': False, 'message': 'Path goes through walls', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Analyze behavior
        session_id = session.get('session_id', 'anonymous')
        analysis = analyze_and_learn(
            captcha_data.get('mouse_data', []),
            solve_time,
            path_tuples,
            session_id
        )
        
        # Update analytics and rate limiting
        if analysis['is_human']:
            analytics['human_detected'] += 1
            analytics['successful_verifications'] += 1
            analytics['difficulty_stats'][captcha_data['difficulty']]['success'] += 1
            message = "Human verified - captcha solved successfully!"
            # Record successful attempt for rate limiting
            if hasattr(g, 'client_ip'):
                rate_limiter.record_successful_attempt(g.client_ip)
        else:
            analytics['bot_detected'] += 1
            message = f"Bot detected: {analysis['reasons'][0] if analysis['reasons'] else 'Unknown'}"
            # Record failed attempt for rate limiting
            if hasattr(g, 'client_ip'):
                was_banned = rate_limiter.record_failed_attempt(g.client_ip)
                if was_banned:
                    message += " - IP banned due to repeated failures"
        
        # Clean up
        db_session_manager.delete_captcha(captcha_id)
        if captcha_id in session:
            session.pop(captcha_id)
        
        # Log verification
        db_session_manager.log_analytics(session_id, 'verification', {
            'success': analysis['is_human'],
            'confidence': analysis['confidence'],
            'solve_time': solve_time,
            'difficulty': captcha_data['difficulty'],
            'path_length': len(path_tuples)
        }, getattr(g, 'client_ip', None))
        
        return jsonify({
            'success': analysis['is_human'],
            'message': message,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Verification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0, 'reasons': ['System error']}
        })

@app.route('/api/analytics')
def get_analytics():
    # Get database stats
    db_stats = db_session_manager.get_database_stats()
    db_analytics = db_session_manager.get_analytics_summary(hours=24)
    
    return jsonify({
        **analytics,
        'database_stats': db_stats,
        'recent_events': db_analytics,
        'learning_status': {
            'human_patterns_stored': len(analytics['human_patterns']),
            'behaviors_learned': analytics['learned_behaviors']['sample_count'],
            'avg_human_solve_time': analytics['learned_behaviors']['avg_solve_time'],
            'avg_human_velocity_variance': analytics['learned_behaviors']['avg_velocity_variance']
        }
    })

@app.before_request
def setup_session():
    """Setup session tracking"""
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:16]

# Apply rate limiting and admin endpoints
app = apply_rate_limiting(app)
app = add_admin_endpoints(app)

# Add database-specific admin endpoints
@app.route('/admin/database-stats')
def get_database_stats():
    """Get database statistics"""
    return jsonify(db_session_manager.get_database_stats())

@app.route('/admin/cleanup', methods=['POST'])
def cleanup_database():
    """Clean up expired sessions"""
    deleted = db_session_manager.cleanup_expired()
    return jsonify({'deleted_entries': deleted})

if __name__ == '__main__':
    print("🚀 Starting Production-Ready Maze Captcha with Database Persistence")
    print(f"📚 Learned from {analytics['learned_behaviors']['sample_count']} human patterns")
    print(f"🗄️  Database persistence enabled")
    print(f"🛡️  Rate limiting enabled with thresholds:")
    print(f"   - General requests: {rate_limiter.config['general_requests_per_minute']}/min")
    print(f"   - Captcha requests: {rate_limiter.config['captcha_requests_per_minute']}/min")
    print(f"   - Failed attempts threshold: {rate_limiter.config['failed_attempts_threshold']}")
    
    # Clean up old data on startup
    deleted = db_session_manager.cleanup_expired()
    print(f"🧹 Cleaned up {deleted} expired database entries")
    
    app.run(debug=True, port=8080, host='0.0.0.0')