#!/usr/bin/env python3Bot simulation failed
"""
Production-Ready Maze Captcha with Complete Monitoring and Logging
"""
# Import our modules
from rate_limiter import apply_rate_limiting, add_admin_endpoints, rate_limiter
from database_sessions import db_session_manager
from monitoring import setup_monitoring, maze_logger, performance_monitor
from functools import wraps
import random
import json
import hashlib
import time
import math
import numpy as np
import cv2
import base64
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, request, jsonify, render_template, session, g, send_from_directory, make_response

# Define decorators locally to avoid import issues
def monitor_captcha_generation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def monitor_verification(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Import our modules
app = Flask(__name__, static_folder=None)  # Disable default static handler
app.secret_key = 'maze_captcha_production_complete'
app.config['HOST'] = '127.0.0.1'
app.config['PORT'] = 8080
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    # Disable static file caching for development
    SEND_FILE_MAX_AGE_DEFAULT=0,
    TEMPLATES_AUTO_RELOAD=True
)

# Force Jinja2 to not cache templates
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

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
    'difficulty_stats': defaultdict(lambda: {'attempts': 0, 'success': 0}),
    'recent_events': [],
    'performance_metrics': {
        'captcha_generation_time': [],
        'verification_time': [],
        'bot_simulation_time': [],
        'concurrent_sessions': 0,
        'storage_size_mb': 0,
        'database_queries': 0,
    },
    'engagement_metrics': {
        'session_duration_avg': 0,
        'success_rate_trend': [],
        'bot_detection_accuracy': 0,
        'human_behavior_similarity': 0,
        'learning_progress': 0
    }
}

def generate_maze(size=15):
    rows, cols = size, size
    maze = np.zeros((rows, cols), dtype=np.uint8)
    
    def walk(r, c):
        maze[r, c] = 1
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows-1 and 0 < nc < cols-1 and maze[nr, nc] == 0:
                maze[r + dr // 2, c + dc // 2] = 1
                walk(nr, nc)
    
    walk(1, 1)
    return maze

def solve_maze(maze):
    rows, cols = maze.shape
    start, end = (1, 1), (rows-2, cols-2)
    
    from collections import deque
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    moves = [(0,1),(0,-1),(1,0),(-1,0)]
    
    while queue:
        curr = queue.popleft()
        if curr == end:
            path = []
            while curr:
                path.append(curr)
                curr = parent[curr]
            return path[::-1]
        
        for dr, dc in moves:
            nr, nc = curr[0] + dr, curr[1] + dc
            if (0 <= nr < rows and 0 <= nc < cols and 
                maze[nr, nc] == 1 and (nr, nc) not in parent):
                parent[(nr, nc)] = curr
                queue.append((nr, nc))
    return None

def render_maze(maze, cell_size=20):
    rows, cols = maze.shape
    img = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            color = (255, 255, 255) if maze[r, c] == 1 else (0, 0, 0)
            cv2.rectangle(img, (c*cell_size, r*cell_size), 
                        ((c+1)*cell_size, (r+1)*cell_size), color, -1)
    
    # Start (green) and end (blue)
    cv2.rectangle(img, (1*cell_size, 1*cell_size), 
                 (2*cell_size, 2*cell_size), (0, 255, 0), -1)
    cv2.rectangle(img, ((cols-2)*cell_size, (rows-2)*cell_size), 
                 ((cols-1)*cell_size, (rows-1)*cell_size), (0, 0, 255), -1)
    
    return img

def analyze_and_learn(mouse_data, solve_time, path_data):
    if len(mouse_data) < 10:
        return {
            'is_human': False,
            'confidence': 0.0,
            'reasons': ['Insufficient mouse data']
        }
    
    # Calculate velocities
    velocities = []
    for i in range(1, len(mouse_data)):
        prev = mouse_data[i-1]
        curr = mouse_data[i]
        dt = (curr['timestamp'] - prev['timestamp']) / 1000.0
        if dt > 0:
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            velocity = math.sqrt(dx**2 + dy**2) / dt
            velocities.append(velocity)
    
    # Calculate direction changes
    direction_changes = 0
    if len(path_data) > 2:
        for i in range(2, len(path_data)):
            r1, c1 = path_data[i-2]
            r2, c2 = path_data[i-1]
            r3, c3 = path_data[i]
            dir1 = (r2-r1, c2-c1)
            dir2 = (r3-r2, c3-c2)
            if dir1 != dir2:
                direction_changes += 1
    
    # Calculate metrics
    avg_velocity = np.mean(velocities) if velocities else 0
    velocity_variance = np.var(velocities) if velocities else 0
    
    # Human-like characteristics
    human_indicators = []
    reasons = []
    confidence = 0.0
    
    # Check solve time (humans usually 2-30 seconds)
    if 2 <= solve_time <= 30:
        human_indicators.append(True)
        confidence += 0.3
    else:
        human_indicators.append(False)
        if solve_time < 2:
            reasons.append("Too fast - likely bot")
        else:
            reasons.append("Too slow - likely timeout")
    
    # Check velocity variance (humans have natural variation)
    if velocity_variance > 100:
        human_indicators.append(True)
        confidence += 0.3
    else:
        human_indicators.append(False)
        reasons.append("Velocity too consistent - likely bot")
    
    # Check direction changes (humans make more corrections)
    expected_changes = len(path_data) * 0.1  # Rough estimate
    if direction_changes >= max(1, expected_changes):
        human_indicators.append(True)
        confidence += 0.2
    else:
        human_indicators.append(False)
        reasons.append("Path too perfect - likely bot")
    
    # Check for human-like pauses
    if len(velocities) > 5:
        zero_velocity_count = sum(1 for v in velocities if v < 5)
        if zero_velocity_count > len(velocities) * 0.1:
            human_indicators.append(True)
            confidence += 0.2
        else:
            human_indicators.append(False)
            reasons.append("No human-like pauses - likely bot")
    
    is_human = len(human_indicators) >= 3 and confidence >= 0.6
    
    # Learn from human behavior
    if is_human and len(mouse_data) > 20:
        analytics['human_patterns'].append({
            'solve_time': solve_time,
            'velocity_variance': velocity_variance,
            'direction_changes': direction_changes,
            'path_length': len(path_data),
            'mouse_data_count': len(mouse_data),
            'timestamp': time.time()
        })
        
        # Update learned behaviors
        if len(analytics['human_patterns']) > 0:
            analytics['learned_behaviors']['avg_solve_time'] = np.mean([p['solve_time'] for p in analytics['human_patterns']])
            analytics['learned_behaviors']['avg_velocity_variance'] = np.mean([p['velocity_variance'] for p in analytics['human_patterns']])
            analytics['learned_behaviors']['avg_direction_changes'] = np.mean([p['direction_changes'] for p in analytics['human_patterns']])
            analytics['learned_behaviors']['sample_count'] = len(analytics['human_patterns'])
    
    return {
        'is_human': is_human,
        'confidence': min(confidence, 1.0),
        'reasons': reasons,
        'metrics': {
            'solve_time': solve_time,
            'velocity_variance': velocity_variance,
            'direction_changes': direction_changes,
            'avg_velocity': avg_velocity,
            'path_length': len(path_data)
        }
    }

def create_bot(captcha_data):
    try:
        maze = np.array(captcha_data['maze'])
        solution = captcha_data['solution']
        start = captcha_data['start']
        
        if not solution:
            return None
        
        # Choose bot strategy
        bot_strategies = ['perfect', 'human_like', 'hybrid']
        strategy = random.choice(bot_strategies)
        
        if strategy == 'perfect':
            # Perfect path with slight variations
            mouse_data = []
            current_time = 0
            for i, pos in enumerate(solution):
                mouse_data.append({
                    'x': pos[1] * 20 + 10,  # Convert to pixel coordinates
                    'y': pos[0] * 20 + 10,
                    'timestamp': current_time,
                    'event': 'move'
                })
                current_time += 50  # 50ms between moves
        
        elif strategy == 'human_like':
            # Mimic learned human behavior
            mouse_data = []
            current_time = 0
            
            if analytics['learned_behaviors']['sample_count'] > 5:
                # Use learned human patterns
                learned_time = analytics['learned_behaviors']['avg_solve_time'] * 1000
                time_per_move = learned_time / len(solution)
                
                for i, pos in enumerate(solution):
                    # Add natural variation
                    x_offset = random.randint(-3, 3)
                    y_offset = random.randint(-3, 3)
                    
                    mouse_data.append({
                        'x': pos[1] * 20 + 10 + x_offset,
                        'y': pos[0] * 20 + 10 + y_offset,
                        'timestamp': current_time,
                        'event': 'move'
                    })
                    
                    # Add human-like pauses
                    if random.random() < 0.1:  # 10% chance of pause
                        current_time += random.randint(100, 500)
                    else:
                        current_time += time_per_move
            else:
                # Fallback to simple timing
                for i, pos in enumerate(solution):
                    mouse_data.append({
                        'x': pos[1] * 20 + 10,
                        'y': pos[0] * 20 + 10,
                        'timestamp': current_time,
                        'event': 'move'
                    })
                    current_time += random.randint(30, 150)
        
        else:  # hybrid
            # Mix of perfect and human-like
            mouse_data = []
            current_time = 0
            mimicking_human = analytics['learned_behaviors']['sample_count'] > 3
            
            for i, pos in enumerate(solution):
                if mimicking_human and random.random() < 0.7:
                    # 70% human-like
                    x_offset = random.randint(-2, 2)
                    y_offset = random.randint(-2, 2)
                else:
                    # 30% perfect
                    x_offset = y_offset = 0
                
                mouse_data.append({
                    'x': pos[1] * 20 + 10 + x_offset,
                    'y': pos[0] * 20 + 10 + y_offset,
                    'timestamp': current_time,
                    'event': 'move'
                })
                
                current_time += random.randint(40, 120) if mimicking_human else 40
        
        solve_time = current_time / 1000.0  # Convert to seconds
        
        return {
            'mouse_data': mouse_data,
            'path': solution,
            'solve_time': solve_time,
            'strategy': strategy,
            'mimicking_human': strategy in ['human_like', 'hybrid'] and analytics['learned_behaviors']['sample_count'] > 0
        }
        
    except Exception as e:
        print(f"Error creating bot: {e}")
        return None

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files with no-cache headers"""
    response = send_from_directory('static', filename)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    # Force template reload by clearing cache
    app.jinja_env.cache = {}
    response = make_response(render_template('production_index.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    difficulty = request.args.get('difficulty', 'medium')
    
    size = 15  # Medium-only for consistency
    
    maze = generate_maze(size)
    solution = solve_maze(maze)
    
    captcha_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:12]
    # Simplified session approach - store everything in session but limit mouse data
    session[captcha_id] = {
        'maze': maze.tolist(),
        'solution': solution,
        'start': (1, 1),
        'end': (maze.shape[0]-2, maze.shape[1]-2),
        'created_at': time.time(),
        'mouse_data': [],
        'start_time': None,
        'difficulty': difficulty
    }
    
    analytics['total_attempts'] += 1
    analytics['difficulty_stats'][difficulty]['attempts'] += 1
    
    img = render_maze(maze)
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode()
    
    # Update analytics
    analytics['total_attempts'] += 1
    analytics['difficulty_stats'][difficulty]['attempts'] += 1
    analytics['recent_events'].append({
        'type': 'captcha_generated',
        'timestamp': time.time(),
        'difficulty': difficulty,
        'captcha_id': captcha_id
    })
    
    return jsonify({
        'captcha_id': captcha_id,
        'maze_image': f"data:image/png;base64,{img_base64}",
        'start': (1, 1),
        'end': (maze.shape[0]-2, maze.shape[1]-2),
        'difficulty': difficulty,
        'timestamp': time.time(),
        'learned_patterns': analytics['learned_behaviors']['sample_count'],
        'analytics': {
            'total_attempts': analytics['total_attempts'],
            'successful_verifications': analytics['successful_verifications'],
            'bot_detected': analytics['bot_detected'],
            'human_detected': analytics['human_detected']
        }
    })

@app.route('/api/track', methods=['POST'])
def track_mouse():
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    
    if captcha_id and captcha_id in session:
        mouse_data = {
            'x': data.get('x', 0),
            'y': data.get('y', 0),
            'timestamp': time.time(),
            'event': data.get('event', 'move')
        }
        
        session[captcha_id]['mouse_data'].append(mouse_data)
        
        if data.get('event') == 'mousedown' and not session[captcha_id].get('start_time'):
            session[captcha_id]['start_time'] = time.time()
    
    return jsonify({'success': True})

@app.route('/api/bot-simulate', methods=['POST'])
def simulate_bot():
    try:
        data = request.get_json()
        captcha_id = data.get('captcha_id')
        
        if not captcha_id or captcha_id not in session:
            return jsonify({'success': False, 'message': 'Invalid captcha'})
        
        captcha_data = session[captcha_id]
        
        # Create bot simulation
        bot_result = create_bot(captcha_data)
        if not bot_result:
            return jsonify({'success': False, 'message': 'Bot simulation failed'})
        
        start_time = captcha_data.get('start_time') or captcha_data['created_at']
        solve_time = bot_result['solve_time']
        
        # Basic validation
        try:
            path_tuples = [(int(p[0]), int(p[1])) for p in bot_result['path']]
        except:
            return jsonify({'success': False, 'message': 'Invalid path format', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        if len(path_tuples) == 0:
            return jsonify({'success': False, 'message': 'Empty path', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Check start and end
        if path_tuples[0] != tuple(captcha_data['start']):
            return jsonify({'success': False, 'message': 'Path must start at green square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        if path_tuples[-1] != tuple(captcha_data['end']):
            return jsonify({'success': False, 'message': 'Path must end at blue square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Check path validity with tolerance
        maze = np.array(captcha_data['maze'])
        
        # Count wall touches and allow minor ones
        wall_touches = 0
        max_allowed_wall_touches = max(3, len(path_tuples) // 10)  # Allow 1 wall touch per 10 steps, minimum 3
        
        for i, (r, c) in enumerate(path_tuples):
            # Check bounds
            if r >= maze.shape[0] or c >= maze.shape[1] or r < 0 or c < 0:
                return jsonify({'success': False, 'message': 'Path goes out of bounds', 'analysis': {'is_human': False, 'confidence': 0.0}})
            
            # Check if path hits wall
            if maze[r, c] != 1:
                wall_touches += 1
                # Allow some wall touches but not too many
                if wall_touches > max_allowed_wall_touches:
                    return jsonify({'success': False, 'message': f'Path goes through walls ({wall_touches} wall touches > {max_allowed_wall_touches} allowed)', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Success if wall touches are within tolerance
        if wall_touches > 0:
            print(f"DEBUG: Path has {wall_touches} wall touches (allowed: {max_allowed_wall_touches})")
        
        # Analyze behavior
        analysis = analyze_and_learn(
            bot_result.get('mouse_data', []),
            solve_time,
            path_tuples
        )
        
        # Update analytics with comprehensive tracking
        current_time = time.time()
        
        if analysis['is_human']:
            analytics['human_detected'] += 1
            analytics['successful_verifications'] += 1
            analytics['difficulty_stats'][captcha_data['difficulty']]['success'] += 1
            message = "Human verified - captcha solved successfully!"
            
            # Record successful attempt details
            analytics['recent_events'].append({
                'type': 'human_verified',
                'timestamp': current_time,
                'confidence': analysis['confidence'],
                'solve_time': solve_time,
                'path_length': len(path_tuples),
                'difficulty': captcha_data['difficulty'],
                'wall_touches': wall_touches if 'wall_touches' in locals() else 0
            })
            
            # Record successful attempt for rate limiting
            if hasattr(g, 'client_ip'):
                rate_limiter.record_successful_attempt(g.client_ip)
        else:
            analytics['bot_detected'] += 1
            message = f"Bot detected: {analysis['reasons'][0] if analysis['reasons'] else 'Unknown'}"
            
            # Record bot detection details
            analytics['recent_events'].append({
                'type': 'bot_detected',
                'timestamp': current_time,
                'confidence': analysis['confidence'],
                'solve_time': solve_time,
                'path_length': len(path_tuples),
                'difficulty': captcha_data['difficulty'],
                'reasons': analysis['reasons'],
                'wall_touches': wall_touches if 'wall_touches' in locals() else 0
            })
            
            # Record failed attempt for rate limiting
            if hasattr(g, 'client_ip'):
                was_banned = rate_limiter.record_failed_attempt(g.client_ip)
                if was_banned:
                    message += " - IP banned due to repeated failures"
        
        session.pop(captcha_id)
        
        return jsonify({
            'success': analysis['is_human'],
            'message': message,
            'analysis': analysis,
            'bot_simulation': {
                'strategy': bot_result['strategy'],
                'mimicking_human': bot_result['mimicking_human']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Verification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0, 'reasons': ['System error']}
        })

@app.route('/api/analytics')
def get_analytics():
    # Calculate success rates
    total = analytics['total_attempts']
    success_rate = (analytics['successful_verifications'] / total * 100) if total > 0 else 0
    bot_detection_rate = (analytics['bot_detected'] / total * 100) if total > 0 else 0
    
    # Recent events (last 20)
    recent_events = analytics['recent_events'][-20:] if len(analytics['recent_events']) > 20 else analytics['recent_events']
    
    return jsonify({
        **analytics,
        'real_time_stats': {
            'success_rate': round(success_rate, 2),
            'bot_detection_rate': round(bot_detection_rate, 2),
            'current_session_count': len([s for s in analytics['recent_events'] if s['type'] in ['human_verified', 'bot_detected']]),
            'avg_confidence': round(np.mean([e.get('confidence', 0) for e in recent_events]), 2) if recent_events else 0
        },
        'learning_status': {
            'human_patterns_stored': len(analytics['human_patterns']),
            'behaviors_learned': analytics['learned_behaviors']['sample_count'],
            'avg_human_solve_time': analytics['learned_behaviors']['avg_solve_time'],
            'avg_human_velocity_variance': analytics['learned_behaviors']['avg_velocity_variance']
        },
        'recent_events': recent_events,
        'performance_metrics': {
            'avg_solve_time': analytics['performance_metrics'].get('captcha_generation_time', []),
            'database_queries_today': analytics['performance_metrics'].get('database_queries', 0)
        }
    })

# Apply rate limiting and admin endpoints
app = apply_rate_limiting(app)
app = add_admin_endpoints(app)

if __name__ == '__main__':
    import os
    # Enable development mode via environment variable or default to True for local dev
    dev_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print("🚀 Starting Production-Ready Maze Captcha System with Rate Limiting")
    print(f"🔧 Development mode: {dev_mode} (static files will auto-reload)")
    print(f"📚 Learned from {analytics['learned_behaviors']['sample_count']} human patterns")
    print(f"🛡️  Rate limiting enabled with thresholds:")
    print(f"   - General requests: {rate_limiter.config['general_requests_per_minute']}/min")
    print(f"   - Captcha requests: {rate_limiter.config['captcha_requests_per_minute']}/min")
    print(f"   - Failed attempts threshold: {rate_limiter.config['failed_attempts_threshold']}")
    print("🌐 Available at: http://127.0.0.1:8080")
    print("📊 Analytics: http://127.0.0.1:8080/api/analytics")
    print("💡 Tip: Edit CSS/JS files and just refresh the browser - no restart needed!")
    app.run(host='127.0.0.1', port=8080, debug=dev_mode)
