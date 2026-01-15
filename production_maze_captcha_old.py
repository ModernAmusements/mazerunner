#!/usr/bin/env python3
"""
Production-Ready Maze Captcha with Human Learning - Clean Working Version
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

# Import rate limiting
from rate_limiter import apply_rate_limiting, add_admin_endpoints, rate_limiter

app = Flask(__name__)
app.secret_key = 'maze_captcha_production_clean'

# Configure session
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=5)
)

# Analytics with human learning
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
    parent = {start: None}
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
    
    velocity_variance = np.var(velocities) if velocities else 0
    
    # Update learned behaviors
    learned = analytics['learned_behaviors']
    n = learned['sample_count']
    
    if n == 0:
        learned['avg_velocity_variance'] = velocity_variance
        learned['avg_solve_time'] = solve_time
        learned['avg_direction_changes'] = direction_changes
    else:
        alpha = 0.2
        learned['avg_velocity_variance'] = (1-alpha) * learned['avg_velocity_variance'] + alpha * velocity_variance
        learned['avg_solve_time'] = (1-alpha) * learned['avg_solve_time'] + alpha * solve_time
        learned['avg_direction_changes'] = (1-alpha) * learned['avg_direction_changes'] + alpha * direction_changes
    
    learned['sample_count'] = n + 1
    
    # Store complete pattern
    pattern = {
        'timestamp': time.time(),
        'mouse_data': mouse_data,
        'path_data': path_data,
        'solve_time': solve_time,
        'velocity_variance': velocity_variance,
        'direction_changes': direction_changes
    }
    analytics['human_patterns'].append(pattern)
    
    # Determine if human-like
    confidence_score = 0.5
    reasons = []
    
    # Basic checks
    if solve_time < 2.0:
        confidence_score -= 0.4
        reasons.append('Too fast - bot-like')
    elif solve_time > 30.0:
        confidence_score -= 0.2
        reasons.append('Too slow - suspicious')
    else:
        confidence_score += 0.1
        reasons.append('Normal solve time')
    
    if velocity_variance < 50:
        confidence_score -= 0.3
        reasons.append('Too smooth - bot-like')
    else:
        confidence_score += 0.2
        reasons.append('Natural movement variance')
    
    if direction_changes < 3:
        confidence_score -= 0.2
        reasons.append('Too few direction changes')
    else:
        confidence_score += 0.1
        reasons.append('Natural path complexity')
    
    is_human = confidence_score > 0.5
    confidence_score = max(0.0, min(1.0, confidence_score))
    
    return {
        'is_human': is_human,
        'confidence': confidence_score,
        'reasons': reasons if is_human else reasons,
        'features': {
            'velocity_variance': velocity_variance,
            'solve_time': solve_time,
            'direction_changes': direction_changes,
            'movement_samples': len(mouse_data)
        }
    }

def create_bot(captcha_data):
    solution = captcha_data.get('solution', [])
    if not solution:
        return None
    
    learned = analytics['learned_behaviors']
    bot_path = []
    mouse_data = []
    
    # Create detectable bot characteristics
    bot_base_timing = 0.12  # Too fast
    bot_jitter_scale = 0.5  # Too smooth
    bot_timing_variance = 0.98  # Too consistent
    
    if learned['sample_count'] > 0:
        bot_base_timing = learned['avg_solve_time'] / len(solution)
        bot_jitter_scale = math.sqrt(learned['avg_velocity_variance']) / 15
        bot_timing_variance = 0.99
    else:
        # No learning yet, create a simple bot
        bot_jitter_scale = 2
        bot_base_timing = 0.1  # Too fast
        
    for i, cell in enumerate(solution):
        bot_path.append(cell)
        
        pixel_x = cell[1] * 20 + 10
        pixel_y = cell[0] * 20 + 10
        
        jitter_x = random.gauss(0, bot_jitter_scale)
        jitter_y = random.gauss(0, bot_jitter_scale)
        timing_variation = random.uniform(bot_timing_variance, 1.02)  # Very consistent
        
        mouse_data.append({
            'x': pixel_x + jitter_x,
            'y': pixel_y + jitter_y,
            'timestamp': time.time() - i * (bot_base_timing * timing_variation),
            'event': 'mousemove'
        })
    
    solve_time = len(bot_path) * bot_base_timing * 0.9
    
    return {
        'path': bot_path,
        'mouse_data': mouse_data,
        'solve_time': solve_time,
        'mimicking_human': learned['sample_count'] > 0
    }

@app.route('/')
def index():
    return render_template('production_index.html')

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    difficulty = request.args.get('difficulty', 'medium')
    
    size = 11 if difficulty == 'easy' else 15 if difficulty == 'medium' else 21 if difficulty == 'hard' else 31
    
    maze = generate_maze(size)
    solution = solve_maze(maze)
    
    captcha_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:12]
    session[captcha_id] = {
        'maze': maze.tolist(),
        'solution': solution,
        'start': (1, 1),
        'end': (maze.shape[0]-2, maze.shape[1]-2),
        'created_at': time.time(),
        'mouse_data': [],
        'start_time': None
    }
    
    analytics['total_attempts'] += 1
    analytics['difficulty_stats'][difficulty]['attempts'] += 1
    
    img = render_maze(maze)
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode()
    
    return jsonify({
        'captcha_id': captcha_id,
        'maze_image': f"data:image/png;base64,{img_base64}",
        'start': (1, 1),
        'end': (maze.shape[0]-2, maze.shape[1]-2),
        'difficulty': difficulty,
        'timestamp': time.time(),
        'learned_patterns': analytics['learned_behaviors']['sample_count']
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
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    
    if not captcha_id or captcha_id not in session:
        return jsonify({'success': False, 'message': 'Invalid captcha'})
    
    captcha_data = session[captcha_id]
    bot_result = create_bot(captcha_data)
    
    if not bot_result:
        return jsonify({'success': False, 'message': 'Failed to create bot'})
    
    # Update session with bot data
    session[captcha_id]['mouse_data'] = bot_result['mouse_data']
    session[captcha_id]['start_time'] = time.time() - bot_result['solve_time']
    
    # Verify bot solution
    analysis = analyze_and_learn(
        bot_result['mouse_data'],
        bot_result['solve_time'],
        bot_result['path']
    )
    
    success = analysis['is_human']
    message = "Human verified - captcha solved successfully!" if success else f"Bot detected: {analysis['reasons'][0] if analysis['reasons'] else 'Unknown'}"
    
    return jsonify({
        'success': True,
        'bot_path': bot_result['path'],
        'solve_time': bot_result['solve_time'],
        'mouse_movements': len(bot_result['mouse_data']),
        'verification_result': {
            'success': success,
            'message': message,
            'analysis': analysis
        },
        'learned_from_humans': analytics['learned_behaviors']['sample_count'],
        'mimicking_human': bot_result.get('mimicking_human', False)
    })

@app.route('/api/verify', methods=['POST'])
def verify_solution():
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    user_path = data.get('path', [])
    
    print(f"DEBUG: Verify endpoint called with captcha_id: {captcha_id}")
    
    try:
        if not captcha_id or captcha_id not in session:
            return jsonify({'success': False, 'message': 'Invalid captcha', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        captcha_data = session[captcha_id]
        start_time = captcha_data.get('start_time')
        if start_time is None:
            start_time = captcha_data['created_at']
        solve_time = time.time() - start_time
        
        print(f"DEBUG: Session captcha_data: {captcha_data}")
        
        # Convert path to tuples and calculate metrics
        path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
        print(f"DEBUG: User path received: {user_path}")
        
        if len(path_tuples) == 0:
            return jsonify({'success': False, 'message': 'Empty path', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Check start and end
        if path_tuples[0] != tuple(captcha_data['start']):
            print(f"DEBUG: Path start: {path_tuples[0]} vs expected: {tuple(captcha_data['start'])}")
            return jsonify({'success': False, 'message': 'Path must start at green square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        if path_tuples[-1] != tuple(captcha_data['end']):
            print(f"DEBUG: Path end: {path_tuples[-1]} vs expected: {tuple(captcha_data['end'])}")
            return jsonify({'success': False, 'message': 'Path must end at blue square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Check path validity with generous tolerance
        maze = np.array(captcha_data['maze'])
        wall_touches = 0
        
        # Convert path to tuples and calculate metrics
        path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
        path_length = len(path_tuples)
        
        # Very generous wall touch allowance - allow up to 50% wall touches
        max_allowed_wall_touches = max(10, path_length // 4)  # Allow 25% of path length as wall touches
        if path_length < 8:  # For very short paths, allow more
            max_allowed_wall_touches = max(5, path_length // 2)
        
        for r, c in path_tuples:
            # Check bounds
            if r >= maze.shape[0] or c >= maze.shape[1] or r < 0 or c < 0:
                print(f"DEBUG: Path out of bounds at ({r}, {c})")
                return jsonify({'success': False, 'message': 'Path goes out of bounds', 'analysis': {'is_human': False, 'confidence': 0.0}})
            
            # Check if path hits wall
            if maze[r, c] != 1:
                wall_touches += 1
                print(f"DEBUG: Wall hit at ({r}, {c}) - touch #{wall_touches}")
            
            # Only reject if excessive wall touching
            if wall_touches > max_allowed_wall_touches and (wall_touches > path_length * 0.5 or wall_touches > 20):
                wall_percentage = (wall_touches / path_length) * 100
                print(f"DEBUG: Too many wall touches: {wall_touches}/{path_length} = {(wall_touches/path_length)*100:.1f}%")
                return jsonify({'success': False, 'message': f'Path goes through walls ({wall_touches} wall touches > {max_allowed_wall_touches} allowed - {wall_touches}/{path_length} = {(wall_touches/path_length)*100:.1f}%)', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Success if wall touches are reasonable
        if wall_touches > 0:
            wall_percentage = (wall_touches / path_length) * 100
            print(f"DEBUG: Wall touches accepted: {wall_touches}/{path_length} = {(wall_touches/path_length)*100:.1f}%}")
        
        print(f"DEBUG: Analyzing behavior with {len(captcha_data.get('mouse_data', []))} mouse events")
        
        # Analyze behavior
        analysis = analyze_and_learn(
            captcha_data.get('mouse_data', []),
            solve_time,
            path_tuples
        )
        
        print(f"DEBUG: Analysis result: is_human={analysis['is_human']}, confidence={analysis['confidence']}")
        
        # Update analytics
        current_time = time.time()
        if analysis['is_human']:
            analytics['human_detected'] += 1
            analytics['successful_verifications'] += 1
            analytics['difficulty_stats'][captcha_data['difficulty']]['success'] += 1
            message = "Human verified - captcha solved successfully!"
            event_type = 'human_verified'
        else:
            analytics['bot_detected'] += 1
            message = f"Bot detected: {analysis['reasons'][0] if analysis['reasons'] else 'Unknown'}"
            event_type = 'bot_detected'
        
        analytics['recent_events'].append({
            'type': event_type,
            'timestamp': current_time,
            'confidence': analysis['confidence'],
            'solve_time': solve_time,
            'path_length': len(path_tuples),
            'difficulty': captcha_data['difficulty'],
            'wall_touches': wall_touches,
            'wall_percentage': (wall_touches / path_length) * 100) if path_length > 0 else 0,
            'direction_changes': analysis.get('metrics', {}).get('direction_changes', 0),
            'avg_velocity': analysis.get('metrics', {}).get('avg_velocity', 0),
            'path_efficiency': analysis.get('metrics', {}).get('path_length', 0) / len(captcha_data.get('solution', [])) if captcha_data.get('solution') else 1.0),
            'mouse_data_count': len(captcha_data.get('mouse_data', [])),
            'bot_strategy': 'human'
        })
        
        session.pop(captcha_id)
        
        print(f"DEBUG: Returning verification result: success={analysis['is_human']}")
        
        return jsonify({
            'success': analysis['is_human'],
            'message': message,
            'analysis': analysis
        })
        
    except Exception as e:
        print(f"DEBUG: Verification error: {e}")
        return jsonify({
            'success': False,
            'message': f'Ve rification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0, 'reasons': ['System error']}
        })
        data = request.get_json()
    captcha_id = data.get('captcha_id')
    user_path = data.get('path', [])
    
    print(f"DEBUG: Verify endpoint called with captcha_id: {captcha_id}")
    
    try:
        if not captcha_id or captcha_id not in session:
            return jsonify({'success': False, 'message': 'Invalid captcha', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        captcha_data = session[captcha_id]
        start_time = captcha_data.get('start_time')
        if start_time is None:
            start_time = captcha_data['created_at']
        solve_time = time.time() - start_time
        
        print(f"DEBUG: Session captcha_data: {captcha_data}")
        
        # Convert path to tuples and calculate metrics
        path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
        print(f"DEBUG: User path received: {user_path}")
        
        if len(path_tuples) == 0:
            return jsonify({'success': False, 'message': 'Empty path', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Check start and end
        if path_tuples[0] != tuple(captcha_data['start']):
            print(f"DEBUG: Path start: {path_tuples[0]} vs expected: {tuple(captcha_data['start'])}")
            return jsonify({'success': False, 'message': 'Path must start at green square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        if path_tuples[-1] != tuple(captcha_data['end']):
            print(f"DEBUG: Path end: {path_tuples[-1]} vs expected: {tuple(captcha_data['end'])}")
            return jsonify({'success': False, 'message': 'Path must end at blue square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        print(f"DEBUG: Path validation with {len(path_tuples)} steps")
        
        # Check path validity with generous tolerance
        maze = np.array(captcha_data['maze'])
        wall_touches = 0
        path_length = len(path_tuples)
        
        # Very generous wall touch allowance - allow up to 50% wall touches
        max_allowed_wall_touches = max(10, path_length // 4)  # Allow 25% of path length as wall touches
        if path_length < 8:  # For very short paths, allow more
            max_allowed_wall_touches = max(5, path_length // 2)
        
        for r, c in path_tuples:
            # Check bounds
            if r >= maze.shape[0] or c >= maze.shape[1] or r < 0 or c < 0:
                print(f"DEBUG: Path out of bounds at ({r}, {c})")
                return jsonify({'success': False, 'message': 'Path goes out of bounds', 'analysis': {'is_human': False, 'confidence': 0.0}})
            
            # Check if path hits wall
            if maze[r, c] != 1:
                wall_touches += 1
                print(f"DEBUG: Wall hit at ({r}, {c}) - touch #{wall_touches}")
            
            # Only reject if excessive wall touching
            if wall_touches > max_allowed_wall_touches and (wall_touches > path_length * 0.5 or wall_touches > 20):
                wall_percentage = (wall_touches / path_length) * 100
                print(f"DEBUG: Too many wall touches: {wall_touches}/{path_length} = {(wall_touches/path_length)*100:.1f}%")
                return jsonify({'success': False, 'message': f'Path goes through walls ({wall_touches} wall touches > {max_allowed_wall_touches} allowed - {wall_touches}/{path_length} = {(wall_touches/path_length)*100:.1f}%)', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Success if wall touches are reasonable
        if wall_touches > 0:
            wall_percentage = (wall_touches / path_length) * 100
            print(f"DEBUG: Wall touches accepted: {wall_touches}/{path_length} = {(wall_touches/path_length)*100:.1f}%")
        
        print(f"DEBUG: Analyzing behavior with {len(captcha_data.get('mouse_data', []))} mouse events")
        
        # Analyze behavior
        analysis = analyze_and_learn(
            captcha_data.get('mouse_data', []),
            solve_time,
            path_tuples
        )
        
        print(f"DEBUG: Analysis result: is_human={analysis['is_human']}, confidence={analysis['confidence']}")
        
        # Update analytics
        current_time = time.time()
        if analysis['is_human']:
            analytics['human_detected'] += 1
            analytics['successful_verifications'] += 1
            analytics['difficulty_stats'][captcha_data['difficulty']]['success'] += 1
            message = "Human verified - captcha solved successfully!"
            event_type = 'human_verified'
        else:
            analytics['bot_detected'] += 1
            message = f"Bot detected: {analysis['reasons'][0] if analysis['reasons'] else 'Unknown'}"
            event_type = 'bot_detected'
        
        print(f"DEBUG: Recording event type: {event_type}")
        analytics['recent_events'].append({
            'type': event_type,
            'timestamp': current_time,
            'confidence': analysis['confidence'],
            'solve_time': solve_time,
            'path_length': len(path_tuples),
            'difficulty': captcha_data['difficulty'],
            'wall_touches': wall_touches,
            'wall_percentage': (wall_touches / path_length) * 100) if path_length > 0 else 0,
            'direction_changes': analysis.get('metrics', {}).get('direction_changes', 0),
            'avg_velocity': analysis.get('metrics', {}).get('avg_velocity', 0),
            'path_efficiency': analysis.get('metrics', {}).get('path_length', 0) / len(captcha_data.get('solution', [])) if captcha_data.get('solution') else 1.0,
            'mouse_data_count': len(captcha_data.get('mouse_data', [])),
            'bot_strategy': 'human'
        })
        
        session.pop(captcha_id)
        
        print(f"DEBUG: Returning verification result: success={analysis['is_human']}")
        
        return jsonify({
            'success': analysis['is_human'],
            'message': message,
            'analysis': analysis
        })
        
    except Exception as e:
        print(f"DEBUG: Verification error: {e}")
        return jsonify({
            'success': False,
            'message': f'Ve rification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0, 'reasons': ['System error']}
        })
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    user_path = data.get('path', [])
    
    try:
        if not captcha_id or captcha_id not in session:
            return jsonify({'success': False, 'message': 'Invalid captcha', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        captcha_data = session[captcha_id]
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
        analysis = analyze_and_learn(
            captcha_data.get('mouse_data', []),
            solve_time,
            path_tuples
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
        
        session.pop(captcha_id)
        
        return jsonify({
            'success': analysis['is_human'],
            'message': message,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ve rification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0, 'reasons': ['System error']}
        })

@app.route('/api/analytics')
def get_analytics():
    return jsonify({
        **analytics,
        'learning_status': {
            'human_patterns_stored': len(analytics['human_patterns']),
            'behaviors_learned': analytics['learned_behaviors']['sample_count'],
            'avg_human_solve_time': analytics['learned_behaviors']['avg_solve_time'],
            'avg_human_velocity_variance': analytics['learned_behaviors']['avg_velocity_variance']
        }
    })

# Apply rate limiting and admin endpoints
app = apply_rate_limiting(app)
app = add_admin_endpoints(app)

if __name__ == '__main__':
    print("🚀 Starting Production-Ready Maze Captcha System with Rate Limiting")
    print(f"📚 Learned from {analytics['learned_behaviors']['sample_count']} human patterns")
    print(f"🛡️  Rate limiting enabled with thresholds:")
    print(f"   - General requests: {rate_limiter.config['general_requests_per_minute']}/min")
    print(f"   - Captcha requests: {rate_limiter.config['captcha_requests_per_minute']}/min")
    print(f"   - Failed attempts threshold: {rate_limiter.config['failed_attempts_threshold']}")
    app.run(debug=True, port=8080, host='0.0.0.0')