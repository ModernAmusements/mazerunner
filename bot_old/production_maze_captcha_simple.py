#!/usr/bin/env python3
"""
Simple Working Maze Captcha System
"""

import cv2
import numpy as np
import json
import hashlib
import time
import math
from flask import Flask, render_template, request, jsonify, session
import base64
from datetime import datetime, timedelta
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'maze_captcha_simple'

# Basic analytics
analytics = {
    'total_attempts': 0,
    'successful_verifications': 0,
    'bot_detected': 0,
    'human_detected': 0
    'human_patterns': []
    'learned_behaviors': {
        'avg_velocity_variance': 0,
        'avg_solve_time': 0,
        'sample_count': 0
    },
    'difficulty_stats': defaultdict(lambda: {'attempts': 0, 'success': 0})
}

def generate_maze(size=15):
    """Generate simple maze"""
    rows, cols = size, size
    maze = np.zeros((rows, cols, dtype=np.uint8)
    
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
    """Simple maze solver"""
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
    """Render maze to image"""
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

def analyze_human_behavior(mouse_data, solve_time, path_data):
    """Simple human behavior analysis"""
    if len(mouse_data) < 5:
        return {
            'is_human': False,
            'confidence': 0.0,
            'reasons': ['Insufficient mouse data']
        }
    
    # Basic checks
    confidence_score = 0.5
    reasons = []
    
    if solve_time < 2.0:
        confidence_score -= 0.4
        reasons.append('Too fast - bot-like')
    elif solve_time > 30.0:
        confidence_score -= 0.2
        reasons.append('Too slow - suspicious')
    else:
        confidence_score += 0.1
        reasons.append('Normal solve time')
    
    if len(path_data) < 10:
        confidence_score -= 0.2
        reasons.append('Insufficient mouse data')
    
    # Simple velocity check
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
    
    if velocities:
        velocity_variance = np.var(velocities) if velocities else 0
    
    # Movement variance check
    if velocity_variance < 100:
        confidence_score -= 0.3
        reasons.append('Too smooth - bot-like')
    else:
        confidence_score += 0.2
        reasons.append('Natural movement variance')
    
    is_human = confidence_score > 0.5
    confidence_score = max(0.0, min(1.0, confidence_score))
    
    return {
        'is_human': is_human,
        'confidence': confidence_score,
        'reasons': reasons if is_human else reasons
    }

def create_simple_bot(captcha_data):
    """Create simple bot"""
    solution = captcha_data.get('solution', [])
    if not solution:
        return None
    
    bot_path = []
    mouse_data = []
    
    # Simple bot with consistent characteristics
    for i, cell in enumerate(solution):
        bot_path.append(cell)
        
        pixel_x = cell[1] * 20 + 10
        pixel_y = cell[0] * 20 + 10
        
        # Minimal jitter (too smooth)
        jitter_x = random.gauss(0, 1)
        jitter_y = random.gauss(0, 1)
        
        # Very consistent timing
        base_timing = 0.05  # Too fast
        mouse_data.append({
            'x': pixel_x + jitter_x,
            'y': pixel_y + jitter_y,
            'timestamp': time.time() - i * base_timing
        })
    
    solve_time = len(bot_path) * base_timing * 0.8
    
    return {
        'path': bot_path,
        'mouse_data': mouse_data,
        'solve_time': solve_time,
        'mimicking_human': False
    }

@app.route('/')
def index():
    return render_template('production_index.html')

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    difficulty = request.args.get('difficulty', 'medium')
    
    size = 11 if difficulty == 'easy' else 15 if difficulty == 'medium' else 21
    
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
    bot_result = create_simple_bot(captcha_data)
    
    if not bot_result:
        return jsonify({'success': False, 'message': 'Failed to create bot'})
    
    session[captcha_id]['mouse_data'] = bot_result['mouse_data']
    session[captcha_id]['start_time'] = time.time() - bot_result['solve_time']
    
    # Simple verification
    analysis = analyze_human_behavior(
        bot_result['mouse_data'],
        bot_result['solve_time'],
        bot_result['path']
    )
    
    if analysis['is_human']:
        analytics['human_detected'] += 1
        analytics['successful_verifications'] += 1
        analytics['difficulty_stats'][captcha_data['difficulty']]['success'] += 1
        message = "Human verified - captcha solved successfully!"
    else:
        analytics['bot_detected'] += 1
        message = f"Bot detected: {analysis['reasons'][0] if analysis['reasons'] else 'Unknown'}"
    
    session.pop(captcha_id)
    
    return jsonify({
        'success': analysis['is_human'],
        'message': message,
        'analysis': analysis
    })

@app.route('/api/verify', methods=['POST'])
def verify_solution():
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    user_path = data.get('path', [])
    
    if not captcha_id or captcha_id not in session:
        return jsonify({
            'success': False,
            'message': 'Invalid captcha',
            'analysis': {'is_human': False, 'confidence': 0.0}
        })
    
    captcha_data = session[captcha_id]
    
    # Basic validation
    try:
        path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
    except:
        return jsonify({
            'success': False,
            'message': 'Invalid path format',
            'analysis': {'is_human': False, 'confidence': 0.0}
        })
    
    if len(path_tuples) == 0:
        return jsonify({
            'success': False,
            'message': 'Empty path',
            'analysis': {'is_human': False, 'confidence': 0.0}
        })
    
    # Check start and end
    if path_tuples[0] != tuple(captcha_data['start']):
        return jsonify({
            'success': False,
            'message': 'Path must start at green square',
            'analysis': {'is_human': False, 'confidence': 0.0}
        })
    
    if path_tuples[-1] != tuple(captcha_data['end']):
        return jsonify({
            'success': False,
            'message': 'Path must end at blue square',
            'analysis': {'is_human': False, 'confidence': 0.0}
        })
    
    # Check path validity
    maze = np.array(captcha_data['maze'])
    for r, c in path_tuples:
        if maze[r, c] != 1:
            return jsonify({
                'success': False,
                'message': 'Path goes through walls',
                'analysis': {'is_human': False, 'confidence': 0.0}
                })
    
    # Analyze behavior and learn
    solve_time = time.time() - captcha_data.get('start_time', captcha_data['created_at'])
    analysis = analyze_human_behavior(
        captcha_data.get('mouse_data', []),
        solve_time,
        path_tuples
    )
    
    # Update analytics
    if analysis['is_human']:
        analytics['human_detected'] += 1
        analytics['successful_verifications'] += 1
        analytics['difficulty_stats'][captcha_data['difficulty']]['success'] += 1
        message = "Human verified - captcha solved successfully!"
    else:
        analytics['bot_detected'] += 1
        message = f"Bot detected: {analysis['reasons'][0] if analysis['reasons'] else 'Unknown'}"
    
    # Store learning if human
    if analysis['is_human']:
        analytics['human_patterns'].append({
            'timestamp': time.time(),
            'mouse_data': captcha_data.get('mouse_data', []),
            'path_data': path_tuples,
            'solve_time': solve_time,
            'velocity_variance': analysis['features']['velocity_variance'],
            'direction_changes': analysis['features']['direction_changes']
        })
    
    session.pop(captcha_id)
    
    return jsonify({
        'success': analysis['is_human'],
        'message': message,
        'analysis': analysis
    })

@app.route('/api/analytics')
def get_analytics():
    return jsonify(analytics)

if __name__ == '__main__':
    print("🚀 Starting Simple Maze Captcha System")
    app.run(debug=True, port=8080, host='0.0.0.0')