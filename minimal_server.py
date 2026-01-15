#!/usr/bin/env python3
"""
Minimal working server for maze captcha system
"""

from flask import Flask, jsonify, render_template, request, session
import numpy as np
import random
import time
import base64
import cv2

app = Flask(__name__)
app.secret_key = 'maze_captcha_minimal'
app.config['HOST'] = '127.0.0.1'
app.config['PORT'] = 8080

# Very simple analytics
analytics = {'total_attempts': 0}

def generate_maze():
    size = 20
    maze = np.ones((size, size), dtype=int)
    
    # Add walls (0) randomly
    for i in range(size):
        for j in range(size):
            if i == 0 or i == size-1 or j == 0 or j == size-1:
                maze[i, j] = 0
            elif random.random() < 0.3:
                maze[i, j] = 0
    
    # Ensure start and end are clear
    maze[1, 1] = 1  # Start
    maze[size-2, size-2] = 1  # End
    
    return maze

def solve_maze(maze):
    start = [1, 1]
    end = [maze.shape[0]-2, maze.shape[1]-2]
    
    # Simple path - go right then down
    path = [start.copy()]
    current = start.copy()
    
    while current[0] < end[0]:
        current[0] += 1
        path.append(current.copy())
    
    while current[1] < end[1]:
        current[1] += 1
        path.append(current.copy())
    
    return path

@app.route('/')
def index():
    return render_template('production_index.html')

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    captcha_id = f"test_{time.time()}"
    
    maze = generate_maze()
    solution = solve_maze(maze)
    
    # Store in session
    if 'captchas' not in session:
        session['captchas'] = {}
    
    session['captchas'][captcha_id] = {
        'maze': maze.tolist(),
        'solution': solution,
        'start': (1, 1),
        'end': (maze.shape[0]-2, maze.shape[1]-2),
        'created_at': time.time(),
        'mouse_data': [],
        'start_time': None,
        'difficulty': 'medium'
    }
    
    analytics['total_attempts'] += 1
    
    # Create simple maze image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cell_size = 20
    
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            x, y = j * cell_size, i * cell_size
            if maze[i, j] == 0:
                img[y:y+cell_size, x:x+cell_size] = [0, 0, 0]  # Black wall
            else:
                img[y:y+cell_size, x:x+cell_size] = [255, 255, 255]  # White path
    
    # Mark start and end
    img[20:40, 20:40] = [0, 255, 0]  # Green start
    img[360:380, 360:380] = [0, 0, 255]  # Blue end
    
    _, buffer = cv2.imencode('.png', img)
    maze_image = base64.b64encode(buffer).decode()
    
    return jsonify({
        'captcha_id': captcha_id,
        'maze_image': f"data:image/png;base64,{maze_image}",
        'start': (1, 1),
        'end': (maze.shape[0]-2, maze.shape[1]-2),
        'difficulty': 'medium'
    })

@app.route('/api/verify', methods=['POST'])
def verify_solution():
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    user_path = data.get('path', [])
    
    if not captcha_id or captcha_id not in session:
        return jsonify({'success': False, 'message': 'Invalid captcha', 'analysis': {'is_human': False, 'confidence': 0.0}})
    
    captcha_data = session[captcha_id]
    start_time = captcha_data.get('start_time')
    if start_time is None:
        start_time = captcha_data['created_at']
    
    solve_time = time.time() - start_time
    
    try:
        path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
        
        if len(path_tuples) == 0:
            return jsonify({'success': False, 'message': 'Empty path', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Simple validation
        if path_tuples[0] != tuple(captcha_data['start']):
            return jsonify({'success': False, 'message': 'Path must start at green square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        if path_tuples[-1] != tuple(captcha_data['end']):
            return jsonify({'success': False, 'message': 'Path must end at blue square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Basic path validation (no wall touches allowed)
        maze = np.array(captcha_data['maze'])
        for r, c in path_tuples:
            if maze[r, c] != 1:
                return jsonify({'success': False, 'message': 'Path goes through walls', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Success if no wall touches
        return jsonify({
            'success': True,
            'message': 'Path validated successfully!',
            'analysis': {
                'is_human': True,
                'confidence': 1.0,
                'reasons': []
            }
        })
        
        # Update analytics
        analytics['successful_verifications'] += 1
        analytics['human_detected'] += 1
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ve rification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0, 'reasons': ['System error']}
        })
        
        session.pop(captcha_id)
    
    return jsonify({
        'success': analysis['is_human'],
        'message': message,
        'analysis': analysis
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)