#!/usr/bin/env python3
"""
Simple test server for maze captcha system
"""

from flask import Flask, jsonify, render_template
import numpy as np
import cv2
import base64
import random
import time
import math

app = Flask(__name__)
app.secret_key = 'maze_captcha_test'
app.config['HOST'] = '127.0.0.1'
app.config['PORT'] = 8080

# Simple analytics
analytics = {
    'total_attempts': 0,
    'successful_verifications': 0,
    'bot_detected': 0,
    'human_detected': 0,
    'human_patterns': []
    'learned_behaviors': {
        'avg_velocity_variance': 0,
        'avg_solve_time': 0,
        'avg_direction_changes': 0,
        'sample_count': 0
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
                maze[nr, nc] = 1
                maze[nr + dr // 2, c + dc // 2] = 1
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
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows-1 and 0 <= nc < cols-1 and 
                maze[nr, nc] == 0):
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
                        ((c+1)*cell_size, r*cell_size), color, -1)
    
    # Start (green) and end (blue)
    cv2.rectangle(img, (1*cell_size, 1*cell_size), ((2*cell_size, 2*cell_size), (0, 255, 0), -1)
    
    return img

def analyze_and_learn(mouse_data, solve_time, path_data):
    if len(mouse_data) < 10:
        return {
            'is_human': False,
            'confidence': 0.0,
            'reasons': ['Insufficient mouse data']
        }
    
    return {
        'is_human': False,
        'confidence': 0.0
    }

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    captcha_id = f"test_{time.time()}"
    
    maze = generate_maze()
    solution = solve_maze(maze)
    
    session[captcha_id] = {
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
    
    return jsonify({
        'captcha_id': captcha_id,
        'maze_image': f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', render_maze(maze))}",
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
    
    try:
        path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
        
        if len(path_tuples) == 0:
            return jsonify({'success': False, 'message': 'Empty path', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        # Basic validation
        if path_tuples[0] != tuple(captcha_data['start']):
            return jsonify({'success': False, 'message': 'Path must start at green square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        if path_tuples[-1] != tuple(captcha_data['end']):
            return jsonify({'success': False, 'message': 'Path must end at blue square', 'analysis': {'is_human': False, 'confidence': 0.0}})
        
        return jsonify({
            'success': False,
            'message': 'Invalid captcha',
            'analysis': {'is_human': False, 'confidence': 0.0}
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ve rification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0, 'reasons': ['System error']}
        })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)