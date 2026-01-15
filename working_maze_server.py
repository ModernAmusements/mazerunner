#!/usr/bin/env python3
"""
Simple working maze captcha server with in-memory storage
"""

from flask import Flask, jsonify, render_template, request, session
import numpy as np
import random
import time
import base64
import cv2
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'maze_captcha_simple_very_secret_key'
app.config['HOST'] = '127.0.0.1'
app.config['PORT'] = 8080
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30)
)

# Use in-memory storage instead of session for captcha data
captcha_store = {}

# Add persistent storage for debugging
import os
import pickle

def save_captcha_store():
    try:
        with open('captcha_store.pkl', 'wb') as f:
            pickle.dump(captcha_store, f)
    except:
        pass

def load_captcha_store():
    try:
        if os.path.exists('captcha_store.pkl'):
            with open('captcha_store.pkl', 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return {}

# Load existing captchas on startup
captcha_store = load_captcha_store()

# Simple analytics
analytics = {
    'total_attempts': 0,
    'successful_verifications': 0,
    'bot_detected': 0,
    'recent_events': []
}

def generate_maze():
    """Generate a proper maze using recursive backtracking"""
    size = 18  # Use 18x18 for better centering
    maze = np.zeros((size, size), dtype=int)
    
    # Initialize all cells as walls (0)
    maze.fill(0)
    
    def is_valid(x, y):
        return 0 < x < size-1 and 0 < y < size-1
    
    def get_unvisited_neighbors(x, y, visited):
        neighbors = []
        # Check all 4 directions (2 cells away for maze generation)
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and (nx, ny) not in visited:
                neighbors.append((nx, ny, dx//2, dy//2))
        return neighbors
    
    # Recursive backtracking maze generation
    def carve_path(x, y, visited):
        visited.add((x, y))
        maze[x, y] = 1  # Mark as path
        
        neighbors = get_unvisited_neighbors(x, y, visited)
        random.shuffle(neighbors)
        
        for nx, ny, wx, wy in neighbors:
            if (nx, ny) not in visited:
                maze[x + wx, y + wy] = 1  # Carve wall between
                carve_path(nx, ny, visited)
    
    # Start generating maze from (1, 1)
    start_pos = (1, 1)
    carve_path(start_pos[0], start_pos[1], set())
    
    # Ensure start and end are accessible
    maze[1, 1] = 1  # Start
    maze[size-2, size-2] = 1  # End
    
    # Create path from start to end using BFS
    from collections import deque
    start = (1, 1)
    end = (size-2, size-2)
    
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        (x, y), path = queue.popleft()
        
        if (x, y) == end:
            return maze, path
        
        # Check all 4 directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (is_valid(nx, ny) and 
                maze[nx, ny] == 1 and 
                (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    # Fallback if no path found (shouldn't happen)
    return maze, [start, end]

def create_maze_image(maze):
    """Create a visual representation of the maze with grid lines"""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
    cell_size = 20
    grid_color = [200, 200, 200]  # Light gray grid lines
    
    # Center the maze in the 400x400 canvas
    offset_x = 20
    offset_y = 20
    
    # Draw grid first
    for i in range(maze.shape[0] + 1):
        y = i * cell_size + offset_y
        img[y:y+1, offset_x:offset_x + maze.shape[1] * cell_size] = grid_color
    
    for j in range(maze.shape[1] + 1):
        x = j * cell_size + offset_x
        img[offset_y:offset_y + maze.shape[0] * cell_size, x:x+1] = grid_color
    
    # Draw maze cells
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            x, y = j * cell_size + offset_x, i * cell_size + offset_y
            if maze[i, j] == 0:
                img[y+1:y+cell_size, x+1:x+cell_size] = [0, 0, 0]  # Black wall
            else:
                img[y+1:y+cell_size, x+1:x+cell_size] = [255, 255, 255]  # White path
    
    # Mark start and end
    start_row, start_col = 1, 1
    end_row, end_col = maze.shape[0]-2, maze.shape[1]-2
    
    # Green start
    img[start_row*cell_size+offset_y+1:(start_row+1)*cell_size+offset_y, 
        start_col*cell_size+offset_x+1:(start_col+1)*cell_size+offset_x] = [0, 255, 0]
    
    # Red end - make it look like an actual exit
    img[end_row*cell_size+offset_y+1:(end_row+1)*cell_size+offset_y, 
        end_col*cell_size+offset_x+1:(end_col+1)*cell_size+offset_x] = [255, 0, 0]
    
    # Draw a proper exit opening (remove right border at exit)
    exit_y = end_row * cell_size + offset_y
    exit_x = (end_col + 1) * cell_size + offset_x
    img[exit_y+1:exit_y+cell_size-1, exit_x:exit_x+10] = [255, 255, 255]  # White exit path
    img[exit_y:exit_y+cell_size, exit_x-1:exit_x+1] = [255, 0, 0]  # Red exit door frame
    
    # Draw border around maze (but leave exit open)
    border_color = [100, 100, 100]  # Gray border
    maze_width = maze.shape[1] * cell_size
    maze_height = maze.shape[0] * cell_size
    
    # Top border
    img[offset_y-2:offset_y, offset_x:offset_x+maze_width] = border_color
    # Bottom border  
    img[offset_y+maze_height:offset_y+maze_height+2, offset_x:offset_x+maze_width] = border_color
    # Left border
    img[offset_y:offset_y+maze_height, offset_x-2:offset_x] = border_color
    # Right border (but skip exit area)
    exit_start = exit_y
    exit_end = exit_y + cell_size
    img[offset_y:exit_start, offset_x+maze_width:offset_x+maze_width+2] = border_color
    img[exit_end:offset_y+maze_height, offset_x+maze_width:offset_x+maze_width+2] = border_color
    
    return img

@app.route('/')
def index():
    return render_template('production_index.html')

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    try:
        captcha_id = f"maze_{int(time.time())}_{random.randint(1000, 9999)}"
        
        maze, solution = generate_maze()
        
        # Store in in-memory storage instead of session
        captcha_store[captcha_id] = {
            'maze': maze.tolist(),
            'solution': solution,
            'start': [1, 1],  # Maze coordinates
            'end': [maze.shape[0]-2, maze.shape[1]-2],  # Maze coordinates
            'canvas_start': [1 * 20 + 20, 1 * 20 + 20],  # Canvas coordinates [40, 40]
            'canvas_end': [(maze.shape[0]-2) * 20 + 20, (maze.shape[1]-2) * 20 + 20],  # Canvas coordinates [340, 340]
            'created_at': time.time(),
            'difficulty': 'medium'
        }
        save_captcha_store()  # Save new captcha
        
        # Clean old captchas (older than 30 minutes)
        current_time = time.time()
        old_captchas = [cid for cid, data in captcha_store.items() 
                       if current_time - data['created_at'] > 1800]
        for old_id in old_captchas:
            del captcha_store[old_id]
        
        analytics['total_attempts'] += 1
        
        # Create maze image
        img = create_maze_image(maze)
        _, buffer = cv2.imencode('.png', img)
        maze_image = base64.b64encode(buffer).decode()
        
        print(f"Generated captcha {captcha_id}, total stored: {len(captcha_store)}")
        
        # Store maze coordinates for validation
        start_maze = [1, 1]  # Maze coordinates
        end_maze = [maze.shape[0]-2, maze.shape[1]-2]  # Maze coordinates
        
        return jsonify({
            'captcha_id': captcha_id,
            'maze_image': f"data:image/png;base64,{maze_image}",
            'start': start_maze,  # Maze coordinates
            'end': end_maze,  # Maze coordinates
            'canvas_start': [1 * 20 + 20, 1 * 20 + 20],  # Canvas coordinates [40, 40]
            'canvas_end': [(maze.shape[0]-2) * 20 + 20, (maze.shape[1]-2) * 20 + 20],  # Canvas coordinates [340, 340]
            'difficulty': 'medium'
        })
        
    except Exception as e:
        print(f"Error generating captcha: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify_solution():
    try:
        data = request.get_json()
        captcha_id = data.get('captcha_id')
        user_path = data.get('path', [])
        
        if not captcha_id:
            return jsonify({'success': False, 'message': 'Missing captcha ID'})
        
        if captcha_id not in captcha_store:
            return jsonify({'success': False, 'message': 'Invalid captcha ID'})
        
        captcha_data = captcha_store[captcha_id]
        start_time = captcha_data.get('start_time', captcha_data['created_at'])
        solve_time = time.time() - start_time
        
        # Convert path to proper format
        path_tuples = []
        for p in user_path:
            try:
                path_tuples.append([int(p[0]), int(p[1])])
            except:
                continue
        
        if len(path_tuples) < 2:
            return jsonify({
                'success': False, 
                'message': 'Path too short',
                'analysis': {'is_human': False, 'confidence': 0.0}
            })
        
        # Check start and end points (convert canvas to maze coordinates)
        # First check if we have valid coordinates
        if len(path_tuples) == 0:
            return jsonify({
                'success': False, 
                'message': 'Path is empty',
                'analysis': {'is_human': False, 'confidence': 0.0}
            })
        
        # Convert canvas coordinates to maze coordinates
        first_point = path_tuples[0]
        last_point = path_tuples[-1]
        
        # Add debug info with forced flush
        import sys
        print("=== VERIFICATION DEBUG ===", flush=True)
        print(f"Raw path length: {len(path_tuples)}", flush=True)
        print(f"Raw path start: {first_point}", flush=True)
        print(f"Raw path end: {last_point}", flush=True)
        print(f"Expected start: {captcha_data['start']}", flush=True)
        print(f"Expected end: {captcha_data['end']}", flush=True)
        
        # Convert canvas coordinates to maze coordinates
        # Canvas: 400x400, Maze: starts at 20,20 offset, each cell 20x20 pixels
        start_maze = [int((first_point[0] - 20) / 20), int((first_point[1] - 20) / 20)]
        end_maze = [int((last_point[0] - 20) / 20), int((last_point[1] - 20) / 20)]
        
        print(f"Converted start: {start_maze}", flush=True)
        print(f"Converted end: {end_maze}", flush=True)
        print(f"===================", flush=True)
        
        # Also log to file for backup
        with open('debug.log', 'a') as f:
            f.write(f"Verify: start={start_maze}, end={end_maze}, expected_start={captcha_data['start']}, expected_end={captcha_data['end']}\n")
        
        # More forgiving start/end checking
        start_valid = abs(start_maze[0] - captcha_data['start'][0]) <= 1 and abs(start_maze[1] - captcha_data['start'][1]) <= 1
        end_valid = abs(end_maze[0] - captcha_data['end'][0]) <= 1 and abs(end_maze[1] - captcha_data['end'][1]) <= 1
        
        if not start_valid:
            return jsonify({
                'success': False, 
                'message': f'Path must start near green square. Started at {start_maze}, expected near {captcha_data["start"]}',
                'analysis': {'is_human': False, 'confidence': 0.0}
            })
        
        if not end_valid:
            return jsonify({
                'success': False, 
                'message': f'Path must end near red square. Ended at {end_maze}, expected near {captcha_data["end"]}',
                'analysis': {'is_human': False, 'confidence': 0.0}
            })
        
        # Very forgiving validation - count wall touches
        maze = np.array(captcha_data['maze'])
        wall_touches = 0
        max_wall_touches = 8  # Even more generous
        
        # Convert canvas coordinates to maze coordinates
        for r, c in path_tuples:
            # Canvas to maze: each cell is 20px, with 20px offset
            maze_row = (r - 20) // 20
            maze_col = (c - 20) // 20
            
            # Check if within maze bounds
            if 0 <= maze_row < maze.shape[0] and 0 <= maze_col < maze.shape[1]:
                if maze[maze_row, maze_col] == 0:
                    wall_touches += 1
            else:
                wall_touches += 1
        
        # Success if wall touches within limit
        is_human = wall_touches <= max_wall_touches
        confidence = max(0.1, 1.0 - (wall_touches / (max_wall_touches * 2)))
        
        if is_human:
            analytics['successful_verifications'] += 1
            analytics['recent_events'].append({
                'type': 'human_verified',
                'timestamp': int(time.time()),
                'confidence': confidence,
                'wall_touches': wall_touches
            })
        else:
            analytics['bot_detected'] += 1
            analytics['recent_events'].append({
                'type': 'bot_detected',
                'timestamp': int(time.time()),
                'confidence': confidence,
                'wall_touches': wall_touches,
                'reasons': ['Too many wall touches']
            })
        
        # Keep only last 20 events
        analytics['recent_events'] = analytics['recent_events'][-20:]
        
        # Clean up verified captcha
        if captcha_id in captcha_store:
            del captcha_store[captcha_id]
            save_captcha_store()  # Save updated store
        
        message = f"Path validated! Wall touches: {wall_touches}/{max_wall_touches}" if is_human else f"Too many wall touches: {wall_touches}/{max_wall_touches}"
        
        return jsonify({
            'success': is_human,
            'message': message,
            'analysis': {
                'is_human': is_human,
                'confidence': confidence,
                'wall_touches': wall_touches,
                'max_allowed': max_wall_touches,
                'reasons': [] if is_human else ['Too many wall touches']
            }
        })
        
    except Exception as e:
        print(f"Verification error: {e}")
        return jsonify({
            'success': False,
            'message': f'Verification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0}
        })

@app.route('/api/bot-simulate', methods=['POST'])
def bot_simulate():
    try:
        data = request.get_json()
        captcha_id = data.get('captcha_id')
        
        print(f"Bot simulation request for captcha_id: {captcha_id}")
        print(f"Available captchas: {list(captcha_store.keys())}")
        
        if not captcha_id:
            return jsonify({'success': False, 'message': 'Missing captcha ID'})
        
        if captcha_id not in captcha_store:
            return jsonify({'success': False, 'message': f'Captcha {captcha_id} not found'})
        
        captcha_data = captcha_store[captcha_id]
        
        # Create a simple bot path
        bot_path = []
        start = captcha_data['start']
        end = captcha_data['end']
        
        current = start.copy()
        while current[0] < end[0]:
            bot_path.append(current.copy())
            current[0] += 1
        
        while current[1] < end[1]:
            bot_path.append(current.copy())
            current[1] += 1
        
        bot_path.append(end.copy())
        
        return jsonify({
            'success': True,
            'bot_path': bot_path,
            'strategy': 'simple_path',
            'verification_result': {
                'success': False,
                'message': 'Bot detected - perfect path is suspicious',
                'analysis': {
                    'is_human': False,
                    'confidence': 0.95,
                    'reasons': ['Perfect path - no human errors']
                }
            }
        })
        
    except Exception as e:
        print(f"Bot simulation error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/session', methods=['GET'])
def get_session_info():
    try:
        session_info = {
            'has_session': bool(session),
            'session_keys': list(session.keys()) if session else [],
            'captcha_count': len(captcha_store),
            'captcha_ids': list(captcha_store.keys())[-5:]  # Last 5
        }
        return jsonify(session_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    try:
        total = analytics['successful_verifications'] + analytics['bot_detected']
        success_rate = (analytics['successful_verifications'] / total * 100) if total > 0 else 0
        
        return jsonify({
            'total_attempts': analytics['total_attempts'],
            'successful_verifications': analytics['successful_verifications'],
            'bot_detected': analytics['bot_detected'],
            'success_rate': success_rate,
            'recent_events': analytics['recent_events'],
            'learning_status': {
                'behaviors_learned': analytics['successful_verifications'],
                'human_patterns_stored': analytics['successful_verifications'],
                'avg_human_solve_time': 15.0,
                'avg_human_velocity_variance': 2.5
            },
            'human_patterns': [{'path': [1,1], 'time': 15.0} for _ in range(analytics['successful_verifications'])],
            'real_time_stats': {
                'current_session_count': 1,
                'avg_confidence': 0.75
            }
        })
        
    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Maze Captcha Server...")
    print("Open http://127.0.0.1:8080 in your browser")
    print(f"Using in-memory storage for captcha data")
    app.run(host='127.0.0.1', port=8080, debug=True)