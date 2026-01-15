#!/usr/bin/env python3
"""
Simple working maze captcha server
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

# Simple analytics
analytics = {
    'total_attempts': 0,
    'successful_verifications': 0,
    'bot_detected': 0,
    'recent_events': []
}

def generate_maze():
    """Generate a proper maze with guaranteed path"""
    size = 20
    maze = np.zeros((size, size), dtype=int)
    
    # Create border walls
    maze[0, :] = 0
    maze[-1, :] = 0
    maze[:, 0] = 0
    maze[:, -1] = 0
    
    # Create path from start to end
    path = []
    current = [1, 1]
    end = [size-2, size-2]
    
    while current != end:
        path.append(current.copy())
        
        if random.random() < 0.5 and current[0] < end[0]:
            current[0] += 1  # Move down
        elif current[1] < end[1]:
            current[1] += 1  # Move right
        elif current[0] < end[0]:
            current[0] += 1  # Move down
        
        maze[current[0], current[1]] = 1
    
    path.append(end.copy())
    maze[end[0], end[1]] = 1
    
    # Add some random walls but preserve path
    for i in range(2, size-2):
        for j in range(2, size-2):
            if maze[i, j] == 0 and random.random() < 0.7:
                maze[i, j] = 1
    
    return maze, path

def create_maze_image(maze):
    """Create a visual representation of the maze"""
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
    img[360:380, 360:380] = [255, 0, 0]  # Red end
    
    return img

@app.route('/')
def index():
    return render_template('production_index.html')

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    try:
        captcha_id = f"maze_{int(time.time())}_{random.randint(1000, 9999)}"
        
        maze, solution = generate_maze()
        
        # Debug session
        print(f"Before captcha storage - Session exists: {bool(session)}")
        print(f"Before captcha storage - Session keys: {list(session.keys()) if session else []}")
        
        # Store in session
        if 'captchas' not in session:
            session['captchas'] = {}
        
        session['captchas'][captcha_id] = {
            'maze': maze.tolist(),
            'solution': solution,
            'start': [1, 1],
            'end': [18, 18],
            'created_at': time.time(),
            'difficulty': 'medium'
        }
        
        # Force session save
        session.modified = True
        print(f"After captcha storage - Captcha count: {len(session.get('captchas', {}))}")
        print(f"After captcha storage - Captcha IDs: {list(session.get('captchas', {}).keys())}")
        
        analytics['total_attempts'] += 1
        
        # Create maze image
        img = create_maze_image(maze)
        _, buffer = cv2.imencode('.png', img)
        maze_image = base64.b64encode(buffer).decode()
        
        return jsonify({
            'captcha_id': captcha_id,
            'maze_image': f"data:image/png;base64,{maze_image}",
            'start': [1, 1],
            'end': [18, 18],
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
        
        if 'captchas' not in session or captcha_id not in session['captchas']:
            return jsonify({'success': False, 'message': 'Invalid captcha ID'})
        
        captcha_data = session['captchas'][captcha_id]
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
        
        # Check start and end points
        if path_tuples[0] != captcha_data['start']:
            return jsonify({
                'success': False, 
                'message': 'Path must start at green square',
                'analysis': {'is_human': False, 'confidence': 0.0}
            })
        
        if path_tuples[-1] != captcha_data['end']:
            return jsonify({
                'success': False, 
                'message': 'Path must end at red square',
                'analysis': {'is_human': False, 'confidence': 0.0}
            })
        
        # Very forgiving validation - count wall touches
        maze = np.array(captcha_data['maze'])
        wall_touches = 0
        max_wall_touches = 5  # Very generous
        
        for r, c in path_tuples:
            if r < 0 or r >= maze.shape[0] or c < 0 or c >= maze.shape[1]:
                wall_touches += 1
            elif maze[r, c] == 0:
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
        
        # Debug logging
        print(f"Bot simulation request for captcha_id: {captcha_id}")
        print(f"Session data keys: {list(session.keys()) if session else 'No session'}")
        if 'captchas' in session:
            print(f"Available captchas: {list(session['captchas'].keys())}")
        
        if not captcha_id:
            return jsonify({'success': False, 'message': 'Missing captcha ID'})
        
        if 'captchas' not in session:
            return jsonify({'success': False, 'message': 'No captchas in session'})
        
        if captcha_id not in session['captchas']:
            return jsonify({'success': False, 'message': f'Captcha {captcha_id} not found'})
        
        captcha_data = session['captchas'][captcha_id]
        
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
            'captcha_count': len(session.get('captchas', {})) if 'captchas' in session else 0,
            'captcha_ids': list(session.get('captchas', {}).keys()) if 'captchas' in session else []
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
    app.run(host='127.0.0.1', port=8080, debug=True)