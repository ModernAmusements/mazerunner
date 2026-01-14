import cv2
import numpy as np
import random
import json
import hashlib
import time
from flask import Flask, render_template, request, jsonify, session
import base64
import io

app = Flask(__name__)
app.secret_key = 'maze_captcha_secret_key'

# Simple maze generator for captcha
def generate_simple_maze(size=15):
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
    
    def bfs():
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
    
    return bfs()

def render_maze_to_png(maze, cell_size=20):
    rows, cols = maze.shape
    img = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    
    # Draw maze
    for r in range(rows):
        for c in range(cols):
            color = (255, 255, 255) if maze[r, c] == 1 else (0, 0, 0)
            cv2.rectangle(img, (c*cell_size, r*cell_size), 
                        ((c+1)*cell_size, (r+1)*cell_size), color, -1)
    
    # Draw start and end markers
    cv2.rectangle(img, (1*cell_size, 1*cell_size), 
                 (2*cell_size, 2*cell_size), (0, 255, 0), -1)
    cv2.rectangle(img, ((cols-2)*cell_size, (rows-2)*cell_size), 
                 ((cols-1)*cell_size, (rows-1)*cell_size), (0, 0, 255), -1)
    
    return img

def create_captcha():
    maze = generate_simple_maze(15)
    solution = solve_maze(maze)
    
    # Create session data
    captcha_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:8]
    session_data = {
        'maze': maze.tolist(),
        'solution': solution,
        'start': (1, 1),
        'end': (maze.shape[0]-2, maze.shape[1]-2),
        'created_at': time.time()
    }
    
    session[captcha_id] = session_data
    
    # Render maze to image
    img = render_maze_to_png(maze)
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode()
    
    return {
        'captcha_id': captcha_id,
        'maze_image': f"data:image/png;base64,{img_base64}",
        'start': session_data['start'],
        'end': session_data['end']
    }

def verify_solution(captcha_id, user_path):
    if captcha_id not in session:
        return False, "Invalid captcha"
    
    captcha_data = session[captcha_id]
    
    # Check timeout (5 minutes)
    if time.time() - captcha_data['created_at'] > 300:
        session.pop(captcha_id)
        return False, "Captcha expired"
    
    # Convert user path to tuples and filter to unique grid cells
    try:
        user_path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
        # Remove consecutive duplicates and keep only unique cells
        unique_path = []
        for cell in user_path_tuples:
            if not unique_path or cell != unique_path[-1]:
                unique_path.append(cell)
        user_path_tuples = unique_path
    except:
        return False, "Invalid path format"
    
    # Verify path starts and ends correctly
    if user_path_tuples[0] != captcha_data['start']:
        return False, "Path must start at green square"
    
    if user_path_tuples[-1] != captcha_data['end']:
        return False, "Path must end at red square"
    
    # Verify path is valid (no walls)
    maze = np.array(captcha_data['maze'])
    for r, c in user_path_tuples:
        if maze[r, c] != 1:
            return False, "Path goes through walls"
    
    # Verify consecutive steps are adjacent (allow diagonal moves)
    for i in range(len(user_path_tuples) - 1):
        r1, c1 = user_path_tuples[i]
        r2, c2 = user_path_tuples[i + 1]
        distance = max(abs(r1 - r2), abs(c1 - c2))
        if distance > 1:
            return False, "Invalid path detected"
    
    # Clean up session
    session.pop(captcha_id)
    return True, "Captcha solved successfully!"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    captcha = create_captcha()
    return jsonify(captcha)

@app.route('/api/verify', methods=['POST'])
def verify_captcha():
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    user_path = data.get('path', [])
    
    success, message = verify_solution(captcha_id, user_path)
    return jsonify({'success': success, 'message': message})

if __name__ == '__main__':
    app.run(debug=True, port=5000)