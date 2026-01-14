import cv2
import numpy as np
import random
import json
import hashlib
import time
import math
from flask import Flask, render_template, request, jsonify, session
import base64
import io
from datetime import datetime, timedelta
from collections import defaultdict, deque

app = Flask(__name__)
app.secret_key = 'maze_captcha_advanced_secret_key'

# Analytics storage
analytics = {
    'total_attempts': 0,
    'successful_verifications': 0,
    'failed_verifications': 0,
    'bot_detected': 0,
    'human_detected': 0,
    'avg_solve_time': 0,
    'median_solve_time': 0,
    'fastest_solve': float('inf'),
    'slowest_solve': 0,
    'difficulty_stats': defaultdict(lambda: {'attempts': 0, 'success': 0, 'avg_time': 0}),
    'hourly_stats': defaultdict(lambda: {'attempts': 0, 'success': 0, 'bot_rate': 0}),
    'mouse_patterns': [],
    'session_data': [],
    'user_agents': defaultdict(int),
    'screen_resolutions': defaultdict(int),
    'timezones': defaultdict(int),
    'path_efficiency': [],  # How close user path is to optimal solution
    'bot_detection_reasons': defaultdict(int),
    'peak_activity_hour': 0,
    'detection_accuracy': {'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0}
}

# Bot detection parameters
BOT_DETECTION_CONFIG = {
    'min_solve_time': 2.0,  # seconds - too fast is suspicious
    'max_solve_time': 300.0,  # seconds - too slow is suspicious
    'min_mouse_movements': 10,  # need some mouse activity
    'max_linear_deviation': 0.3,  # how straight the path is (bots are too straight)
    'min_path_complexity': 5,  # minimum number of direction changes
    'suspicious_timing_variance': 0.5,  # variance in click timing
    'require_hover_time': 0.5  # minimum time hovering over start
}

def generate_maze_with_difficulty(size=15, difficulty='medium'):
    """Generate maze with different difficulty levels"""
    if difficulty == 'easy':
        size = 11
    elif difficulty == 'hard':
        size = 21
    elif difficulty == 'expert':
        size = 31
    
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
    return maze, difficulty

def solve_maze_with_pathfinding(maze):
    """Solve maze using BFS for path validation"""
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

def simulate_bot_solution(captcha_data):
    """Simulate a sophisticated bot solving the maze"""
    maze = np.array(captcha_data['maze'])
    solution = captcha_data['solution']
    
    if not solution:
        return None
    
    # Add realistic human-like variations to the perfect solution
    bot_path = []
    
    # Start with slight delay (thinking time)
    time.sleep(random.uniform(0.5, 1.5))
    
    for i, cell in enumerate(solution):
        # Add small random offsets to simulate imperfect mouse control
        if i == 0:
            # Start exactly at start
            bot_path.append(cell)
        else:
            # 95% chance to follow optimal path, 5% to make small detours
            if random.random() < 0.95:
                bot_path.append(cell)
            else:
                # Add a small, realistic detour
                prev_cell = bot_path[-1]
                possible_moves = [(0,1),(0,-1),(1,0),(-1,0)]
                
                for dr, dc in possible_moves:
                    detour_cell = (prev_cell[0] + dr, prev_cell[1] + dc)
                    if (maze[detour_cell[0], detour_cell[1]] == 1 and 
                        detour_cell != solution[i-2] if i >= 2 else True):
                        bot_path.append(detour_cell)
                        bot_path.append(cell)
                        break
                else:
                    bot_path.append(cell)
        
        # Add realistic typing delays between movements
        if i < len(solution) - 1:
            time.sleep(random.uniform(0.1, 0.3))
    
    return bot_path

def create_bot_mouse_simulation(path, solve_time=5.0, realistic=True):
    """Create realistic mouse tracking data for bot simulation"""
    if not path:
        return []
    
    mouse_data = []
    start_time = time.time() - solve_time
    
    # Generate mouse movements that follow the path
    for i, cell in enumerate(path):
        # Calculate when this movement should happen
        progress = i / len(path)
        current_time = start_time + (solve_time * progress)
        
        # Add some mouse wandering
        base_x = cell[1] * 20 + 10  # Convert to pixel coordinates
        base_y = cell[0] * 20 + 10
        
        if realistic:
            # Add realistic mouse jitter
            jitter_x = random.uniform(-3, 3)
            jitter_y = random.uniform(-3, 3)
            
            mouse_data.append({
                'x': base_x + jitter_x,
                'y': base_y + jitter_y,
                'timestamp': current_time,
                'event': 'mousemove'
            })
            
            # Add occasional extra mouse movements for realism
            if random.random() < 0.2:
                extra_x = base_x + random.uniform(-5, 5)
                extra_y = base_y + random.uniform(-5, 5)
                mouse_data.append({
                    'x': extra_x,
                    'y': extra_y,
                    'timestamp': current_time + random.uniform(0.05, 0.15),
                    'event': 'mousemove'
                })
        else:
            # Suspicious bot - too perfect movements
            mouse_data.append({
                'x': base_x,
                'y': base_y,
                'timestamp': current_time,
                'event': 'mousemove'
            })
    
    return mouse_data

def render_maze_to_png(maze, cell_size=20):
    """Render maze in black and white with colored endpoints"""
    rows, cols = maze.shape
    img = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    
    # Draw maze - black walls, white paths
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                color = (255, 255, 255)  # White path
            else:
                color = (0, 0, 0)  # Black wall
            cv2.rectangle(img, (c*cell_size, r*cell_size), 
                        ((c+1)*cell_size, (r+1)*cell_size), color, -1)
    
    # Draw start and end markers
    cv2.rectangle(img, (1*cell_size, 1*cell_size), 
                 (2*cell_size, 2*cell_size), (0, 255, 0), -1)  # Green start
    cv2.rectangle(img, ((cols-2)*cell_size, (rows-2)*cell_size), 
                 ((cols-1)*cell_size, (rows-1)*cell_size), (0, 0, 255), -1)  # Blue end
    
    return img

def analyze_mouse_pattern(mouse_data):
    """Analyze mouse movement patterns for bot detection"""
    if len(mouse_data) < BOT_DETECTION_CONFIG['min_mouse_movements']:
        return {'is_bot': True, 'reason': 'Insufficient mouse movements'}
    
    # Calculate movement statistics
    velocities = []
    accelerations = []
    angles = []
    
    for i in range(1, len(mouse_data)):
        prev = mouse_data[i-1]
        curr = mouse_data[i]
        
        # Velocity
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        velocity = math.sqrt(dx**2 + dy**2)
        velocities.append(velocity)
        
        # Acceleration
        if i > 1:
            prev_vel = velocities[-2]
            acceleration = abs(velocity - prev_vel)
            accelerations.append(acceleration)
        
        # Angle changes
        if i > 1:
            prev_prev = mouse_data[i-2]
            angle1 = math.atan2(prev['y'] - prev_prev['y'], prev['x'] - prev_prev['x'])
            angle2 = math.atan2(curr['y'] - prev['y'], curr['x'] - prev['x'])
            angle_change = abs(angle2 - angle1)
            angles.append(angle_change)
    
    # Bot detection heuristics
    bot_score = 0
    reasons = []
    
    # Too consistent velocity (bots are very steady)
    if velocities and np.std(velocities) < 5:
        bot_score += 0.3
        reasons.append('Too consistent velocity')
    
    # Too straight paths
    if angles and np.mean(angles) < 0.2:
        bot_score += 0.4
        reasons.append('Path too straight')
    
    # Instantaneous movements (no acceleration)
    if accelerations and np.mean(accelerations) < 1:
        bot_score += 0.3
        reasons.append('Unnatural acceleration pattern')
    
    is_bot = bot_score > 0.5
    return {'is_bot': is_bot, 'bot_score': bot_score, 'reasons': reasons}

def create_advanced_captcha(difficulty='medium'):
    """Create captcha with advanced features"""
    maze, actual_difficulty = generate_maze_with_difficulty(difficulty=difficulty)
    solution = solve_maze_with_pathfinding(maze)
    
    # Create session data with tracking
    captcha_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:12]
    session_data = {
        'maze': maze.tolist(),
        'solution': solution,
        'start': (1, 1),
        'end': (maze.shape[0]-2, maze.shape[1]-2),
        'created_at': time.time(),
        'difficulty': actual_difficulty,
        'mouse_data': [],
        'start_time': None,
        'first_click_time': None,
        'hover_times': [],
        'path_complexity': 0
    }
    
    session[captcha_id] = session_data
    
    # Render maze
    img = render_maze_to_png(maze)
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode()
    
    # Update analytics
    analytics['total_attempts'] += 1
    analytics['difficulty_stats'][actual_difficulty]['attempts'] += 1
    
    current_hour = datetime.now().hour
    analytics['hourly_stats'][current_hour]['attempts'] += 1
    
    return {
        'captcha_id': captcha_id,
        'maze_image': f"data:image/png;base64,{img_base64}",
        'start': session_data['start'],
        'end': session_data['end'],
        'difficulty': actual_difficulty,
        'timestamp': time.time()
    }

def verify_advanced_solution(captcha_id, user_path, client_data):
    """Advanced verification with bot detection"""
    if captcha_id not in session:
        return False, "Invalid captcha", {'is_bot': True, 'reason': 'Invalid session'}
    
    captcha_data = session[captcha_id]
    
    # Check timeout (5 minutes)
    if time.time() - captcha_data['created_at'] > 300:
        session.pop(captcha_id)
        return False, "Captcha expired", {'is_bot': True, 'reason': 'Timeout'}
    
    # Basic path validation
    try:
        user_path_tuples = [(int(p[0]), int(p[1])) for p in user_path]
        unique_path = []
        for cell in user_path_tuples:
            if not unique_path or cell != unique_path[-1]:
                unique_path.append(cell)
        user_path_tuples = unique_path
    except:
        return False, "Invalid path format", {'is_bot': True, 'reason': 'Invalid path data'}
    
    # Verify start and end
    if user_path_tuples[0] != captcha_data['start']:
        return False, "Path must start at green square", {'is_bot': True, 'reason': 'Wrong start'}
    
    if user_path_tuples[-1] != captcha_data['end']:
        return False, "Path must end at blue square", {'is_bot': True, 'reason': 'Wrong end'}
    
    # Verify path is valid (no walls)
    maze = np.array(captcha_data['maze'])
    for r, c in user_path_tuples:
        if maze[r, c] != 1:
            return False, "Path goes through walls", {'is_bot': True, 'reason': 'Invalid path through walls'}
    
    # Timing analysis
    solve_time = time.time() - captcha_data.get('start_time', captcha_data['created_at'])
    if solve_time < BOT_DETECTION_CONFIG['min_solve_time']:
        return False, "Solved too quickly - suspicious", {'is_bot': True, 'reason': 'Too fast'}
    
    if solve_time > BOT_DETECTION_CONFIG['max_solve_time']:
        return False, "Captcha expired", {'is_bot': True, 'reason': 'Too slow'}
    
    # Mouse pattern analysis
    mouse_analysis = analyze_mouse_pattern(captcha_data.get('mouse_data', []))
    
    # Path complexity analysis
    direction_changes = 0
    for i in range(2, len(user_path_tuples)):
        r1, c1 = user_path_tuples[i-2]
        r2, c2 = user_path_tuples[i-1]
        r3, c3 = user_path_tuples[i]
        
        dir1 = (r2-r1, c2-c1)
        dir2 = (r3-r2, c3-c2)
        
        if dir1 != dir2:
            direction_changes += 1
    
    # Final bot detection decision
    bot_indicators = []
    confidence_human = 1.0
    
    if mouse_analysis['is_bot']:
        bot_indicators.append(mouse_analysis['reasons'])
        confidence_human -= 0.4
    
    if direction_changes < BOT_DETECTION_CONFIG['min_path_complexity']:
        bot_indicators.append('Path too simple')
        confidence_human -= 0.3
    
    if len(captcha_data.get('mouse_data', [])) < BOT_DETECTION_CONFIG['min_mouse_movements']:
        bot_indicators.append('Insufficient mouse activity')
        confidence_human -= 0.3
    
    is_bot = confidence_human < 0.5
    
    # Calculate path efficiency (how close to optimal solution)
    optimal_length = len(captcha_data['solution']) if captcha_data['solution'] else 1
    user_efficiency = optimal_length / max(len(user_path_tuples), 1)
    analytics['path_efficiency'].append(user_efficiency)
    
    # Update analytics
    current_hour = datetime.now().hour
    analytics['hourly_stats'][current_hour]['attempts'] += 1
    
    if is_bot:
        analytics['bot_detected'] += 1
        analytics['hourly_stats'][current_hour]['bot_rate'] = (
            analytics['bot_detected'] / analytics['total_attempts'] * 100
        )
        
        # Track bot detection reasons
        for reason in bot_indicators:
            analytics['bot_detection_reasons'][reason] += 1
        
        result_msg = "Bot detected - verification failed"
    else:
        analytics['human_detected'] += 1
        analytics['successful_verifications'] += 1
        analytics['difficulty_stats'][captcha_data['difficulty']]['success'] += 1
        analytics['hourly_stats'][current_hour]['success'] += 1
        
        # Update solve time statistics
        analytics['fastest_solve'] = min(analytics['fastest_solve'], solve_time)
        analytics['slowest_solve'] = max(analytics['slowest_solve'], solve_time)
        
        if analytics['avg_solve_time'] == 0:
            analytics['avg_solve_time'] = solve_time
        else:
            analytics['avg_solve_time'] = (analytics['avg_solve_time'] * 0.9 + solve_time * 0.1)
        
        # Update difficulty-specific stats
        diff_stats = analytics['difficulty_stats'][captcha_data['difficulty']]
        if diff_stats['avg_time'] == 0:
            diff_stats['avg_time'] = solve_time
        else:
            diff_stats['avg_time'] = (diff_stats['avg_time'] * 0.9 + solve_time * 0.1)
        
        result_msg = "Human verified - captcha solved successfully!"
    
    # Track user agent and client info
    if 'user_agent' in client_data:
        analytics['user_agents'][client_data['user_agent']] += 1
    if 'screen_resolution' in client_data:
        analytics['screen_resolutions'][client_data['screen_resolution']] += 1
    if 'timezone' in client_data:
        analytics['timezones'][client_data['timezone']] += 1
    
    # Update peak activity hour
    if analytics['hourly_stats'][current_hour]['attempts'] > analytics['hourly_stats'][analytics['peak_activity_hour']]['attempts']:
        analytics['peak_activity_hour'] = current_hour
    
    # Store mouse pattern for analysis
    analytics['mouse_patterns'].append({
        'timestamp': time.time(),
        'is_bot': is_bot,
        'pattern_length': len(captcha_data.get('mouse_data', [])),
        'solve_time': solve_time,
        'direction_changes': direction_changes
    })
    
    # Clean up session
    session.pop(captcha_id)
    
    return not is_bot, result_msg, {
        'is_bot': is_bot,
        'confidence_human': confidence_human,
        'bot_indicators': bot_indicators,
        'solve_time': solve_time,
        'direction_changes': direction_changes
    }

@app.route('/')
def index():
    return render_template('advanced_index.html')

@app.route('/api/captcha', methods=['GET'])
def get_advanced_captcha():
    difficulty = request.args.get('difficulty', 'medium')
    
    captcha = create_advanced_captcha(difficulty=difficulty)
    return jsonify(captcha)

@app.route('/api/track', methods=['POST'])
def track_mouse_data():
    """Track mouse movements for bot detection"""
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    
    if captcha_id in session:
        mouse_data = {
            'x': data.get('x'),
            'y': data.get('y'),
            'timestamp': time.time(),
            'event': data.get('event', 'move')
        }
        
        session[captcha_id]['mouse_data'].append(mouse_data)
        
        if data.get('event') == 'mousedown' and not session[captcha_id].get('start_time'):
            session[captcha_id]['start_time'] = time.time()
    
    return jsonify({'success': True})

@app.route('/api/verify', methods=['POST'])
def verify_advanced_captcha():
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    user_path = data.get('path', [])
    client_data = data.get('client_data', {})
    
    success, message, analysis = verify_advanced_solution(captcha_id, user_path, client_data)
    
    return jsonify({
        'success': success,
        'message': message,
        'analysis': analysis
    })

@app.route('/api/analytics')
def get_analytics():
    """Return analytics data for dashboard"""
    return jsonify(analytics)

@app.route('/api/bot-simulate', methods=['POST'])
def bot_simulate():
    """Simulate a bot solving the captcha"""
    data = request.get_json()
    captcha_id = data.get('captcha_id')
    skill_level = data.get('skill_level', 'good')  # 'good', 'perfect', 'suspicious'
    
    if captcha_id not in session:
        return jsonify({'success': False, 'message': 'Invalid captcha'})
    
    captcha_data = session[captcha_id]
    maze = np.array(captcha_data['maze'])
    
    # Generate bot solution based on skill level
    if skill_level == 'perfect':
        bot_path = captcha_data['solution']
        solve_time = random.uniform(8.0, 15.0)  # Slower, more human-like
        mouse_data = create_bot_mouse_simulation(bot_path, solve_time)
    elif skill_level == 'good':
        bot_path = simulate_bot_solution(captcha_data)
        solve_time = random.uniform(5.0, 12.0)
        mouse_data = create_bot_mouse_simulation(bot_path, solve_time)
    else:  # suspicious
        bot_path = captcha_data['solution']
        solve_time = random.uniform(0.5, 2.0)  # Too fast - suspicious
        mouse_data = create_bot_mouse_simulation(bot_path, solve_time, realistic=False)
    
    if not bot_path:
        return jsonify({'success': False, 'message': 'Could not generate bot solution'})
    
    # Update session with bot data
    captcha_data['mouse_data'] = mouse_data
    captcha_data['start_time'] = time.time() - solve_time
    
    # Simulate the verification process
    client_data = {
        'solveTime': solve_time,
        'mouseMovements': len(mouse_data),
        'userAgent': 'Bot-Simulator/1.0',
        'screenResolution': '1920x1080',
        'timezone': 'UTC',
        'is_simulation': True
    }
    
    success, message, analysis = verify_advanced_solution(captcha_id, bot_path, client_data)
    
    return jsonify({
        'success': True,
        'bot_path': bot_path,
        'solve_time': solve_time,
        'mouse_movements': len(mouse_data),
        'verification_result': {
            'success': success,
            'message': message,
            'analysis': analysis
        }
    })

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')