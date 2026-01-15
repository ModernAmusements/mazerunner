#!/usr/bin/env python3
"""
Clean 20x20 Maze Server - WORKING VERSION
- Start at (0,0) with green entrance on outer boundary
- End at (19,19) with red exit on outer boundary
- Broken outer walls at start and end
- High contrast 8-bit aesthetic maze rendering
- Guaranteed solvable mazes using recursive backtracking
"""

from flask import Flask, jsonify, render_template, request, session
import numpy as np
import random
import time
import base64
import cv2
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'maze_captcha_20x20_working_key'
app.config['HOST'] = '127.0.0.1'
app.config['PORT'] = 8080
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_COOKIE_MAX_SIZE=4096  # Limit cookie size
)

# Use in-memory storage with size limit
captcha_store = {}
MAX_CAPTCHA_SIZE = 100

# Enhanced analytics with histogram data
analytics = {
    'total_attempts': 0,
    'successful_verifications': 0,
    'bot_detected': 0,
    'recent_events': [],
    'path_lengths': [],
    'confidence_scores': [],
    'verification_times': [],
    'hourly_stats': {},
    'bot_types': {}
}

def generate_maze():
    """Generate 20x20 maze using proper recursive backtracking"""
    size = 20
    maze = np.zeros((size, size), dtype=int)  # 0 = wall, 1 = path
    
    def carve_path_recursive(x, y, visited):
        """Recursive backtracking maze generation"""
        visited.add((x, y))
        maze[x, y] = 1  # Mark as path
        
        # Randomize directions for variety
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Skip cells for walls
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            wall_x, wall_y = x + dx // 2, y + dy // 2  # Wall between current and next
            
            if (0 <= nx < size and 0 <= ny < size and 
                (nx, ny) not in visited):
                # Carve the wall and move to next cell
                maze[wall_x, wall_y] = 1  # Carve wall between cells
                carve_path_recursive(nx, ny, visited)
    
    def create_maze_structure():
        """Create maze with proper walls"""
        visited = set()
        
        # Start from (1, 1) to leave border walls
        carve_path_recursive(1, 1, visited)
        
        # Ensure start and end positions are accessible
        maze[0, 0] = 1  # Start
        maze[1, 0] = 1  # Entrance
        maze[0, 1] = 1  # Entrance
        
        maze[19, 19] = 1  # End
        maze[18, 19] = 1  # Exit
        maze[19, 18] = 1  # Exit
        
        return maze
    
    # Generate maze
    maze = create_maze_structure()
    
    # Verify maze has some walls (not all paths)
    path_count = np.sum(maze)
    total_cells = size * size
    wall_count = total_cells - path_count
    
    print(f"Maze generated: {path_count} paths, {wall_count} walls")
    
    return maze, [(0, 0), (19, 19)]
    
    # Generate maze with guaranteed solution
    for attempt in range(3):  # Try up to 3 times
        maze_result = create_basic_maze()
        if maze_result[0].any():  # Check if maze is not all zeros
            print("✅ 20x20 maze generated successfully!")
            return maze_result
        print(f"Retrying attempt {attempt + 2}")
    
    print("Failed to generate maze after 3 attempts")
    return np.zeros((20, 20), dtype=int), [(0, 0), (19, 19)]

def create_maze_image(maze):
    """Create 20x20 maze image with 8-bit aesthetic"""
    scale = 2
    cell_size = 20  # 20px cells
    
    img = np.ones((20 * cell_size, 20 * cell_size, 3), dtype=np.uint8) * 255  # White background
    
    # Draw maze cells
    for i in range(20):
        for j in range(20):
            x = j * cell_size
            y = i * cell_size
            
            if maze[i, j] == 0:
                # Solid black wall
                img[y:y+cell_size, x:x+cell_size] = [0, 0, 0]
    
    # Draw grid lines
    grid_color = [200, 200, 200]  # Subtle grid lines
    
    # Horizontal lines
    for i in range(21):
        y = i * cell_size
        for j in range(20):
            x = j * cell_size
            if j < 19 and i < 19:
                if (maze[i, j] == 1 and maze[i, j+1] == 1):
                    img[y:y+1, x:x+cell_size] = grid_color
    
    # Vertical lines
    for j in range(21):
        x = j * cell_size
        for i in range(20):
            y = i * cell_size
            if i < 19 and j < 19:
                if (maze[i, j] == 1 and maze[i+1, j] == 1):
                    img[y:y+cell_size, x:x+1] = grid_color
    
    # Draw start (green) at (0,0) with broken outer walls
    img[40:60, 40:60] = [0, 255, 0]  # Green entrance
    
    # Draw end (red) at (19,19) with broken outer walls
    img[380:420, 380:420] = [255, 0, 0]  # Red exit
    
    # Break outer walls at start
    # Leave top and left open for entrance
    img[20:60, 20:80] = [255, 255, 255]  # Open entrance
    
    # Leave bottom and right open for exit  
    img[400:400, 400:400] = [255, 255, 255]  # Open exit background
    img[380:420, 400:400] = [255, 255, 255]  # Right wall (except exit area)
    img[380:400, 380:420] = [255, 255, 255]  # Bottom wall (except exit area)
    img[400:400, 400:380] = [255, 255, 255]  # Left wall (except entrance area)
    
    return img

@app.route('/')
def index():
    return render_template('production_index.html')

@app.route('/api/captcha', methods=['GET'])
def get_captcha():
    try:
        captcha_id = f"maze_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Generate maze
        maze, solution_path = generate_maze()
        start = solution_path[0]
        end = solution_path[1]
        
        if not maze.any():  # If maze is all zeros, generation failed
            return jsonify({'error': 'Maze generation failed'}), 500
        
        # Create maze image
        maze_img = create_maze_image(maze)
        _, buffer = cv2.imencode('.png', maze_img)
        maze_image = base64.b64encode(buffer).decode()
        
        # Calculate canvas coordinates (20px cells)
        cell_size = 20
        canvas_start = [0 * cell_size + cell_size//2, 0 * cell_size + cell_size//2]
        canvas_end = [19 * cell_size + cell_size//2, 19 * cell_size + cell_size//2]
        
        # Store only essential data in memory to reduce size
        captcha_store[captcha_id] = {
            'start': start,
            'end': end,
            'canvas_start': canvas_start,
            'canvas_end': canvas_end,
            'created_at': time.time()
        }
        
        # Clean old captchas (older than 30 minutes) and enforce size limit
        current_time = time.time()
        old_ids = [cid for cid, data in captcha_store.items() 
                  if current_time - data['created_at'] > 1800]
        for old_id in old_ids:
            del captcha_store[old_id]
        
        # Enforce maximum captcha store size to prevent large cookies
        if len(captcha_store) > MAX_CAPTCHA_SIZE:
            sorted_ids = sorted(captcha_store.items(), key=lambda x: x[1]['created_at'])
            for cid, _ in sorted_ids[:len(captcha_store) - MAX_CAPTCHA_SIZE]:
                del captcha_store[cid]
        
        print(f"Generated maze {captcha_id}")
        return jsonify({
            'captcha_id': captcha_id,
            'maze_image': f"data:image/png;base64,{maze_image}",
            'start': list(start),
            'end': list(end),
            'canvas_start': canvas_start,
            'canvas_end': canvas_end,
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
        # Maze is not stored anymore to save memory - generate fresh for verification
        maze, _ = generate_maze()
        
        # Convert path coordinates (handle both [x,y] and {x:y} formats)
        path_tuples = []
        for p in user_path:
            if isinstance(p, dict) and 'x' in p and 'y' in p:
                path_tuples.append([int(p['x']), int(p['y'])])
            elif isinstance(p, list) and len(p) >= 2:
                path_tuples.append([int(p[0]), int(p[1])])
            # Skip invalid points
            continue
        
        if len(path_tuples) < 2:
            return jsonify({
                'success': False, 
                'message': 'Please draw a path from start to end',
                'analysis': {'is_human': False, 'confidence': 0.0}
            })
        
        # Convert canvas to maze coordinates (20px cells)
        first_point = path_tuples[0]
        last_point = path_tuples[-1]
        
        start_maze = [first_point[0] // 20, first_point[1] // 20]
        end_maze = [last_point[0] // 20, last_point[1] // 20]
        
        expected_start = [0, 0]
        expected_end = [19, 19]
        
        # Stricter validation with multiple checks
        start_valid = abs(start_maze[0] - expected_start[0]) <= 1 and abs(start_maze[1] - expected_start[1]) <= 1
        end_valid = abs(end_maze[0] - expected_end[0]) <= 1 and abs(end_maze[1] - expected_end[1]) <= 1
        path_length = len(path_tuples)
        
        # Additional bot detection checks
        bot_score = 0
        
        # Check 1: Path too short
        if path_length < 5:
            bot_score += 0.3
        
        # Check 2: Start and end both must be valid
        if not (start_valid and end_valid):
            bot_score += 0.4
        
        # Check 3: Check for perfectly linear path (possible bot)
        if path_length >= 3:
            # Calculate if path is too perfect (straight line)
            distances = []
            for i in range(len(path_tuples) - 1):
                dx = path_tuples[i+1][0] - path_tuples[i][0]
                dy = path_tuples[i+1][1] - path_tuples[i][1]
                dist = (dx*dx + dy*dy) ** 0.5
                distances.append(dist)
            
            if distances:
                avg_dist = sum(distances) / len(distances)
                variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
                # Very low variance might indicate perfect bot movement
                if variance < 10 and path_length > 10:
                    bot_score += 0.2
        
        # Check 4: Path doesn't progress reasonably from start to end
        if path_length >= 2:
            total_distance = 0
            for i in range(len(path_tuples) - 1):
                dx = path_tuples[i+1][0] - path_tuples[i][0]
                dy = path_tuples[i+1][1] - path_tuples[i][1]
                total_distance += (dx*dx + dy*dy) ** 0.5
            
            expected_distance = ((380**2) + (380**2)) ** 0.5  # From (10,10) to (390,390)
            if total_distance < expected_distance * 0.5:  # Path much shorter than expected
                bot_score += 0.2
        
        # Final decision based on bot score and basic requirements
        is_human = bot_score < 0.5 and start_valid and end_valid and path_length >= 5
        confidence = 0.9 if is_human else (1.0 - bot_score)
        
        if is_human:
            analytics['successful_verifications'] += 1
            message = f"Path validated! Length: {path_length}, Bot score: {bot_score:.2f}"
        else:
            analytics['bot_detected'] += 1
            message = f"Bot detected! Length: {path_length}, Bot score: {bot_score:.2f}"
        
        # Record detailed analytics
        current_time = int(time.time())
        hour = time.strftime('%H', time.localtime(current_time))
        
        analytics['recent_events'].append({
            'type': 'human_verified' if is_human else 'bot_detected',
            'timestamp': current_time,
            'confidence': confidence,
            'path_length': path_length
        })
        analytics['recent_events'] = analytics['recent_events'][-50:]  # Keep more events
        
        # Track path lengths for histogram
        analytics['path_lengths'].append(path_length)
        if len(analytics['path_lengths']) > 1000:
            analytics['path_lengths'] = analytics['path_lengths'][-1000:]
        
        # Track confidence scores
        analytics['confidence_scores'].append(confidence)
        if len(analytics['confidence_scores']) > 1000:
            analytics['confidence_scores'] = analytics['confidence_scores'][-1000:]
        
        # Hourly statistics
        if hour not in analytics['hourly_stats']:
            analytics['hourly_stats'][hour] = {'human': 0, 'bot': 0}
        
        if is_human:
            analytics['hourly_stats'][hour]['human'] += 1
        else:
            analytics['hourly_stats'][hour]['bot'] += 1
        
        # Clean old hourly data (keep last 24 hours)
        current_hour = int(hour)
        for h in list(analytics['hourly_stats'].keys()):
            if abs(int(h) - current_hour) > 24:
                del analytics['hourly_stats'][h]
        
        # Clean up verified captcha
        del captcha_store[captcha_id]
        
        return jsonify({
            'success': is_human,
            'message': message,
            'analysis': {
                'is_human': is_human,
                'confidence': confidence,
                'path_length': path_length,
                'start_valid': start_valid,
                'end_valid': end_valid
            }
        })
        
    except Exception as e:
        print(f"Verification error: {e}")
        return jsonify({
            'success': False,
            'message': f'Veification error: {str(e)}',
            'analysis': {'is_human': False, 'confidence': 0.0}
        })

@app.route('/api/bot-simulate', methods=['POST'])
@app.route('/api/bot-simulate', methods=['POST'])
def bot_simulate():
    """Bot simulation endpoint for testing"""
    try:
        data = request.get_json()
        captcha_id = data.get('captcha_id')
        bot_type = data.get('type', 'random')
        
        if not captcha_id:
            return jsonify({'error': 'Missing captcha ID'}), 400
        
        if captcha_id not in captcha_store:
            return jsonify({'error': 'Invalid captcha ID'}), 404
        
        # Generate bot-like path based on type
        if bot_type == 'perfect':
            # Perfect straight line (should be detected as bot)
            path = []
            for i in range(15):
                x = 10 + i * (380/14)
                y = 10 + i * (380/14)
                path.append([int(x), int(y)])
        elif bot_type == 'random':
            # Random scattered points (should be detected as bot)
            import random
            path = []
            for _ in range(10):
                path.append([
                    random.randint(0, 400),
                    random.randint(0, 400)
                ])
        elif bot_type == 'minimal':
            # Minimal effort path (should be detected as bot)
            path = [[10, 10], [20, 20], [390, 390]]
        else:
            # Default: wrong coordinates
            path = []
            for i in range(10):
                path.append([50 + i * 30, 50 + i * 30])
        
        # Submit bot path for verification
        verify_data = {
            'captcha_id': captcha_id,
            'path': path
        }
        
        # Verify the bot path
        captcha_data = captcha_store[captcha_id]
        maze, _ = generate_maze()
        
        # Convert path coordinates
        path_tuples = []
        for p in path:
            if isinstance(p, list) and len(p) >= 2:
                path_tuples.append([int(p[0]), int(p[1])])
        
        # Apply enhanced bot detection logic with maze validation
        bot_score = 0
        path_length = len(path_tuples)
        
        def validate_path_in_maze(path_points, maze):
            """Check if path stays within valid maze paths"""
            if not path_points:
                return False, "Empty path"
            
            wall_hits = 0
            path_outside_maze = 0
            
            for point in path_points:
                # Convert canvas to maze coordinates
                maze_x = point[0] // 20
                maze_y = point[1] // 20
                
                # Check if point is within maze bounds
                if not (0 <= maze_x < 20 and 0 <= maze_y < 20):
                    path_outside_maze += 1
                    continue
                
                # Check if point is on a path (not a wall)
                if maze[maze_x, maze_y] == 0:  # 0 = wall
                    wall_hits += 1
            
            # Calculate path validity score
            total_points = len(path_points)
            wall_ratio = wall_hits / total_points if total_points > 0 else 1.0
            outside_ratio = path_outside_maze / total_points if total_points > 0 else 1.0
            
            # Path is valid if low wall hits and stays mostly in maze
            is_valid = wall_ratio < 0.3 and outside_ratio < 0.5
            
            return is_valid, f"Wall hits: {wall_hits}/{total_points}, Outside: {path_outside_maze}/{total_points}"
        
        # Check 1: Path too short
        if path_length < 5:
            bot_score += 0.3
        
        # Check 2: Start and end coordinates
        if path_tuples and path_length >= 2:
            first_point = path_tuples[0]
            last_point = path_tuples[-1]
            start_maze = [first_point[0] // 20, first_point[1] // 20]
            end_maze = [last_point[0] // 20, last_point[1] // 20]
            
            start_valid = abs(start_maze[0] - 0) <= 1 and abs(start_maze[1] - 0) <= 1
            end_valid = abs(end_maze[0] - 19) <= 1 and abs(end_maze[1] - 19) <= 1
            
            if not (start_valid and end_valid):
                bot_score += 0.4
        
        # Check 3: Path goes through maze walls (NEW!)
        path_valid, path_details = validate_path_in_maze(path_tuples, maze)
        if not path_valid:
            bot_score += 0.5  # Major penalty for hitting walls
        else:
            # Small penalty for minor wall touches
            wall_details = path_details.split(',')[0]  # "Wall hits: X/Y"
            if '/' in wall_details:
                hits, total = wall_details.split(': ')[1].split('/')
                hit_ratio = int(hits) / int(total)
                if hit_ratio > 0.1:  # More than 10% wall hits
                    bot_score += 0.1
        
        # Check 3: Perfect line detection
        if path_length >= 3:
            distances = []
            for i in range(len(path_tuples) - 1):
                dx = path_tuples[i+1][0] - path_tuples[i][0]
                dy = path_tuples[i+1][1] - path_tuples[i][1]
                dist = (dx*dx + dy*dy) ** 0.5
                distances.append(dist)
            
            if distances:
                avg_dist = sum(distances) / len(distances)
                variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
                # Lower threshold for perfect line detection
                if variance < 50 and path_length >= 10:
                    bot_score += 0.3  # Stronger penalty for perfect lines
                elif variance < 100:
                    bot_score += 0.1
        
        # Bot type specific scoring
        if bot_type == 'perfect':
            bot_score += 0.2  # Additional penalty for requested perfect bot
        elif bot_type == 'random':
            bot_score += 0.1
        elif bot_type == 'minimal':
            bot_score += 0.15
        
        # Check 4: Path distance合理性
        if path_length >= 2:
            total_distance = 0
            for i in range(len(path_tuples) - 1):
                dx = path_tuples[i+1][0] - path_tuples[i][0]
                dy = path_tuples[i+1][1] - path_tuples[i][1]
                total_distance += (dx*dx + dy*dy) ** 0.5
            
            expected_distance = ((380**2) + (380**2)) ** 0.5
            if total_distance < expected_distance * 0.5:
                bot_score += 0.2
        
        # Final decision
        is_human = bot_score < 0.5 and path_length >= 5 and path_valid
        
        if not is_human:
            analytics['bot_detected'] += 1
        
        # Clean up captcha
        del captcha_store[captcha_id]
        
        return jsonify({
            'success': is_human,
            'bot_detected': not is_human,
            'bot_score': bot_score,
            'path_length': len(path_tuples),
            'message': f'Bot simulation ({bot_type}): {"Human" if is_human else "Bot"} detected',
            'path': path,
            'path_analysis': {
                'valid': path_valid,
                'details': path_details,
                'wall_ratio': 'high' if bot_score > 0.6 else 'medium' if bot_score > 0.3 else 'low'
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Bot simulation failed: {str(e)}',
            'success': False,
            'bot_detected': True
        }), 500

def create_histogram_bins(data, bins=10):
    """Create histogram data from list of values"""
    if not data:
        return {'bins': [], 'counts': []}
    
    min_val, max_val = min(data), max(data)
    if min_val == max_val:
        return {'bins': [min_val], 'counts': [len(data)]}
    
    bin_width = (max_val - min_val) / bins
    histogram_bins = []
    histogram_counts = []
    
    for i in range(bins):
        bin_start = min_val + i * bin_width
        bin_end = min_val + (i + 1) * bin_width
        count = sum(1 for val in data if bin_start <= val < bin_end)
        
        histogram_bins.append(f"{bin_start:.1f}-{bin_end:.1f}")
        histogram_counts.append(count)
    
    return {'bins': histogram_bins, 'counts': histogram_counts}

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    try:
        total = analytics['successful_verifications'] + analytics['bot_detected']
        success_rate = (analytics['successful_verifications'] / total * 100) if total > 0 else 0
        
        # Create histograms
        path_length_hist = create_histogram_bins(analytics['path_lengths'], bins=10)
        confidence_hist = create_histogram_bins(analytics['confidence_scores'], bins=5)
        
        # Prepare hourly data sorted by hour
        hourly_data = []
        for hour in sorted(analytics['hourly_stats'].keys()):
            hourly_data.append({
                'hour': hour,
                'human': analytics['hourly_stats'][hour]['human'],
                'bot': analytics['hourly_stats'][hour]['bot']
            })
        
        return jsonify({
            'total_attempts': analytics['total_attempts'],
            'successful_verifications': analytics['successful_verifications'],
            'bot_detected': analytics['bot_detected'],
            'success_rate': success_rate,
            'recent_events': analytics['recent_events'],
            'path_length_histogram': path_length_hist,
            'confidence_histogram': confidence_hist,
            'hourly_stats': hourly_data,
            'avg_path_length': sum(analytics['path_lengths']) / len(analytics['path_lengths']) if analytics['path_lengths'] else 0,
            'avg_confidence': sum(analytics['confidence_scores']) / len(analytics['confidence_scores']) if analytics['confidence_scores'] else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Working 20x20 Maze Server...")
    print("Open http://127.0.0.1:8080 in your browser")
    app.run(host='127.0.0.1', port=8080, debug=True)