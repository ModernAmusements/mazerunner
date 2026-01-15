#!/usr/bin/env python3
"""
Fixed version of bot simulation with proper maze validation
"""

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

def create_bot_simulation_response():
    return '''        # Apply enhanced bot detection logic with maze validation
        bot_score = 0
        path_length = len(path_tuples)
        
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
        
        # Check 4: Perfect line detection
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
        
        # Check 5: Path distance合理性
        if path_length >= 2:
            total_distance = 0
            for i in range(len(path_tuples) - 1):
                dx = path_tuples[i+1][0] - path_tuples[i][0]
                dy = path_tuples[i+1][1] - path_tuples[i][1]
                total_distance += (dx*dx + dy*dy) ** 0.5
            
            expected_distance = ((380**2) + (380**2)) ** 0.5
            if total_distance < expected_distance * 0.5:
                bot_score += 0.2
        
        # Bot type specific scoring
        if bot_type == 'perfect':
            bot_score += 0.2  # Additional penalty for requested perfect bot
        elif bot_type == 'random':
            bot_score += 0.1
        elif bot_type == 'minimal':
            bot_score += 0.15
        
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
        })'''

if __name__ == "__main__":
    print("Bot simulation enhancement code")
    print("Key features:")
    print("1. Maze wall collision detection")
    print("2. Path boundary validation") 
    print("3. Enhanced bot scoring")
    print("4. Detailed path analysis")
    print("\\nPaste this into working_maze_server.py to replace bot_simulate function")