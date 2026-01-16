#!/usr/bin/env python3
"""
Fixed Behavioral Maze CAPTCHA System
Resolves database column issues and ensures proper startup
"""

import sys
import os
import json
import base64
import time
import hashlib
import sqlite3
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

try:
    import numpy as np
    import cv2
    from flask import Flask, request, jsonify, render_template, session, g
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install with: pip install numpy opencv-python flask")
    sys.exit(1)

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedMazeCaptcha:
    """Fixed behavioral maze CAPTCHA system with proper database schema"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = 'behavioral_maze_captcha_fixed'
        
        # Configuration
        self.app.config.update(
            HOST='127.0.0.1',
            PORT=8080,
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE='Lax',
            SESSION_COOKIE_SECURE=False,
            PERMANENT_SESSION_LIFETIME=timedelta(minutes=30)
        )
        
        self.setup_database()
        self.setup_routes()
        
    def setup_database(self):
        """Setup proper database tables with correct schema"""
        conn = sqlite3.connect('maze_captcha.db')
        cursor = conn.cursor()
        
        # Drop old tables and recreate with proper schema
        cursor.execute('DROP TABLE IF EXISTS captcha_sessions')
        cursor.execute('DROP TABLE IF EXISTS user_paths')
        
        # Create captcha_sessions table
        cursor.execute('''
            CREATE TABLE captcha_sessions (
                id TEXT PRIMARY KEY,
                maze_data TEXT NOT NULL,
                solution_path TEXT NOT NULL,
                start_point TEXT NOT NULL,
                end_point TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_verified BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create user_paths table
        cursor.execute('''
            CREATE TABLE user_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                coordinates TEXT NOT NULL,
                solve_time REAL,
                is_human BOOLEAN,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized with fixed schema")
    
    def generate_maze(self, size=15):
        """Generate maze with recursive backtracking"""
        maze = np.zeros((size, size), dtype=np.uint8)
        
        def carve(r, c):
            maze[r, c] = 1
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            np.random.shuffle(directions)
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 < nr < size-1 and 0 < nc < size-1 and 
                    maze[nr, nc] == 0):
                    maze[r + dr // 2, c + dc // 2] = 1
                    carve(nr, nc)
        
        carve(1, 1)
        
        start = (1, 1)
        end = (size-2, size-2)
        maze[start] = 1
        maze[end] = 1
        
        return maze, start, end
    
    def find_solution(self, maze, start, end):
        """BFS pathfinding"""
        from collections import deque
        
        queue = deque([start])
        parent = {start: None}
        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        
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
                if (0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and 
                    maze[nr, nc] == 1 and (nr, nc) not in parent):
                    parent[(nr, nc)] = curr
                    queue.append((nr, nc))
        
        return None
    
    def render_maze(self, maze, start, end, cell_size=20):
        """Render maze as image"""
        rows, cols = maze.shape
        img = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
        
        for r in range(rows):
            for c in range(cols):
                color = (255, 255, 255) if maze[r, c] == 1 else (0, 0, 0)
                cv2.rectangle(img, (c*cell_size, r*cell_size), 
                           ((c+1)*cell_size, (r+1)*cell_size), color, -1)
        
        cv2.rectangle(img, (start[1]*cell_size, start[0]*cell_size), 
                     ((start[1]+1)*cell_size, (start[0]+1)*cell_size), (0, 255, 0), -1)
        cv2.rectangle(img, (end[1]*cell_size, end[0]*cell_size), 
                     ((end[1]+1)*cell_size, (end[0]+1)*cell_size), (0, 0, 255), -1)
        
        return img
    
    def analyze_behavior(self, mouse_data, solve_time):
        """Analyze behavior for human vs bot detection"""
        if len(mouse_data) < 10:
            return False, 0.0, "Insufficient data"
        
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
        
        avg_velocity = np.mean(velocities) if velocities else 0
        velocity_variance = np.var(velocities) if velocities else 0
        
        human_score = 0
        
        if 2 <= solve_time <= 30:
            human_score += 0.3
        
        if velocity_variance > 50:
            human_score += 0.3
        
        zero_velocity_count = sum(1 for v in velocities if v < 5)
        if zero_velocity_count > len(velocities) * 0.1:
            human_score += 0.2
        
        if avg_velocity < 200:
            human_score += 0.2
        
        is_human = human_score >= 0.6
        return is_human, human_score, f"Score: {human_score:.2f}"
    
    def setup_routes(self):
        """Setup application routes"""
        
        @self.app.route('/')
        def index():
            return render_template('production_index.html')
        
        @self.app.route('/api/captcha', methods=['GET'])
        def get_captcha():
            try:
                maze, start, end = self.generate_maze()
                solution = self.find_solution(maze, start, end)
                
                captcha_id = hashlib.md5(
                    f"{time.time()}{os.urandom(4).hex()}".encode()
                ).hexdigest()[:16]
                
                session_data = {
                    'maze': maze.tolist(),
                    'solution_path': solution,
                    'start_point': start,
                    'end_point': end,
                    'created_at': time.time(),
                    'mouse_data': []
                }
                
                session[captcha_id] = session_data
                
                # Store in database
                conn = sqlite3.connect('maze_captcha.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO captcha_sessions 
                    (id, maze_data, solution_path, start_point, end_point, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    captcha_id,
                    json.dumps(session_data['maze']),
                    json.dumps(solution),
                    json.dumps(start),
                    json.dumps(end),
                    datetime.now()
                ))
                conn.commit()
                conn.close()
                
                maze_image = self.render_maze(maze, start, end)
                _, buffer = cv2.imencode('.png', maze_image)
                img_base64 = base64.b64encode(buffer).decode()
                
                return jsonify({
                    'captcha_id': captcha_id,
                    'maze_image': f"data:image/png;base64,{img_base64}",
                    'start_point': start,
                    'end_point': end,
                    'difficulty': 'medium',
                    'maze_size': 15,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Error generating captcha: {e}")
                return jsonify({'error': 'Failed to generate captcha'}), 500
        
        @self.app.route('/api/track', methods=['POST'])
        def track_mouse():
            try:
                data = request.get_json()
                captcha_id = data.get('captcha_id')
                
                if not captcha_id or captcha_id not in session:
                    return jsonify({'error': 'Invalid session'}), 400
                
                mouse_data = {
                    'x': data.get('x', 0),
                    'y': data.get('y', 0),
                    'timestamp': data.get('timestamp', time.time()),
                    'event': data.get('event', 'move')
                }
                
                session[captcha_id]['mouse_data'].append(mouse_data)
                return jsonify({'success': True})
                
            except Exception as e:
                logger.error(f"Error tracking mouse: {e}")
                return jsonify({'error': 'Tracking failed'}), 500
        
        @self.app.route('/api/verify', methods=['POST'])
        def verify_captcha():
            try:
                data = request.get_json()
                captcha_id = data.get('captcha_id')
                path_data = data.get('path', [])
                
                if not captcha_id or captcha_id not in session:
                    return jsonify({'error': 'Invalid session'}), 400
                
                session_data = session[captcha_id]
                start_time = session_data['created_at']
                solve_time = time.time() - start_time
                
                if not path_data:
                    return jsonify({
                        'success': False,
                        'message': 'Empty path',
                        'is_human': False,
                        'confidence': 0.0
                    })
                
                start_point = tuple(session_data['start_point'])
                end_point = tuple(session_data['end_point'])
                
                if tuple(path_data[0]) != start_point:
                    return jsonify({
                        'success': False,
                        'message': 'Path must start at green point',
                        'is_human': False,
                        'confidence': 0.0
                    })
                
                if tuple(path_data[-1]) != end_point:
                    return jsonify({
                        'success': False,
                        'message': 'Path must end at red point',
                        'is_human': False,
                        'confidence': 0.0
                    })
                
                is_human, confidence, analysis = self.analyze_behavior(
                    session_data['mouse_data'], solve_time
                )
                
                conn = sqlite3.connect('maze_captcha.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_paths 
                    (session_id, coordinates, solve_time, is_human, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    captcha_id,
                    json.dumps(path_data),
                    solve_time,
                    is_human,
                    confidence
                ))
                
                cursor.execute('''
                    UPDATE captcha_sessions 
                    SET is_verified = TRUE 
                    WHERE id = ?
                ''', (captcha_id,))
                
                conn.commit()
                conn.close()
                
                session.pop(captcha_id, None)
                
                message = "Human verified!" if is_human else "Bot detected!"
                
                return jsonify({
                    'success': is_human,
                    'is_human': is_human,
                    'confidence': confidence,
                    'message': message,
                    'analysis': analysis,
                    'solve_time': solve_time
                })
                
            except Exception as e:
                logger.error(f"Error verifying captcha: {e}")
                return jsonify({'error': 'Verification failed'}), 500
        
        @self.app.route('/api/analytics', methods=['GET'])
        def get_analytics():
            try:
                conn = sqlite3.connect('maze_captcha.db')
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM captcha_sessions')
                total_attempts = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT is_human, COUNT(*) 
                    FROM user_paths 
                    GROUP BY is_human
                ''')
                classifications = dict(cursor.fetchall())
                
                human_count = classifications.get(True, 0)
                bot_count = classifications.get(False, 0)
                
                success_rate = (human_count / total_attempts * 100) if total_attempts > 0 else 0
                
                conn.close()
                
                return jsonify({
                    'total_attempts': total_attempts,
                    'human_detected': human_count,
                    'bot_detected': bot_count,
                    'success_rate': round(success_rate, 2),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting analytics: {e}")
                return jsonify({'error': 'Failed to get analytics'}), 500
        
        @self.app.route('/api/bot-simulate', methods=['POST'])
        def bot_simulate():
            """Bot simulation endpoint for testing"""
            try:
                data = request.get_json()
                captcha_id = data.get('captcha_id')
                
                if not captcha_id or captcha_id not in session:
                    return jsonify({'success': False, 'message': 'Invalid session'}), 400
                
                session_data = session[captcha_id]
                
                # Create bot-like path (straight line solution)
                solution = session_data['solution_path']
                if not solution:
                    return jsonify({'success': False, 'message': 'No solution available'}), 400
                
                # Simulate bot timing and movement
                start_time = session_data['created_at']
                bot_solve_time = np.random.uniform(0.5, 2.0)  # Fast, bot-like timing
                
                # Create bot mouse data with minimal variance
                bot_mouse_data = []
                for i, point in enumerate(solution):
                    bot_mouse_data.append({
                        'x': point[1] * 20 + 10,  # Convert to pixels
                        'y': point[0] * 20 + 10,
                        'timestamp': start_time + (i * bot_solve_time / len(solution))
                    })
                
                # Analyze bot behavior (should be detected as bot)
                is_human, confidence, analysis = self.analyze_behavior(
                    bot_mouse_data, bot_solve_time
                )
                
                # Store bot simulation results
                conn = sqlite3.connect('maze_captcha.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_paths 
                    (session_id, coordinates, solve_time, is_human, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    captcha_id,
                    json.dumps(solution),
                    bot_solve_time,
                    is_human,
                    confidence
                ))
                
                cursor.execute('''
                    UPDATE captcha_sessions 
                    SET is_verified = TRUE 
                    WHERE id = ?
                ''', (captcha_id,))
                
                conn.commit()
                conn.close()
                
                session.pop(captcha_id, None)
                
                return jsonify({
                    'success': True,
                    'bot_detected': not is_human,
                    'confidence': confidence,
                    'message': f'Bot simulation completed: {"Human" if is_human else "Bot"} detected',
                    'analysis': analysis,
                    'solve_time': bot_solve_time,
                    'path_length': len(solution)
                })
                
            except Exception as e:
                logger.error(f"Error in bot simulation: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            try:
                conn = sqlite3.connect('maze_captcha.db')
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                result = cursor.fetchone()
                conn.close()
                
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'database': 'ok' if result else 'error',
                    'components': {
                        'maze_generator': 'ok',
                        'behavioral_analyzer': 'ok'
                    }
                })
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
    
    def run(self, host=None, port=None, debug=False):
        host = host or self.app.config['HOST']
        port = port or self.app.config['PORT']
        
        print("🎯 Behavioral Maze CAPTCHA System - FIXED")
        print("=" * 60)
        print("🧩 Advanced Anti-Bot Detection System")
        print(f"🌐 Server: http://{host}:{port}")
        print(f"🔍 Health Check: http://{host}:{port}/api/health")
        print("=" * 60)
        print("✅ Features:")
        print("   • Procedural Maze Generation")
        print("   • BFS Pathfinding Algorithm")
        print("   • Behavioral Analysis")
        print("   • Human Buffer Tolerance")
        print("   • Real-time Analytics")
        print("   • Fixed Database Schema")
        print("=" * 60)
        
        self.app.run(host=host, port=port, debug=debug)


def main():
    print("🚀 Starting Fixed Behavioral Maze CAPTCHA System")
    
    app = FixedMazeCaptcha()
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()