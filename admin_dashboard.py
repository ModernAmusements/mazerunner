#!/usr/bin/env python3
"""
Admin Dashboard for Behavioral Maze CAPTCHA System
Features path replay, analytics visualization, and system monitoring
"""

import json
import sqlite3
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from flask import Flask, render_template, jsonify, request, session
import logging

logger = logging.getLogger(__name__)

class AdminDashboard:
    """Comprehensive admin dashboard with analytics and path replay"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.setup_routes()
        
    def setup_routes(self):
        """Setup admin dashboard routes"""
        
        @self.app.route('/admin')
        def admin_dashboard():
            """Main admin dashboard"""
            if not self._is_admin():
                return jsonify({'error': 'Unauthorized'}), 403
            
            return render_template('admin_dashboard.html')
        
        @self.app.route('/api/admin/stats')
        def get_admin_stats():
            """Get comprehensive system statistics"""
            if not self._is_admin():
                return jsonify({'error': 'Unauthorized'}), 403
            
            try:
                stats = self._calculate_system_stats()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting admin stats: {str(e)}")
                return jsonify({'error': 'Failed to retrieve statistics'}), 500
        
        @self.app.route('/api/admin/recent-sessions')
        def get_recent_sessions():
            """Get recent captcha sessions with detailed analytics"""
            if not self._is_admin():
                return jsonify({'error': 'Unauthorized'}), 403
            
            try:
                limit = request.args.get('limit', 50, type=int)
                sessions = self._get_recent_sessions(limit)
                return jsonify(sessions)
            except Exception as e:
                logger.error(f"Error getting recent sessions: {str(e)}")
                return jsonify({'error': 'Failed to retrieve sessions'}), 500
        
        @self.app.route('/api/admin/session/<session_id>')
        def get_session_details(session_id):
            """Get detailed information about a specific session"""
            if not self._is_admin():
                return jsonify({'error': 'Unauthorized'}), 403
            
            try:
                details = self._get_session_details(session_id)
                if not details:
                    return jsonify({'error': 'Session not found'}), 404
                
                return jsonify(details)
            except Exception as e:
                logger.error(f"Error getting session details: {str(e)}")
                return jsonify({'error': 'Failed to retrieve session details'}), 500
        
        @self.app.route('/api/admin/path-replay/<session_id>')
        def get_path_replay_data(session_id):
            """Get path replay data for visualization"""
            if not self._is_admin():
                return jsonify({'error': 'Unauthorized'}), 403
            
            try:
                replay_data = self._get_path_replay_data(session_id)
                if not replay_data:
                    return jsonify({'error': 'No path data found'}), 404
                
                return jsonify(replay_data)
            except Exception as e:
                logger.error(f"Error getting path replay data: {str(e)}")
                return jsonify({'error': 'Failed to retrieve replay data'}), 500
        
        @self.app.route('/api/admin/bot-indicators')
        def get_bot_indicators():
            """Get recent bot detection indicators"""
            if not self._is_admin():
                return jsonify({'error': 'Unauthorized'}), 403
            
            try:
                indicators = self._get_bot_indicators()
                return jsonify(indicators)
            except Exception as e:
                logger.error(f"Error getting bot indicators: {str(e)}")
                return jsonify({'error': 'Failed to retrieve indicators'}), 500
        
        @self.app.route('/api/admin/human-patterns')
        def get_human_patterns():
            """Get learned human behavioral patterns"""
            if not self._is_admin():
                return jsonify({'error': 'Unauthorized'}), 403
            
            try:
                patterns = self._get_human_patterns()
                return jsonify(patterns)
            except Exception as e:
                logger.error(f"Error getting human patterns: {str(e)}")
                return jsonify({'error': 'Failed to retrieve patterns'}), 500
    
    def _is_admin(self) -> bool:
        """Check if current user has admin privileges"""
        # Simple admin check - in production, use proper authentication
        return session.get('is_admin', False) or request.remote_addr in ['127.0.0.1', '::1']
    
    def _calculate_system_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive system statistics"""
        conn = sqlite3.connect('maze_captcha.db')
        cursor = conn.cursor()
        
        try:
            # Total captcha attempts
            cursor.execute("SELECT COUNT(*) FROM captcha_sessions")
            total_attempts = cursor.fetchone()[0]
            
            # Human vs Bot classification
            cursor.execute("""
                SELECT is_human, COUNT(*) 
                FROM user_paths 
                GROUP BY is_human
            """)
            classifications = dict(cursor.fetchall())
            
            human_count = classifications.get(True, 0)
            bot_count = classifications.get(False, 0)
            uncertain_count = classifications.get(None, 0)
            
            # Average solve time
            cursor.execute("""
                SELECT AVG(solve_time) 
                FROM user_paths 
                WHERE solve_time IS NOT NULL
            """)
            avg_solve_time = cursor.fetchone()[0] or 0
            
            # Success rate
            cursor.execute("""
                SELECT COUNT(*) FROM user_paths WHERE is_human = TRUE
            """)
            successful_attempts = cursor.fetchone()[0]
            
            success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
            
            # Recent activity (last 24 hours)
            yesterday = datetime.now() - timedelta(days=1)
            cursor.execute("""
                SELECT COUNT(*) FROM captcha_sessions 
                WHERE created_at > ?
            """, (yesterday,))
            recent_activity = cursor.fetchone()[0]
            
            # Bot detection accuracy
            cursor.execute("""
                SELECT confidence_score, AVG(confidence_score) as avg_confidence
                FROM user_paths 
                WHERE confidence_score IS NOT NULL
            """)
            confidence_data = cursor.fetchone()
            avg_confidence = confidence_data[1] or 0
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size_bytes = cursor.fetchone()[0]
            db_size_mb = db_size_bytes / (1024 * 1024)
            
            return {
                'total_attempts': total_attempts,
                'human_detected': human_count,
                'bot_detected': bot_count,
                'uncertain_detected': uncertain_count,
                'success_rate': round(success_rate, 2),
                'avg_solve_time': round(avg_solve_time, 2),
                'recent_activity_24h': recent_activity,
                'avg_confidence_score': round(avg_confidence, 2),
                'database_size_mb': round(db_size_mb, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            conn.close()
    
    def _get_recent_sessions(self, limit: int = 50) -> List[Dict]:
        """Get recent captcha sessions with analytics"""
        conn = sqlite3.connect('maze_captcha.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    cs.id,
                    cs.created_at,
                    cs.is_verified,
                    up.is_human,
                    up.confidence_score,
                    up.solve_time,
                    up.velocity_variance,
                    up.direction_changes,
                    up.jitter_magnitude
                FROM captcha_sessions cs
                LEFT JOIN user_paths up ON cs.id = up.session_id
                ORDER BY cs.created_at DESC
                LIMIT ?
            """, (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                session_data = {
                    'session_id': row[0],
                    'created_at': row[1],
                    'is_verified': bool(row[2]),
                    'is_human': bool(row[3]) if row[3] is not None else None,
                    'confidence_score': row[4],
                    'solve_time': row[5],
                    'velocity_variance': row[6],
                    'direction_changes': row[7],
                    'jitter_magnitude': row[8]
                }
                sessions.append(session_data)
            
            return sessions
            
        finally:
            conn.close()
    
    def _get_session_details(self, session_id: str) -> Optional[Dict]:
        """Get detailed information about a specific session"""
        conn = sqlite3.connect('maze_captcha.db')
        cursor = conn.cursor()
        
        try:
            # Get session data
            cursor.execute("""
                SELECT id, maze_data, solution_path, start_point, end_point, 
                       created_at, expires_at, is_verified
                FROM captcha_sessions 
                WHERE id = ?
            """, (session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                return None
            
            # Get user path data
            cursor.execute("""
                SELECT coordinates, timestamp_data, velocity_data, solve_time,
                       is_human, confidence_score, created_at
                FROM user_paths 
                WHERE session_id = ?
            """, (session_id,))
            
            path_row = cursor.fetchone()
            
            # Get bot indicators
            cursor.execute("""
                SELECT indicator_type, value, threshold, is_flagged
                FROM bot_indicators 
                WHERE session_id = ?
            """, (session_id,))
            
            indicators = []
            for ind_row in cursor.fetchall():
                indicators.append({
                    'type': ind_row[0],
                    'value': ind_row[1],
                    'threshold': ind_row[2],
                    'flagged': bool(ind_row[3])
                })
            
            return {
                'session_id': session_id,
                'maze_data': session_row[1],
                'solution_path': session_row[2],
                'start_point': session_row[3],
                'end_point': session_row[4],
                'created_at': session_row[5],
                'expires_at': session_row[6],
                'is_verified': bool(session_row[7]),
                'user_path': {
                    'coordinates': path_row[0] if path_row else None,
                    'timestamp_data': path_row[1] if path_row else None,
                    'velocity_data': path_row[2] if path_row else None,
                    'solve_time': path_row[3] if path_row else None,
                    'is_human': bool(path_row[4]) if path_row and path_row[4] is not None else None,
                    'confidence_score': path_row[5] if path_row else None,
                    'created_at': path_row[6] if path_row else None
                },
                'bot_indicators': indicators
            }
            
        finally:
            conn.close()
    
    def _get_path_replay_data(self, session_id: str) -> Optional[Dict]:
        """Get path replay data for visualization"""
        conn = sqlite3.connect('maze_captcha.db')
        cursor = conn.cursor()
        
        try:
            # Get session and path data
            cursor.execute("""
                SELECT cs.maze_data, cs.start_point, cs.end_point,
                       up.coordinates, up.timestamp_data, up.velocity_data,
                       up.is_human, up.confidence_score
                FROM captcha_sessions cs
                LEFT JOIN user_paths up ON cs.id = up.session_id
                WHERE cs.id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            maze_data = json.loads(row[0]) if row[0] else None
            start_point = json.loads(row[1]) if row[1] else None
            end_point = json.loads(row[2]) if row[2] else None
            coordinates = json.loads(row[3]) if row[3] else []
            timestamp_data = json.loads(row[4]) if row[4] else []
            velocity_data = json.loads(row[5]) if row[5] else []
            
            # Process coordinates for replay
            replay_points = []
            for i, coord in enumerate(coordinates):
                point = {
                    'x': coord[0] if isinstance(coord, list) else coord.get('x', 0),
                    'y': coord[1] if isinstance(coord, list) else coord.get('y', 0),
                    'timestamp': timestamp_data[i] if i < len(timestamp_data) else 0,
                    'velocity': velocity_data[i] if i < len(velocity_data) else 0
                }
                replay_points.append(point)
            
            return {
                'session_id': session_id,
                'maze_data': maze_data,
                'start_point': start_point,
                'end_point': end_point,
                'replay_points': replay_points,
                'is_human': bool(row[6]) if row[6] is not None else None,
                'confidence_score': row[7],
                'frame_count': len(replay_points)
            }
            
        finally:
            conn.close()
    
    def _get_bot_indicators(self, limit: int = 100) -> List[Dict]:
        """Get recent bot detection indicators"""
        conn = sqlite3.connect('maze_captcha.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT bi.session_id, bi.indicator_type, bi.value, 
                       bi.threshold, bi.is_flagged, bi.created_at,
                       cs.created_at as session_created
                FROM bot_indicators bi
                JOIN captcha_sessions cs ON bi.session_id = cs.id
                ORDER BY bi.created_at DESC
                LIMIT ?
            """, (limit,))
            
            indicators = []
            for row in cursor.fetchall():
                indicators.append({
                    'session_id': row[0],
                    'indicator_type': row[1],
                    'value': row[2],
                    'threshold': row[3],
                    'is_flagged': bool(row[4]),
                    'created_at': row[5],
                    'session_created_at': row[6]
                })
            
            return indicators
            
        finally:
            conn.close()
    
    def _get_human_patterns(self, limit: int = 50) -> List[Dict]:
        """Get learned human behavioral patterns"""
        conn = sqlite3.connect('maze_captcha.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT data, timestamp
                FROM analytics 
                WHERE event_type = 'human_pattern'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            patterns = []
            for row in cursor.fetchall():
                try:
                    pattern_data = json.loads(row[0])
                    patterns.append({
                        'data': pattern_data,
                        'timestamp': row[1]
                    })
                except json.JSONDecodeError:
                    continue
            
            return patterns
            
        finally:
            conn.close()

# Admin Dashboard Template
ADMIN_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Maze CAPTCHA</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #ffffff; }
        .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-bottom: 1px solid #333; }
        .header h1 { font-size: 28px; font-weight: 600; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; padding: 20px; }
        .stat-card { background: #1a1a1a; border-radius: 12px; padding: 20px; border: 1px solid #333; }
        .stat-value { font-size: 32px; font-weight: 700; color: #4CAF50; margin-bottom: 8px; }
        .stat-label { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
        .bot-value { color: #f44336; }
        .uncertain-value { color: #ff9800; }
        .content-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }
        .panel { background: #1a1a1a; border-radius: 12px; border: 1px solid #333; overflow: hidden; }
        .panel-header { background: #2a2a2a; padding: 15px 20px; font-weight: 600; border-bottom: 1px solid #333; }
        .panel-content { padding: 20px; max-height: 400px; overflow-y: auto; }
        .session-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #333; }
        .session-item:last-child { border-bottom: none; }
        .session-id { font-family: monospace; font-size: 12px; color: #888; }
        .session-status { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }
        .status-human { background: #4CAF50; color: white; }
        .status-bot { background: #f44336; color: white; }
        .status-uncertain { background: #ff9800; color: white; }
        .replay-canvas { width: 100%; height: 300px; background: #000; border-radius: 8px; margin-top: 10px; }
        .controls { display: flex; gap: 10px; margin-top: 15px; }
        .btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: 500; transition: all 0.2s; }
        .btn-primary { background: #2196F3; color: white; }
        .btn-primary:hover { background: #1976D2; }
        .btn-secondary { background: #666; color: white; }
        .btn-secondary:hover { background: #555; }
        .chart-container { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Behavioral Maze CAPTCHA - Admin Dashboard</h1>
    </div>
    
    <div class="stats-grid" id="statsGrid">
        <!-- Stats will be populated by JavaScript -->
    </div>
    
    <div class="content-grid">
        <div class="panel">
            <div class="panel-header">Recent Sessions</div>
            <div class="panel-content" id="recentSessions">
                <!-- Sessions will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-header">Path Replay</div>
            <div class="panel-content">
                <select id="sessionSelector" class="btn btn-secondary" style="width: 100%; margin-bottom: 10px;">
                    <option value="">Select a session to replay</option>
                </select>
                <canvas id="replayCanvas" class="replay-canvas"></canvas>
                <div class="controls">
                    <button id="playReplay" class="btn btn-primary">▶️ Play</button>
                    <button id="pauseReplay" class="btn btn-secondary">⏸️ Pause</button>
                    <button id="resetReplay" class="btn btn-secondary">⏹️ Reset</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="panel" style="margin: 20px;">
        <div class="panel-header">Analytics Overview</div>
        <div class="panel-content">
            <div class="chart-container">
                <canvas id="analyticsChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        let replayData = null;
        let replayIndex = 0;
        let isPlaying = false;
        let animationId = null;
        
        // Load dashboard data
        async function loadDashboard() {
            try {
                const [stats, sessions] = await Promise.all([
                    fetch('/api/admin/stats').then(r => r.json()),
                    fetch('/api/admin/recent-sessions').then(r => r.json())
                ]);
                
                updateStats(stats);
                updateRecentSessions(sessions);
                await loadAnalyticsChart();
            } catch (error) {
                console.error('Error loading dashboard:', error);
            }
        }
        
        function updateStats(stats) {
            const statsGrid = document.getElementById('statsGrid');
            statsGrid.innerHTML = \`
                <div class="stat-card">
                    <div class="stat-value">\${stats.total_attempts}</div>
                    <div class="stat-label">Total Attempts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">\${stats.human_detected}</div>
                    <div class="stat-label">Humans Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value bot-value">\${stats.bot_detected}</div>
                    <div class="stat-label">Bots Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value uncertain-value">\${stats.uncertain_detected || 0}</div>
                    <div class="stat-label">Uncertain</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">\${stats.success_rate}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">\${stats.avg_solve_time}s</div>
                    <div class="stat-label">Avg Solve Time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">\${stats.recent_activity_24h}</div>
                    <div class="stat-label">Last 24h Activity</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">\${stats.avg_confidence_score}</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            \`;
        }
        
        function updateRecentSessions(sessions) {
            const container = document.getElementById('recentSessions');
            const selector = document.getElementById('sessionSelector');
            
            container.innerHTML = sessions.map(session => \`
                <div class="session-item">
                    <div>
                        <div class="session-id">\${session.session_id}</div>
                        <small>\${new Date(session.created_at).toLocaleString()}</small>
                    </div>
                    <div class="session-status status-\${session.is_human ? 'human' : session.is_human === false ? 'bot' : 'uncertain'}">
                        \${session.is_human ? 'Human' : session.is_human === false ? 'Bot' : 'Uncertain'}
                    </div>
                </div>
            \`).join('');
            
            selector.innerHTML = '<option value="">Select a session to replay</option>' +
                sessions.map(session => \`<option value="\${session.session_id}">\${session.session_id.substring(0, 8)}...</option>\`).join('');
        }
        
        async function loadReplayData(sessionId) {
            if (!sessionId) {
                replayData = null;
                return;
            }
            
            try {
                replayData = await fetch(\`/api/admin/path-replay/\${sessionId}\`).then(r => r.json());
                resetReplay();
            } catch (error) {
                console.error('Error loading replay data:', error);
            }
        }
        
        function drawReplay() {
            if (!replayData) return;
            
            const canvas = document.getElementById('replayCanvas');
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw maze (simplified)
            if (replayData.maze_data) {
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 1;
                // Draw maze grid (simplified representation)
            }
            
            // Draw start and end points
            ctx.fillStyle = '#4CAF50';
            ctx.fillRect(10, 10, 20, 20);
            ctx.fillStyle = '#f44336';
            ctx.fillRect(canvas.width - 30, canvas.height - 30, 20, 20);
            
            // Draw path up to current index
            if (replayData.replay_points && replayIndex > 0) {
                ctx.strokeStyle = replayData.is_human ? '#4CAF50' : '#f44336';
                ctx.lineWidth = 2;
                ctx.beginPath();
                
                for (let i = 0; i <= replayIndex && i < replayData.replay_points.length; i++) {
                    const point = replayData.replay_points[i];
                    if (i === 0) {
                        ctx.moveTo(point.x, point.y);
                    } else {
                        ctx.lineTo(point.x, point.y);
                    }
                }
                ctx.stroke();
                
                // Draw current position
                if (replayIndex < replayData.replay_points.length) {
                    const currentPoint = replayData.replay_points[replayIndex];
                    ctx.fillStyle = '#fff';
                    ctx.beginPath();
                    ctx.arc(currentPoint.x, currentPoint.y, 5, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }
        
        function playReplay() {
            if (!replayData || isPlaying) return;
            isPlaying = true;
            
            function animate() {
                if (!isPlaying) {
                    cancelAnimationFrame(animationId);
                    return;
                }
                
                if (replayIndex < replayData.replay_points.length - 1) {
                    replayIndex++;
                    drawReplay();
                    animationId = requestAnimationFrame(animate);
                } else {
                    isPlaying = false;
                }
            }
            
            animate();
        }
        
        function pauseReplay() {
            isPlaying = false;
        }
        
        function resetReplay() {
            isPlaying = false;
            replayIndex = 0;
            drawReplay();
        }
        
        async function loadAnalyticsChart() {
            const indicators = await fetch('/api/admin/bot-indicators').then(r => r.json());
            
            const ctx = document.getElementById('analyticsChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: indicators.map(i => new Date(i.created_at).toLocaleTimeString()),
                    datasets: [{
                        label: 'Bot Indicator Values',
                        data: indicators.map(i => i.value),
                        borderColor: '#f44336',
                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        }
        
        // Event listeners
        document.getElementById('sessionSelector').addEventListener('change', (e) => {
            loadReplayData(e.target.value);
        });
        
        document.getElementById('playReplay').addEventListener('click', playReplay);
        document.getElementById('pauseReplay').addEventListener('click', pauseReplay);
        document.getElementById('resetReplay').addEventListener('click', resetReplay);
        
        // Initialize dashboard
        loadDashboard();
    </script>
</body>
</html>
"""

# Template writer function
def create_admin_template():
    """Create admin dashboard template file"""
    with open('/Users/modernamusmenet/Desktop/BOT/templates/admin_dashboard.html', 'w') as f:
        f.write(ADMIN_DASHBOARD_TEMPLATE)

if __name__ == "__main__":
    create_admin_template()
    print("✅ Admin dashboard template created successfully!")