#!/usr/bin/env python3
"""
Database session persistence for Maze Captcha
"""

import sqlite3
import json
import time
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager

class DatabaseSessionManager:
    def __init__(self, db_path='maze_captcha.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT
                )
            ''')
            
            # Captcha sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS captcha_sessions (
                    captcha_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    captcha_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Rate limiting table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rate_limits (
                    ip_address TEXT PRIMARY KEY,
                    requests_data TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Human patterns table for persistence
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS human_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with thread safety"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def create_session(self, session_id, session_data, expires_minutes=30, ip_address=None, user_agent=None):
        """Create a new session"""
        expires_at = datetime.now() + timedelta(minutes=expires_minutes)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, data, expires_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, json.dumps(session_data), expires_at, ip_address, user_agent))
            conn.commit()
    
    def get_session(self, session_id):
        """Get session data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT data, expires_at FROM sessions 
                WHERE session_id = ? AND expires_at > datetime('now')
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row['data'])
            return None
    
    def update_session(self, session_id, session_data, expires_minutes=None):
        """Update session data"""
        expires_at = datetime.now() + timedelta(minutes=expires_minutes or 30) if expires_minutes else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if expires_at:
                cursor.execute('''
                    UPDATE sessions SET data = ?, expires_at = ?
                    WHERE session_id = ?
                ''', (json.dumps(session_data), expires_at, session_id))
            else:
                cursor.execute('''
                    UPDATE sessions SET data = ?
                    WHERE session_id = ?
                ''', (json.dumps(session_data), session_id))
            conn.commit()
    
    def delete_session(self, session_id):
        """Delete a session and its associated data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM captcha_sessions WHERE session_id = ?', (session_id,))
            conn.commit()
    
    def store_captcha(self, captcha_id, session_id, captcha_data, expires_minutes=5):
        """Store captcha data"""
        expires_at = datetime.now() + timedelta(minutes=expires_minutes)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO captcha_sessions 
                (captcha_id, session_id, captcha_data, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (captcha_id, session_id, json.dumps(captcha_data), expires_at))
            conn.commit()
    
    def get_captcha(self, captcha_id):
        """Get captcha data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT captcha_data FROM captcha_sessions 
                WHERE captcha_id = ? AND expires_at > datetime('now')
            ''', (captcha_id,))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row['captcha_data'])
            return None
    
    def delete_captcha(self, captcha_id):
        """Delete captcha data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM captcha_sessions WHERE captcha_id = ?', (captcha_id,))
            conn.commit()
    
    def log_analytics(self, session_id, event_type, data=None, ip_address=None):
        """Log analytics event"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analytics (session_id, event_type, data, ip_address)
                VALUES (?, ?, ?, ?)
            ''', (session_id, event_type, json.dumps(data) if data else None, ip_address))
            conn.commit()
    
    def get_analytics_summary(self, hours=24):
        """Get analytics summary for the last N hours"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    event_type,
                    COUNT(*) as count,
                    data
                FROM analytics 
                WHERE timestamp > datetime('now', '-{} hours')
                GROUP BY event_type
                ORDER BY count DESC
            '''.format(hours))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def store_human_pattern(self, pattern_data, session_id=None):
        """Store learned human pattern"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO human_patterns (pattern_data, session_id)
                VALUES (?, ?)
            ''', (json.dumps(pattern_data), session_id))
            conn.commit()
    
    def get_human_patterns(self, limit=None):
        """Get stored human patterns"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT pattern_data FROM human_patterns ORDER BY created_at DESC'
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query)
            patterns = []
            for row in cursor.fetchall():
                try:
                    patterns.append(json.loads(row['pattern_data']))
                except:
                    continue
            return patterns
    
    def cleanup_expired(self):
        """Clean up expired sessions and captcha data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE expires_at <= datetime("now")')
            cursor.execute('DELETE FROM captcha_sessions WHERE expires_at <= datetime("now")')
            deleted = cursor.rowcount
            conn.commit()
            return deleted
    
    def get_database_stats(self):
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            for table in ['sessions', 'captcha_sessions', 'analytics', 'human_patterns']:
                cursor.execute(f'SELECT COUNT(*) as count FROM {table}')
                stats[table] = cursor.fetchone()['count']
            
            return stats

# Global database session manager
db_session_manager = DatabaseSessionManager()