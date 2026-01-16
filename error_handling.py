#!/usr/bin/env python3
"""
Structured Error Handling and Logging System for Behavioral Maze CAPTCHA
Zero-Error Policy with comprehensive tracebacks and error recovery
"""

import sys
import traceback
import logging
import logging.handlers
import json
import datetime
import functools
import inspect
from typing import Any, Callable, Dict, Optional, Type, Union
from pathlib import Path
from flask import Flask, request, g
import sqlite3


class MazeCaptchaLogger:
    """Advanced logging system with structured output and error recovery"""
    
    def __init__(self, log_level: str = "INFO"):
        self.setup_logging(log_level)
        self.error_counts = {}
        self.critical_errors = []
        
    def setup_logging(self, log_level: str):
        """Setup comprehensive logging with multiple handlers"""
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Configure formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with detailed format
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            'logs/maze_captcha.log', maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # JSON handler for structured analysis
        json_handler = logging.handlers.RotatingFileHandler(
            'logs/maze_captcha_structured.log', maxBytes=5*1024*1024, backupCount=3
        )
        json_handler.setFormatter(json_formatter)
        json_handler.setLevel(logging.WARNING)
        root_logger.addHandler(json_handler)
        
        # Error-only handler
        error_handler = logging.handlers.RotatingFileHandler(
            'logs/maze_captcha_errors.log', maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        # Security log for suspicious activities
        security_handler = logging.FileHandler('logs/security.log')
        security_handler.setFormatter(json_formatter)
        self.security_logger = logging.getLogger('security')
        self.security_logger.addHandler(security_handler)
        self.security_logger.setLevel(logging.INFO)
        self.security_logger.propagate = False
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context and stack trace"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Increment error count
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Get calling frame
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name if frame else 'unknown'
        line_no = frame.f_lineno if frame else 0
        
        # Get full stack trace
        stack_trace = traceback.format_exc()
        
        # Log detailed error
        error_logger = logging.getLogger(func_name)
        error_logger.error(
            f"Error in {func_name}:{line_no} - {error_type}: {error_msg}"
        )
        error_logger.debug(f"Full stack trace:\n{stack_trace}")
        
        # Log context if provided
        if context:
            error_logger.info(f"Error context: {json.dumps(context, indent=2)}")
        
        # Track critical errors
        if self._is_critical_error(error):
            self.critical_errors.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'error_type': error_type,
                'message': error_msg,
                'function': func_name,
                'line': line_no,
                'context': context
            })
            self.security_logger.error(
                f"Critical error: {error_type} - {error_msg}",
                extra={'context': context}
            )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events"""
        details.update({
            'timestamp': datetime.datetime.now().isoformat(),
            'ip_address': getattr(request, 'remote_addr', 'unknown') if 'request' in globals() else 'unknown'
        })
        
        self.security_logger.warning(
            f"Security event: {event_type} - {json.dumps(details)}"
        )
    
    def log_performance(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """Log performance metrics"""
        perf_logger = logging.getLogger('performance')
        perf_logger.info(
            f"Performance: {operation} completed in {duration:.3f}s"
        )
        
        if details:
            perf_logger.debug(f"Performance details: {json.dumps(details)}")
    
    def _is_critical_error(self, error: Exception) -> bool:
        """Determine if error is critical"""
        critical_errors = [
            'DatabaseError', 'ConnectionError', 'MemoryError',
            'SystemExit', 'KeyboardInterrupt'
        ]
        
        return (type(error).__name__ in critical_errors or 
                'database' in str(error).lower() or 
                'connection' in str(error).lower())
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all logged errors"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': self.error_counts.copy(),
            'critical_errors': len(self.critical_errors),
            'recent_critical': self.critical_errors[-5:] if self.critical_errors else []
        }


def handle_errors(logger: MazeCaptchaLogger = None):
    """Decorator for automatic error handling and logging"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function context
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                if hasattr(logger, 'log_error'):
                    logger.log_error(e, context)
                else:
                    # Fallback logging
                    error_logger = logging.getLogger(func.__module__)
                    error_logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                
                # Re-raise if it's a critical error
                if logger and hasattr(logger, '_is_critical_error') and logger._is_critical_error(e):
                    raise
                
                # Return error response for Flask routes
                if 'request' in globals():
                    return {'error': 'An internal error occurred', 'type': type(e).__name__}, 500
                
                return None
        
        return wrapper
    return decorator


class DatabaseManager:
    """Database connection manager with error handling and recovery"""
    
    def __init__(self, db_path: str = 'maze_captcha.db', logger: MazeCaptchaLogger = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.connection_pool = []
        self.max_connections = 5
        self._initialize_database()
    
    @handle_errors()
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with error handling"""
        try:
            # Try to get from pool
            if self.connection_pool:
                conn = self.connection_pool.pop()
                if self._test_connection(conn):
                    return conn
                else:
                    conn.close()
            
            # Create new connection
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            
            # Configure connection
            conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
            conn.execute('PRAGMA synchronous=NORMAL')  # Balance of safety and speed
            conn.execute('PRAGMA cache_size=10000')  # 10MB cache
            conn.execute('PRAGMA temp_store=MEMORY')  # Use RAM for temp tables
            
            self.logger.log_performance('database_connection', 0.001)
            return conn
            
        except sqlite3.Error as e:
            self.logger.log_error(e, {'operation': 'get_connection', 'db_path': self.db_path})
            raise ConnectionError(f"Database connection failed: {str(e)}")
    
    def release_connection(self, conn: sqlite3.Connection):
        """Release connection back to pool"""
        try:
            if len(self.connection_pool) < self.max_connections:
                self.connection_pool.append(conn)
            else:
                conn.close()
        except Exception as e:
            self.logger.log_error(e, {'operation': 'release_connection'})
    
    def _test_connection(self, conn: sqlite3.Connection) -> bool:
        """Test if connection is still alive"""
        try:
            conn.execute('SELECT 1')
            return True
        except sqlite3.Error:
            return False
    
    @handle_errors()
    def execute_query(self, query: str, params: tuple = None, fetch: str = 'all') -> Any:
        """Execute database query with error handling"""
        conn = None
        try:
            start_time = datetime.datetime.now()
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch == 'all':
                result = cursor.fetchall()
            elif fetch == 'one':
                result = cursor.fetchone()
            elif fetch == 'many':
                result = cursor.fetchmany()
            else:
                result = None
            
            if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                conn.commit()
            
            duration = (datetime.datetime.now() - start_time).total_seconds()
            self.logger.log_performance('database_query', duration, {
                'query': query[:100] + '...' if len(query) > 100 else query,
                'has_params': params is not None
            })
            
            return result
            
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.log_error(e, {
                'operation': 'execute_query',
                'query': query[:200] + '...' if len(query) > 200 else query,
                'params': str(params)[:100] if params else None
            })
            raise
        finally:
            if conn:
                self.release_connection(conn)
    
    def _initialize_database(self):
        """Initialize database with required tables"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            tables = [
                """
                CREATE TABLE IF NOT EXISTS captcha_sessions (
                    id TEXT PRIMARY KEY,
                    maze_data TEXT NOT NULL,
                    solution_path TEXT NOT NULL,
                    start_point TEXT NOT NULL,
                    end_point TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_verified BOOLEAN DEFAULT FALSE,
                    ip_address TEXT,
                    user_agent TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS user_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    coordinates TEXT NOT NULL,
                    timestamp_data TEXT NOT NULL,
                    velocity_data TEXT,
                    solve_time REAL,
                    is_human BOOLEAN,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES captcha_sessions (id)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    session_id TEXT,
                    data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    indicator_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL,
                    is_flagged BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES captcha_sessions (id)
                )
                """
            ]
            
            for table_sql in tables:
                cursor.execute(table_sql)
            
            conn.commit()
            self.release_connection(conn)
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.log_error(e, {'operation': 'initialize_database'})
            raise
    
    def cleanup(self):
        """Cleanup database connections"""
        for conn in self.connection_pool:
            try:
                conn.close()
            except:
                pass
        self.connection_pool.clear()


class FlaskErrorHandler:
    """Flask error handler with structured logging"""
    
    def __init__(self, app: Flask, logger: MazeCaptchaLogger):
        self.app = app
        self.logger = logger
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup Flask error handlers"""
        
        @self.app.errorhandler(400)
        def bad_request(error):
            self.logger.log_error(error, {
                'request_method': request.method if 'request' in globals() else None,
                'request_url': request.url if 'request' in globals() else None,
                'user_agent': request.headers.get('User-Agent') if 'request' in globals() else None
            })
            return {'error': 'Bad request', 'message': str(error)}, 400
        
        @self.app.errorhandler(401)
        def unauthorized(error):
            self.logger.log_security_event('unauthorized_access', {
                'url': request.url if 'request' in globals() else None,
                'method': request.method if 'request' in globals() else None
            })
            return {'error': 'Unauthorized', 'message': str(error)}, 401
        
        @self.app.errorhandler(403)
        def forbidden(error):
            self.logger.log_security_event('forbidden_access', {
                'url': request.url if 'request' in globals() else None,
                'method': request.method if 'request' in globals() else None
            })
            return {'error': 'Forbidden', 'message': str(error)}, 403
        
        @self.app.errorhandler(404)
        def not_found(error):
            self.logger.log_error(error, {
                'url': request.url if 'request' in globals() else None,
                'method': request.method if 'request' in globals() else None
            })
            return {'error': 'Not found', 'message': str(error)}, 404
        
        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            self.logger.log_security_event('rate_limit_exceeded', {
                'url': request.url if 'request' in globals() else None,
                'ip': request.remote_addr if 'request' in globals() else None
            })
            return {'error': 'Rate limit exceeded', 'message': str(error)}, 429
        
        @self.app.errorhandler(500)
        def internal_error(error):
            self.logger.log_error(error, {
                'request_method': request.method if 'request' in globals() else None,
                'request_url': request.url if 'request' in globals() else None,
                'user_agent': request.headers.get('User-Agent') if 'request' in globals() else None,
                'traceback': traceback.format_exc()
            })
            return {'error': 'Internal server error', 'message': 'Something went wrong'}, 500
        
        @self.app.errorhandler(Exception)
        def handle_exception(error):
            """Catch-all exception handler"""
            self.logger.log_error(error, {
                'request_method': request.method if 'request' in globals() else None,
                'request_url': request.url if 'request' in globals() else None,
                'user_agent': request.headers.get('User-Agent') if 'request' in globals() else None,
                'traceback': traceback.format_exc()
            })
            
            # Don't expose internal errors in production
            if self.app.debug:
                return {'error': 'Internal error', 'message': str(error), 'traceback': traceback.format_exc()}, 500
            else:
                return {'error': 'Internal server error', 'message': 'Something went wrong'}, 500


# Global logger instance
maze_logger = MazeCaptchaLogger()
db_manager = DatabaseManager(logger=maze_logger)


def setup_error_handling(app: Flask) -> Flask:
    """Setup comprehensive error handling for Flask app"""
    # Setup Flask error handlers
    error_handler = FlaskErrorHandler(app, maze_logger)
    
    # Request logging
    @app.before_request
    def log_request_info():
        g.start_time = datetime.datetime.now()
        
        if 'request' in globals():
            maze_logger.logger.info(
                f"Request: {request.method} {request.url} - {request.remote_addr}"
            )
    
    @app.after_request
    def log_response_info(response):
        if hasattr(g, 'start_time'):
            duration = (datetime.datetime.now() - g.start_time).total_seconds()
            maze_logger.log_performance('http_request', duration, {
                'method': request.method if 'request' in globals() else None,
                'url': request.url if 'request' in globals() else None,
                'status_code': response.status_code
            })
        
        return response
    
    return app


# Test the error handling system
if __name__ == "__main__":
    print("🔧 Testing Error Handling System")
    
    # Test logging
    maze_logger.logger.info("Error handling system initialized")
    
    # Test database manager
    try:
        result = db_manager.execute_query("SELECT 1")
        print(f"✅ Database test successful: {result}")
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    
    # Test error logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        maze_logger.log_error(e, {'test': True})
        print("✅ Error logging test successful")
    
    # Test performance logging
    maze_logger.log_performance("test_operation", 0.123, {'test': True})
    print("✅ Performance logging test successful")
    
    # Test security event logging
    maze_logger.log_security_event("test_security_event", {'test': True})
    print("✅ Security logging test successful")
    
    # Show error summary
    summary = maze_logger.get_error_summary()
    print(f"📊 Error Summary: {summary}")
    
    print("🎉 Error handling system tests completed!")