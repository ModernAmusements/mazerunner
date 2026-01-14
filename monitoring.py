#!/usr/bin/env python3
"""
Production Monitoring and Logging for Maze Captcha
"""

import logging
import logging.handlers
import os
import json
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import request, g
import threading

class MazeCaptchaLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure main logger
        self.logger = logging.getLogger('maze_captcha')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler for access logs with rotation
        access_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, 'access.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        access_handler.setLevel(logging.INFO)
        
        # File handler for error logs with rotation
        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, 'error.log'),
            maxBytes=5*1024*1024,   # 5MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        
        # File handler for security events
        security_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, 'security.log'),
            maxBytes=5*1024*1024,   # 5MB
            backupCount=5
        )
        security_handler.setLevel(logging.WARNING)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        access_formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        
        # Set formatters
        access_handler.setFormatter(access_formatter)
        error_handler.setFormatter(detailed_formatter)
        security_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(detailed_formatter)
        
        # Add handlers
        self.logger.addHandler(access_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(security_handler)
        self.logger.addHandler(console_handler)
        
        # Security specific logger
        self.security_logger = logging.getLogger('maze_captcha.security')
        self.security_logger.addHandler(security_handler)
        self.security_logger.setLevel(logging.WARNING)
    
    def log_request(self, endpoint, method, status_code, response_time, ip=None, user_agent=None):
        """Log HTTP request"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'response_time_ms': round(response_time * 1000, 2),
            'ip': ip or getattr(g, 'client_ip', 'unknown'),
            'user_agent': user_agent or 'unknown'
        }
        
        self.logger.info(f"ACCESS: {json.dumps(log_data)}")
    
    def log_captcha_event(self, event_type, data, ip=None):
        """Log captcha-related events"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'ip': ip or getattr(g, 'client_ip', 'unknown'),
            'data': data
        }
        
        if event_type in ['bot_detected', 'rate_limit_exceeded', 'ip_banned']:
            self.security_logger.warning(f"SECURITY: {json.dumps(log_data)}")
        else:
            self.logger.info(f"CAPTCHA: {json.dumps(log_data)}")
    
    def log_error(self, error, context=None):
        """Log errors with context"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(error),
            'type': type(error).__name__,
            'context': context or {},
            'ip': getattr(g, 'client_ip', 'unknown')
        }
        
        self.logger.error(f"ERROR: {json.dumps(log_data)}")
    
    def log_performance(self, operation, duration, details=None):
        """Log performance metrics"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'details': details or {}
        }
        
        self.logger.info(f"PERFORMANCE: {json.dumps(log_data)}")

class PerformanceMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.metrics = {}
        self.lock = threading.Lock()
    
    def time_operation(self, operation_name):
        """Decorator to time operations"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    self.record_metric(operation_name, duration, success, error)
                return result
            return wrapper
        return decorator
    
    def record_metric(self, operation, duration, success=True, error=None):
        """Record performance metric"""
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = {
                    'count': 0,
                    'total_duration': 0,
                    'success_count': 0,
                    'error_count': 0,
                    'min_duration': float('inf'),
                    'max_duration': 0,
                    'errors': []
                }
            
            metric = self.metrics[operation]
            metric['count'] += 1
            metric['total_duration'] += duration
            metric['min_duration'] = min(metric['min_duration'], duration)
            metric['max_duration'] = max(metric['max_duration'], duration)
            
            if success:
                metric['success_count'] += 1
            else:
                metric['error_count'] += 1
                if error and len(metric['errors']) < 10:
                    metric['errors'].append(error)
            
            # Log performance if significant
            if duration > 1.0:  # Operations taking more than 1 second
                self.logger.log_performance(operation, duration, {
                    'success': success,
                    'avg_duration': metric['total_duration'] / metric['count']
                })
    
    def get_metrics(self):
        """Get all performance metrics"""
        with self.lock:
            result = {}
            for operation, metric in self.metrics.items():
                if metric['count'] > 0:
                    result[operation] = {
                        'count': metric['count'],
                        'avg_duration_ms': round(metric['total_duration'] / metric['count'] * 1000, 2),
                        'min_duration_ms': round(metric['min_duration'] * 1000, 2),
                        'max_duration_ms': round(metric['max_duration'] * 1000, 2),
                        'success_rate': round(metric['success_count'] / metric['count'] * 100, 2),
                        'error_count': metric['error_count'],
                        'recent_errors': metric['errors'][-5:] if metric['errors'] else []
                    }
            return result

class HealthMonitor:
    def __init__(self, logger, db_session_manager):
        self.logger = logger
        self.db_manager = db_session_manager
        self.start_time = time.time()
        self.health_checks = {}
    
    def check_database_health(self):
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            stats = self.db_manager.get_database_stats()
            duration = time.time() - start_time
            
            health = {
                'status': 'healthy' if duration < 0.1 else 'degraded',
                'response_time_ms': round(duration * 1000, 2),
                'database_stats': stats
            }
            
            if duration > 0.5:
                health['status'] = 'unhealthy'
            
            return health
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_rate_limiter_health(self):
        """Check rate limiting system"""
        try:
            # Import rate limiter here to avoid circular imports
            from rate_limiter import rate_limiter
            stats = self.db_manager.get_database_stats()
            return {
                'status': 'healthy',
                'tracked_ips': stats.get('rate_limits', 0),
                'banned_ips': len(rate_limiter.banned_ips)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_memory_usage(self):
        """Check memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'status': 'healthy',
                'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'cpu_percent': process.cpu_percent()
            }
        except ImportError:
            return {'status': 'unknown', 'error': 'psutil not available'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def get_system_health(self):
        """Get overall system health"""
        uptime = time.time() - self.start_time
        
        health = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': round(uptime, 2),
            'uptime_formatted': str(timedelta(seconds=int(uptime))),
            'database': self.check_database_health(),
            'rate_limiter': self.check_rate_limiter_health(),
            'memory': self.check_memory_usage(),
            'overall_status': 'healthy'
        }
        
        # Determine overall status
        for component in ['database', 'rate_limiter', 'memory']:
            if health[component].get('status') == 'unhealthy':
                health['overall_status'] = 'unhealthy'
                break
            elif health[component].get('status') == 'degraded':
                health['overall_status'] = 'degraded'
        
        return health

# Global instances
maze_logger = MazeCaptchaLogger()
performance_monitor = PerformanceMonitor(maze_logger)

def setup_monitoring(app, db_session_manager):
    """Setup monitoring for Flask app"""
    from flask import jsonify
    health_monitor = HealthMonitor(maze_logger, db_session_manager)
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            maze_logger.log_request(
                request.endpoint,
                request.method,
                response.status_code,
                response_time,
                request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
                request.headers.get('User-Agent')
            )
        return response
    
    @app.route('/admin/health')
    def health_check():
        """System health check endpoint"""
        return jsonify(health_monitor.get_system_health())
    
    @app.route('/admin/metrics')
    def get_metrics():
        """Performance metrics endpoint"""
        return jsonify(performance_monitor.get_metrics())
    
    @app.route('/admin/logs', methods=['GET'])
    def get_logs():
        """Get recent logs"""
        from flask import jsonify
        log_type = request.args.get('type', 'access')
        lines = int(request.args.get('lines', 50))
        
        try:
            log_file = os.path.join(maze_logger.log_dir, f'{log_type}.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()
                    recent_lines = all_lines[-lines:] if lines > 0 else all_lines
                    return jsonify({
                        'logs': [line.strip() for line in recent_lines],
                        'total_lines': len(all_lines)
                    })
            else:
                return jsonify({'logs': [], 'message': 'Log file not found'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return health_monitor

# Monitoring decorators
def monitor_captcha_generation(func):
    """Monitor captcha generation performance"""
    return performance_monitor.time_operation('captcha_generation')(func)

def monitor_verification(func):
    """Monitor verification performance"""
    return performance_monitor.time_operation('verification')(func)

def monitor_database_operation(func):
    """Monitor database operations"""
    return performance_monitor.time_operation('database_operation')(func)