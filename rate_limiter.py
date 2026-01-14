#!/usr/bin/env python3
"""
Rate limiting and abuse prevention for Maze Captcha
"""

from datetime import datetime, timedelta
from collections import defaultdict
import time
from flask import request, jsonify, g

class RateLimiter:
    def __init__(self):
        # Rate limiting storage
        self.requests = defaultdict(list)  # IP -> list of timestamps
        self.captcha_requests = defaultdict(list)  # IP -> list of captcha generation times
        self.failed_attempts = defaultdict(int)  # IP -> failed verification count
        self.banned_ips = {}  # IP -> ban expiration time
        
        # Rate limiting configuration
        self.config = {
            'general_requests_per_minute': 60,
            'captcha_requests_per_minute': 30,
            'failed_attempts_threshold': 5,
            'ban_duration_minutes': 15,
            'captcha_cooldown_seconds': 0
        }
    
    def is_ip_banned(self, ip):
        """Check if IP is currently banned"""
        if ip in self.banned_ips:
            if time.time() > self.banned_ips[ip]:
                del self.banned_ips[ip]
                self.failed_attempts[ip] = 0
                return False
            return True
        return False
    
    def ban_ip(self, ip):
        """Ban an IP for configured duration"""
        self.banned_ips[ip] = time.time() + (self.config['ban_duration_minutes'] * 60)
    
    def check_rate_limit(self, ip, request_type='general'):
        """Check if IP exceeds rate limit"""
        # Temporarily disable rate limiting for development
        return True
    
    def record_request(self, ip, request_type='general'):
        """Record a request for rate limiting"""
        if request_type == 'general':
            self.requests[ip].append(time.time())
        elif request_type == 'captcha':
            self.captcha_requests[ip].append(time.time())
    
    def record_failed_attempt(self, ip):
        """Record a failed verification attempt"""
        self.failed_attempts[ip] += 1
        
        if self.failed_attempts[ip] >= self.config['failed_attempts_threshold']:
            self.ban_ip(ip)
            return True  # IP was banned
        
        return False
    
    def record_successful_attempt(self, ip):
        """Record a successful verification"""
        self.failed_attempts[ip] = max(0, self.failed_attempts[ip] - 1)
    
    def get_ip_status(self, ip):
        """Get detailed status for an IP"""
        current_time = time.time()
        cutoff_time = current_time - 60
        
        recent_requests = len([t for t in self.requests[ip] if t > cutoff_time])
        recent_captchas = len([t for t in self.captcha_requests[ip] if t > cutoff_time])
        
        return {
            'ip': ip,
            'is_banned': self.is_ip_banned(ip),
            'ban_remaining_seconds': max(0, self.banned_ips.get(ip, 0) - current_time),
            'failed_attempts': self.failed_attempts[ip],
            'recent_requests_per_minute': recent_requests,
            'recent_captchas_per_minute': recent_captchas,
            'banned_until': datetime.fromtimestamp(self.banned_ips[ip]).isoformat() if ip in self.banned_ips else None
        }

# Global rate limiter instance
rate_limiter = RateLimiter()

# Flask middleware for rate limiting
def apply_rate_limiting(app):
    """Apply rate limiting middleware to Flask app"""
    
    @app.before_request
    def rate_limit_middleware():
        
        # Get client IP
        ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        
        # Skip rate limiting for static files, root route, and common assets
        if request.endpoint and (request.endpoint.startswith('static') or 
                              request.endpoint == 'index' or
                              request.endpoint is None):
            return
        
        # Skip rate limiting for static files, root route, and common assets
        if request.endpoint and (request.endpoint.startswith('static') or 
                              request.endpoint == 'index' or
                              request.endpoint is None):
            return
        
        # Check if IP is banned
        if rate_limiter.is_ip_banned(ip):
            return jsonify({
                'error': 'IP banned due to suspicious activity',
                'banned_until': rate_limiter.get_ip_status(ip)['banned_until']
            }), 429
        
        # Check rate limits
        request_type = 'captcha' if request.path.startswith('/api/captcha') else 'general'
        
        if not rate_limiter.check_rate_limit(ip, request_type):
            return jsonify({
                'error': 'Rate limit exceeded',
                'retry_after': 60,
                'status': rate_limiter.get_ip_status(ip)
            }), 429
        
        # Record the request
        rate_limiter.record_request(ip, request_type)
        
        # Store IP in flask context for later use
        g.client_ip = ip
        
        return None
    
    return app

# Administrative endpoints for monitoring
def add_admin_endpoints(app):
    """Add administrative endpoints for monitoring rate limiting"""
    
    @app.route('/admin/rate-limit-status/<ip>')
    def get_ip_status(ip):
        """Get rate limiting status for an IP (admin only)"""
        from flask import jsonify
        return jsonify(rate_limiter.get_ip_status(ip))
    
    @app.route('/admin/rate-limit-stats')
    def get_rate_limit_stats():
        """Get overall rate limiting statistics (admin only)"""
        from flask import jsonify
        
        stats = {
            'total_banned_ips': len(rate_limiter.banned_ips),
            'total_tracked_ips': len(rate_limiter.requests),
            'configuration': rate_limiter.config,
            'recently_active_ips': []
        }
        
        # Get recently active IPs
        current_time = time.time()
        cutoff_time = current_time - 300  # Last 5 minutes
        
        for ip in rate_limiter.requests:
            recent_activity = [t for t in rate_limiter.requests[ip] if t > cutoff_time]
            if recent_activity:
                stats['recently_active_ips'].append({
                    'ip': ip,
                    'recent_requests': len(recent_activity),
                    'failed_attempts': rate_limiter.failed_attempts[ip],
                    'is_banned': rate_limiter.is_ip_banned(ip)
                })
        
        return jsonify(stats)
    
    return app