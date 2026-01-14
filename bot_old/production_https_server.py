#!/usr/bin/env python3
"""
Production-Ready Maze Captcha with HTTPS Support
"""

import os
import ssl
from datetime import timedelta
from flask import Flask
from production_maze_captcha_clean import app as maze_app

def create_production_app():
    """Create production-ready Flask app with HTTPS support"""
    app = maze_app
    
    # Production configuration
    app.config.update(
        ENV='production',
        DEBUG=False,
        SESSION_COOKIE_SECURE=True,  # HTTPS only
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Strict',
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=10)
    )
    
    # Security headers
    @app.after_request
    def set_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self';"
        return response
    
    return app

def run_https_server(host='0.0.0.0', port=443, certfile='cert.pem', keyfile='key.pem'):
    """Run the Flask app with HTTPS"""
    app = create_production_app()
    
    # Generate self-signed certificate if not exists
    if not os.path.exists(certfile) or not os.path.exists(keyfile):
        print("Generating self-signed SSL certificate...")
        os.system(f"openssl req -x509 -newkey rsa:4096 -keyout {keyfile} -out {certfile} -days 365 -nodes -subj '/CN=localhost'")
    
    # Create SSL context
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    
    print(f"Starting HTTPS server on https://{host}:{port}")
    app.run(host=host, port=port, ssl_context=context, threaded=True)

if __name__ == '__main__':
    run_https_server()