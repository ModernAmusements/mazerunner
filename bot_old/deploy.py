#!/usr/bin/env python3
"""
Production Deployment Script for Maze Captcha
"""

import os
import sys
import subprocess
import time

def create_logs_directory():
    """Create logs directory for production"""
    os.makedirs('logs', exist_ok=True)
    print("✓ Created logs directory")

def install_production_dependencies():
    """Install production dependencies"""
    print("Installing production dependencies...")
    
    production_deps = [
        'gunicorn',
        'flask',
        'opencv-python',
        'numpy'
    ]
    
    for dep in production_deps:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True)
            print(f"✓ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")
            return False
    
    return True

def generate_ssl_certificates():
    """Generate SSL certificates for HTTPS"""
    if not os.path.exists('cert.pem') or not os.path.exists('key.pem'):
        print("Generating SSL certificates...")
        try:
            subprocess.run([
                'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
                '-keyout', 'key.pem', '-out', 'cert.pem',
                '-days', '365', '-nodes',
                '-subj', '/CN=localhost'
            ], check=True, capture_output=True)
            print("✓ Generated SSL certificates")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"✗ Failed to generate SSL certificates: {e}")
            print("Please install OpenSSL or provide existing certificates")
            return False
    else:
        print("✓ SSL certificates already exist")
        return True

def test_production_build():
    """Test production build"""
    print("Testing production build...")
    
    try:
        # Test imports
        import production_maze_captcha_clean
        import rate_limiter
        print("✓ All imports successful")
        
        # Test basic functionality
        from production_maze_captcha_clean import app
        with app.test_client() as client:
            response = client.get('/')
            assert response.status_code == 200
            print("✓ Basic web interface working")
            
            response = client.get('/api/analytics')
            assert response.status_code == 200
            print("✓ API endpoints working")
        
        return True
        
    except Exception as e:
        print(f"✗ Production build test failed: {e}")
        return False

def start_production_server(mode='https'):
    """Start production server"""
    print(f"Starting production server in {mode} mode...")
    
    if mode == 'https':
        try:
            # Start HTTPS server
            subprocess.run([
                sys.executable, 'production_https_server.py'
            ])
        except KeyboardInterrupt:
            print("\n🛑 Production server stopped")
    
    elif mode == 'gunicorn':
        try:
            # Start with Gunicorn
            subprocess.run([
                'gunicorn', '-c', 'gunicorn_config.py', 
                'production_maze_captcha_clean:app'
            ])
        except KeyboardInterrupt:
            print("\n🛑 Production server stopped")
    
    elif mode == 'dev':
        try:
            # Development mode for testing
            subprocess.run([
                sys.executable, 'production_maze_captcha_clean.py'
            ])
        except KeyboardInterrupt:
            print("\n🛑 Development server stopped")

def main():
    """Main deployment function"""
    print("🚀 Maze Captcha Production Deployment")
    print("=" * 50)
    
    # Create necessary directories
    create_logs_directory()
    
    # Install dependencies
    if not install_production_dependencies():
        print("❌ Deployment failed: Dependencies installation")
        sys.exit(1)
    
    # Generate SSL certificates
    if not generate_ssl_certificates():
        print("⚠️  Warning: SSL certificates not available - HTTPS mode will fail")
    
    # Test production build
    if not test_production_build():
        print("❌ Deployment failed: Production build test")
        sys.exit(1)
    
    print("\n✅ Production deployment successful!")
    print("\n📋 Available startup options:")
    print("  1. HTTPS mode (recommended for production):")
    print("     python production_https_server.py")
    print("  2. Gunicorn mode (high performance):")
    print("     gunicorn -c gunicorn_config.py production_maze_captcha_clean:app")
    print("  3. Development mode (for testing):")
    print("     python production_maze_captcha_clean.py")
    print("\n🔗 Access URLs:")
    print("  HTTP:  http://localhost:8080")
    print("  HTTPS: https://localhost:443 (if certificates available)")
    print("\n🛡️  Security Features Enabled:")
    print("  ✓ Rate limiting")
    print("  ✓ IP banning for abuse")
    print("  ✓ HTTPS support")
    print("  ✓ Security headers")
    print("  ✓ Session management")

if __name__ == '__main__':
    main()