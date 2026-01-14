# Gunicorn production configuration for Maze Captcha

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 30
keepalive = 2

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "maze_captcha"

# Server mechanics
daemon = False
pidfile = "logs/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if using HTTPS with Gunicorn)
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Graceful shutdown
graceful_timeout = 30

# Monitoring
statsd_host = None

# Environment
raw_env = [
    'FLASK_ENV=production',
    'FLASK_DEBUG=False'
]