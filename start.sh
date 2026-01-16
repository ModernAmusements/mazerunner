#!/bin/bash
"""
Behavioral Maze CAPTCHA System - Startup Script
Ensures virtual environment is activated and dependencies are available
"""

echo "🚀 Starting Behavioral Maze CAPTCHA System"
echo "========================================"

# Function to check if port 8080 is in use
check_port() {
    if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null ; then
        echo "🔧 Port 8080 is in use. Stopping existing processes..."
        lsof -ti:8080 | xargs kill -9
        sleep 2
        echo "✅ Port 8080 freed"
    else
        echo "✅ Port 8080 is available"
    fi
}

# Check and free port 8080
check_port

# Check if virtual environment exists
if [ ! -d "maze_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv maze_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source maze_env/bin/activate

# Check dependencies
echo "📦 Checking dependencies..."
python -c "
try:
    import numpy, cv2, flask, sqlite3, json, time, hashlib
    print('✅ All dependencies available!')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed!"
    exit 1
fi

# Initialize database if needed
echo "💾 Initializing database..."
python -c "
import sqlite3
import os

if not os.path.exists('maze_captcha.db'):
    conn = sqlite3.connect('maze_captcha.db')
    cursor = conn.cursor()
    
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
    print('✅ Database initialized!')
else:
    print('✅ Database already exists!')
"

# Start the application
echo "🌐 Starting server..."
echo "Access the CAPTCHA system at: http://127.0.0.1:8080"
echo "Health check at: http://127.0.0.1:8080/api/health"
echo "Press Ctrl+C to stop the server"
echo "========================================"

# Run with the activated virtual environment
exec python behavioral_maze_captcha.py