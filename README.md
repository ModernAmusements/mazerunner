# Production-Ready Maze Captcha System

A sophisticated CAPTCHA system that uses maze-solving challenges to distinguish humans from bots with advanced behavioral analysis and learning capabilities.

## Features

### 🧩 Core Functionality
- **Dynamic Maze Generation**: Procedurally generated mazes using recursive backtracking
- **Path Validation**: Robust verification with wall-touch tolerance for human-like behavior
- **Real-time Mouse Tracking**: Comprehensive movement analysis for bot detection

### 🤖 Advanced Bot Detection
- **Behavioral Analysis**: Velocity variance, direction changes, solve time patterns
- **Learning System**: Builds database of human behavior patterns to improve detection
- **Multiple Bot Strategies**: Perfect, human-like, and hybrid bot simulation

### 🛡️ Production Features
- **Rate Limiting**: IP-based throttling and banning for abuse prevention
- **Secure Session Management**: Production-hardened Flask sessions with configurable timeouts
- **Comprehensive Analytics**: Real-time statistics and performance monitoring
- **Memory-Safe Operations**: Prevents memory leaks through data limits and cleanup

### 📊 Monitoring & Analytics
- **Real-time Statistics**: Success rates, bot detection accuracy, performance metrics
- **Event Logging**: Detailed tracking of all captcha interactions
- **Learning Progress**: Visualization of human behavior pattern accumulation

## Project Structure

```
BOT/
├── production_maze_captcha.py      # Main Flask application
├── database_sessions.py            # Database session management
├── rate_limiter.py                 # Rate limiting and security
├── monitoring.py                   # Analytics and logging
├── requirements.txt                # Python dependencies
├── templates/
│   └── production_index.html       # Frontend interface
├── logs/                           # Application logs
├── maze_env/                       # Python virtual environment
└── maze_captcha.db                 # SQLite database
```

## Installation

1. **Set up virtual environment**:
   ```bash
   source maze_env/bin/activate  # On Windows: maze_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python production_maze_captcha.py
   ```

## API Endpoints

- `GET /` - Main captcha interface
- `GET /api/captcha` - Generate new maze captcha
- `POST /api/track` - Track mouse movements
- `POST /api/bot-simulate` - Test bot detection
- `GET /api/analytics` - View system statistics

## Configuration

Default settings:
- **Host**: 127.0.0.1
- **Port**: 8080
- **Session Timeout**: 30 minutes
- **Rate Limits**: Configurable thresholds for abuse prevention

## Security Features

- Wall-touch tolerance (allows minor human errors)
- Path validation (start/end point verification)
- Failed attempt tracking with IP banning
- Secure session management
- Comprehensive audit logging
- Memory-safe analytics to prevent resource exhaustion

## Technologies Used

- **Backend**: Flask (Python)
- **Image Processing**: OpenCV, NumPy
- **Frontend**: HTML5, JavaScript, Chart.js
- **Database**: SQLite
- **Security**: Rate limiting, session management

## Development

The system uses a modular architecture with separate modules for:
- Rate limiting and security (`rate_limiter.py`)
- Database operations (`database_sessions.py`)
- Monitoring and analytics (`monitoring.py`)

Each module can be developed and tested independently.

## Performance

- Optimized maze generation algorithms
- Efficient path validation with tolerance
- Minimal database queries through session management
- Real-time analytics without performance impact
- Memory-safe operations to prevent leaks

## Production Hardening

The system has been hardened for production use with:
- Secure session configuration (SESSION_COOKIE_SECURE configurable)
- Memory-safe analytics storage with data limits
- Comprehensive input validation and error handling
- No debug output in production code
- Rate limiting and abuse prevention

## Decoupled Bot Simulation Feature

The system now supports decoupled bot simulation with side-by-side captchas:
- One captcha for human verification
- One captcha for bot simulation
- Both captchas use the same maze structure but with different path validation

## License

This project is developed for educational and research purposes in CAPTCHA technology and bot detection systems.