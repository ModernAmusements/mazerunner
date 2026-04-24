# Behavioral Maze CAPTCHA System - Development Context

## Important Notes

### Primary Application File
**The main and only application file is `behavioral_maze_captcha.py`**

The startup script (`start.sh`) runs:
```bash
exec python behavioral_maze_captcha.py
```

### Session Data Key Naming Conventions
There's an important naming convention to be aware of:

**Session Storage** (in `get_captcha` function):
```python
session_data = {
    'maze': maze.tolist(),
    'solution_path': solution,       # Note: solution_path
    'start_point': start,              # Note: start_point
    'end_point': end,                  # Note: end_point
    'created_at': time.time(),
    'mouse_data': [],
    'start_time': None
}
```

**Bot Simulation** (in `bot_simulate` function):
The code handles BOTH naming conventions:
```python
start = captcha_data.get('start') or captcha_data.get('start_point')
end = captcha_data.get('end') or captcha_data.get('end_point')
```

### UI Architecture
The frontend uses a scientific minimal black & white design:
- IBM Plex Sans/Mono typography
- Navigation bar with section anchors (#captcha, #analytics, #about)
- Grid-based layout with side-by-side captcha panels
- Bot path visualization (red dashed line after simulation)
- Chart.js for analytics visualization

### Status Message Handling
The `showStatus(message, type, targetId)` function accepts a third parameter for targeting specific status divs:
- 'humanStatus' - Human captcha section
- 'botStatus' - Bot captcha section (default)

### Common Issues and Solutions

1. **Key Name Mismatch**: If you get errors like "Bot simulation error: 'start'", check:
   - Are session keys being stored with one name but accessed with another?
   - Always use `.get()` with fallback to handle both conventions

2. **Status Messages Mixing**: Always pass the targetId parameter to showStatus():
   - Human related: showStatus('msg', 'type', 'humanStatus')
   - Bot related: showStatus('msg', 'type', 'botStatus')

3. **Chart Rendering**: Charts require:
   - Canvas elements with proper wrapper divs (.chart-wrapper)
   - initCharts() called on page load
   - updateAnalytics() called after any captcha action

## System Architecture

### Routes
- `/` - Main CAPTCHA interface (renders production_index.html)
- `/api/captcha` - Generate new maze captcha
- `/api/verify` - Verify user's solution
- `/api/bot-simulate` - Simulate bot behavior for testing
- `/api/analytics` - Get analytics data
- `/api/health` - Health check endpoint

### Database Tables
- `captcha_sessions` - Stores captcha sessions
- `user_paths` - Stores user/bot path attempts

### Dependencies
- Flask (web framework)
- OpenCV (maze rendering)
- NumPy (array operations)
- SQLite3 (database)
- Chart.js (analytics visualization)

## Startup
```bash
./start.sh
# Or manually:
python behavioral_maze_captcha.py
```

Server runs on: http://127.0.0.1:8080

## Project Structure (MVP)

```
BOT/
├── behavioral_maze_captcha.py    # Main application (ONLY)
├── static/
│   ├── script.js                  # Frontend JavaScript
│   └── styles.css                 # Styling
├── templates/
│   └── production_index.html     # Main HTML template
├── requirements.txt               # Python dependencies
├── start.sh                       # Startup script
├── quick_start.sh                 # Quick startup
├── README.md                      # Project documentation
├── ARCHITECTURE.md                # System architecture diagrams
├── DEVELOPMENT_CONTEXT.md         # Development notes (this file)
├── .gitignore                     # Git ignore patterns
└── maze_captcha.db               # SQLite database
```