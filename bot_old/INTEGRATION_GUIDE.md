# Production Maze Captcha System - Integration Documentation

## Overview

The Production Maze Captcha System is an advanced, AI-powered captcha solution that uses maze-solving challenges with behavioral analysis to distinguish between humans and bots. The system learns from human behavior patterns and adapts over time to provide increasingly sophisticated bot detection.

## Key Features

- **AI-Powered Behavioral Analysis**: Analyzes mouse movements, solve time, and path patterns
- **Machine Learning**: Continuously learns from human interactions
- **Rate Limiting & Abuse Prevention**: Built-in protection against automated attacks
- **Mobile Responsive**: Full touch support for mobile devices
- **Database Persistence**: SQLite backend for session and analytics storage
- **HTTPS Support**: Production-ready security with SSL/TLS
- **Real-time Analytics**: Behavioral analytics dashboard
- **Bot Simulation**: Built-in bot testing capabilities

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository>
cd maze-captcha

# Install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Deploy production setup
python deploy.py
```

### Running the System

```bash
# Development mode
python production_maze_captcha_db.py

# HTTPS mode (recommended for production)
python production_https_server.py

# High-performance mode with Gunicorn
gunicorn -c gunicorn_config.py production_maze_captcha_db:app
```

## API Documentation

### Base URL
- Development: `http://localhost:8080`
- Production HTTPS: `https://your-domain.com`

### Authentication
The API uses session-based authentication. All requests automatically maintain session state.

### Endpoints

#### 1. Generate Captcha
**GET** `/api/captcha`

Generate a new maze captcha challenge.

**Query Parameters:**
- `difficulty` (optional): `easy`, `medium`, `hard`, `expert` (default: `medium`)

**Response:**
```json
{
  "captcha_id": "abc123def456",
  "difficulty": "medium",
  "start": [1, 1],
  "end": [13, 13],
  "maze_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "timestamp": 1640995200.123,
  "learned_patterns": 42
}
```

#### 2. Track Mouse Movement
**POST** `/api/track`

Track mouse movements during captcha solving.

**Request Body:**
```json
{
  "captcha_id": "abc123def456",
  "mouse_data": [
    {
      "x": 100,
      "y": 150,
      "timestamp": 1640995200.123
    }
  ]
}
```

**Response:**
```json
{
  "success": true
}
```

#### 3. Verify Solution
**POST** `/api/verify`

Verify a completed maze solution.

**Request Body:**
```json
{
  "captcha_id": "abc123def456",
  "path": [
    [20, 20],
    [40, 20],
    [60, 40]
  ]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Human verified - captcha solved successfully!",
  "analysis": {
    "is_human": true,
    "confidence": 0.85,
    "reasons": [],
    "velocity_variance": 12.3,
    "solve_time": 15.2,
    "direction_changes": 8,
    "path_length": 25,
    "learned_from_humans": 42
  }
}
```

#### 4. Bot Simulation
**POST** `/api/bot-simulate`

Simulate bot behavior for testing purposes.

**Request Body:**
```json
{
  "difficulty": "medium",
  "mode": "good"  // "good", "perfect", "suspicious"
}
```

**Response:**
```json
{
  "captcha_id": "def456ghi789",
  "difficulty": "medium",
  "start": [1, 1],
  "end": [13, 13],
  "maze_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "bot_path": [
    {"x": 20, "y": 20},
    {"x": 40, "y": 20}
  ],
  "analysis": {
    "is_human": false,
    "confidence": 0.3,
    "solve_time": 2.1
  },
  "mimicking_human": true
}
```

#### 5. Get Analytics
**GET** `/api/analytics`

Get system analytics and learning statistics.

**Response:**
```json
{
  "total_attempts": 1250,
  "successful_verifications": 980,
  "bot_detected": 270,
  "human_detected": 980,
  "database_stats": {
    "sessions": 45,
    "captcha_sessions": 1250,
    "analytics": 1520,
    "human_patterns": 42
  },
  "learning_status": {
    "human_patterns_stored": 42,
    "behaviors_learned": 42,
    "avg_human_solve_time": 12.5,
    "avg_human_velocity_variance": 15.3
  }
}
```

### Admin Endpoints

#### Rate Limiting Statistics
**GET** `/admin/rate-limit-stats`

Get rate limiting and abuse prevention statistics.

#### IP Status
**GET** `/admin/rate-limit-status/<ip>`

Get detailed rate limiting status for a specific IP.

#### Database Statistics
**GET** `/admin/database-stats`

Get database table statistics.

#### Cleanup Database
**POST** `/admin/cleanup`

Clean up expired sessions and data.

## Integration Examples

### JavaScript Frontend Integration

```html
<!DOCTYPE html>
<html>
<head>
    <title>Maze Captcha Integration</title>
    <script>
        let currentCaptcha = null;
        
        async function loadCaptcha() {
            const response = await fetch('/api/captcha?difficulty=medium');
            currentCaptcha = await response.json();
            
            // Display the maze
            document.getElementById('maze').src = currentCaptcha.maze_image;
        }
        
        async function verifyCaptcha() {
            if (!currentCaptcha) return;
            
            const response = await fetch('/api/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    captcha_id: currentCaptcha.captcha_id,
                    path: getUserPath()  // Implement path collection
                })
            });
            
            const result = await response.json();
            if (result.success) {
                alert('Captcha passed! User verified as human.');
            } else {
                alert('Bot detected: ' + result.message);
            }
        }
        
        // Initialize
        loadCaptcha();
    </script>
</head>
<body>
    <img id="maze" alt="Maze Captcha">
    <button onclick="verifyCaptcha()">Verify Solution</button>
    <button onclick="loadCaptcha()">New Captcha</button>
</body>
</html>
```

### Python Backend Integration

```python
import requests
import json

class MazeCaptchaClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_captcha(self, difficulty="medium"):
        """Generate a new captcha"""
        response = self.session.get(
            f"{self.base_url}/api/captcha",
            params={"difficulty": difficulty}
        )
        return response.json()
    
    def verify_solution(self, captcha_id, path):
        """Verify a captcha solution"""
        response = self.session.post(
            f"{self.base_url}/api/verify",
            json={
                "captcha_id": captcha_id,
                "path": path
            }
        )
        return response.json()
    
    def simulate_bot(self, difficulty="medium", mode="good"):
        """Simulate bot behavior for testing"""
        response = self.session.post(
            f"{self.base_url}/api/bot-simulate",
            json={
                "difficulty": difficulty,
                "mode": mode
            }
        )
        return response.json()

# Usage example
client = MazeCaptchaClient()

# Generate captcha
captcha = client.generate_captcha("easy")
print(f"Captcha ID: {captcha['captcha_id']}")

# Verify solution (example path)
result = client.verify_solution(captcha['captcha_id'], [[20, 20], [40, 20], [60, 40]])
print(f"Verification result: {result['success']}")
```

## Security Considerations

### Rate Limiting Configuration

Default rate limiting rules:
- General requests: 60 per minute per IP
- Captcha generation: 10 per minute per IP
- Failed attempts threshold: 5 attempts before temporary ban
- Ban duration: 15 minutes
- Captcha cooldown: 2 seconds between requests

### HTTPS Configuration

For production deployment, use the HTTPS server:

```bash
python production_https_server.py
```

This will generate self-signed SSL certificates or use existing ones at:
- `cert.pem` - SSL certificate
- `key.pem` - Private key

### Database Security

The system uses SQLite with the following security features:
- Session isolation
- Automatic cleanup of expired data
- Encrypted session storage (Flask sessions)

## Configuration

### Environment Variables

```bash
# Flask configuration
FLASK_ENV=production
FLASK_DEBUG=False

# Database
DATABASE_URL=sqlite:///maze_captcha.db

# Rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_CAPTCHA_PER_MINUTE=10
RATE_LIMIT_FAILED_THRESHOLD=5

# SSL (optional)
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### Gunicorn Configuration

See `gunicorn_config.py` for production server configuration.

## Monitoring and Maintenance

### Database Maintenance

```bash
# Clean up expired data
curl -X POST http://localhost:8080/admin/cleanup

# Get database statistics
curl http://localhost:8080/admin/database-stats

# Get rate limiting stats
curl http://localhost:8080/admin/rate-limit-stats
```

### Log Monitoring

The system logs to:
- `logs/access.log` - HTTP access logs
- `logs/error.log` - Error logs
- Database - Analytics and event logs

### Performance Monitoring

Monitor these key metrics:
- Captcha generation rate
- Human vs bot detection ratios
- Average solve times
- Rate limiting violations
- Database size and query performance

## Troubleshooting

### Common Issues

1. **Rate Limiting Errors**
   - Check `/admin/rate-limit-stats/<ip>` for IP status
   - Verify rate limiting configuration
   - Check for distributed attacks

2. **Database Issues**
   - Ensure write permissions for `.db` file
   - Run cleanup to remove expired data
   - Check disk space

3. **SSL Certificate Issues**
   - Verify certificate and key file paths
   - Check certificate expiration
   - Ensure proper file permissions

4. **Performance Issues**
   - Monitor database size
   - Check maze complexity (larger mazes = more processing)
   - Review rate limiting effectiveness

### Support

For issues and questions:
1. Check the admin endpoints for system status
2. Review the logs for error details
3. Verify configuration settings
4. Test with bot simulation endpoint

## License

This software is provided as-is for educational and production use. Please ensure compliance with applicable regulations when deploying in production environments.