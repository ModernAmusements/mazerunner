-- Behavioral Maze CAPTCHA System - Complete Database Schema
-- Supports advanced analytics, path tracking, and anomaly detection

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Main captcha sessions table
CREATE TABLE IF NOT EXISTS captcha_sessions (
    id TEXT PRIMARY KEY,                    -- Unique session identifier
    maze_data TEXT NOT NULL,                 -- JSON encoded maze array
    solution_path TEXT NOT NULL,             -- JSON encoded A* solution
    start_point TEXT NOT NULL,               -- JSON coordinates [row, col]
    end_point TEXT NOT NULL,                 -- JSON coordinates [row, col]
    difficulty_level TEXT DEFAULT 'medium',   -- easy, medium, hard
    cell_size INTEGER DEFAULT 20,            -- Pixel size of maze cells
    maze_size INTEGER DEFAULT 21,            -- Odd number for maze generation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,                    -- Session expiration
    is_verified BOOLEAN DEFAULT FALSE,        -- If path was successfully verified
    verification_result TEXT,                 -- JSON with verification details
    ip_address TEXT,                         -- Client IP for security
    user_agent TEXT,                         -- Browser/user agent
    session_fingerprint TEXT,                 -- Browser fingerprinting
    metadata TEXT,                           -- Additional session metadata
);
CREATE INDEX IF NOT EXISTS idx_created_at ON captcha_sessions (created_at);
CREATE INDEX IF NOT EXISTS idx_expires_at ON captcha_sessions (expires_at);
CREATE INDEX IF NOT EXISTS idx_ip_address ON captcha_sessions (ip_address);

-- Detailed user path tracking with behavioral metrics
CREATE TABLE IF NOT EXISTS user_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    coordinates TEXT NOT NULL,               -- JSON array of [x, y, timestamp] tuples
    timestamp_data TEXT NOT NULL,            -- JSON array of timestamps
    velocity_data TEXT,                       -- JSON array of calculated velocities
    acceleration_data TEXT,                   -- JSON array of accelerations
    solve_time REAL,                          -- Total time from start to finish
    path_length INTEGER,                        -- Number of points in path
    path_efficiency REAL,                      -- Path vs optimal ratio
    is_human BOOLEAN,                          -- Human/bot classification
    confidence_score REAL,                     -- Confidence in classification
    -- Behavioral metrics for analysis
    velocity_variance REAL,                     -- Variance in movement speed
    acceleration_variance REAL,                  -- Variance in acceleration
    direction_changes INTEGER,                  -- Number of direction changes
    jitter_magnitude REAL,                     -- Natural hand tremor measurement
    instant_turn_count INTEGER,                 -- Bot-like instant turns
    pause_count INTEGER,                       -- Human-like pauses
    avg_velocity REAL,                         -- Average movement speed
    max_velocity REAL,                         -- Maximum movement speed
    straight_line_ratio REAL,                  -- How direct the path is
    wall_touches INTEGER DEFAULT 0,           -- Number of wall contacts
    out_of_bounds INTEGER DEFAULT 0,           -- Times path went out of maze
    suspicious_indicators TEXT,                  -- JSON of flagged indicators
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES captcha_sessions (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_session_id ON user_paths (session_id);
CREATE INDEX IF NOT EXISTS idx_is_human ON user_paths (is_human);
CREATE INDEX IF NOT EXISTS idx_created_at ON user_paths (created_at);
CREATE INDEX IF NOT EXISTS idx_confidence_score ON user_paths (confidence_score);

-- Individual bot detection indicators
CREATE TABLE IF NOT EXISTS bot_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    indicator_type TEXT NOT NULL,              -- Type of indicator (e.g., 'instant_turn')
    value REAL NOT NULL,                       -- Measured value
    threshold REAL,                            -- Human threshold for this indicator
    is_flagged BOOLEAN DEFAULT FALSE,           -- If value exceeded threshold
    severity TEXT DEFAULT 'medium',             -- low, medium, high, critical
    description TEXT,                           -- Human-readable description
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES captcha_sessions (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_session_indicator ON bot_indicators (session_id, indicator_type);
CREATE INDEX IF NOT EXISTS idx_is_flagged ON bot_indicators (is_flagged);
CREATE INDEX IF NOT EXISTS idx_created_at ON bot_indicators (created_at);

-- Comprehensive analytics and event tracking
CREATE TABLE IF NOT EXISTS analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,                   -- Type of event
    session_id TEXT,                            -- Related session (optional)
    event_data TEXT,                            -- JSON event details
    user_agent TEXT,                            -- Browser information
    ip_address TEXT,                            -- Client IP
    response_time_ms INTEGER,                    -- Response time in milliseconds
    memory_usage_mb REAL,                       -- Server memory usage
    cpu_usage_percent REAL,                     -- Server CPU usage
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES captcha_sessions (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_event_type ON analytics (event_type);
CREATE INDEX IF NOT EXISTS idx_session_id ON analytics (session_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON analytics (created_at);
CREATE INDEX IF NOT EXISTS idx_response_time ON analytics (response_time_ms);

-- Learned human patterns for anomaly detection
CREATE TABLE IF NOT EXISTS human_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_hash TEXT UNIQUE,                   -- Hash of pattern for deduplication
    avg_solve_time REAL,
    avg_velocity_variance REAL,
    avg_direction_changes REAL,
    avg_jitter_magnitude REAL,
    avg_pause_count REAL,
    sample_count INTEGER DEFAULT 1,            -- Number of samples in this pattern
    pattern_data TEXT,                           -- Full pattern JSON
    confidence_weight REAL DEFAULT 1.0,         -- Weight for this pattern
    is_active BOOLEAN DEFAULT TRUE,               -- If pattern is used in detection
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);
CREATE INDEX IF NOT EXISTS idx_pattern_hash ON human_patterns (pattern_hash);
CREATE INDEX IF NOT EXISTS idx_is_active ON human_patterns (is_active);
CREATE INDEX IF NOT EXISTS idx_confidence_weight ON human_patterns (confidence_weight);

-- Rate limiting and security tracking
CREATE TABLE IF NOT EXISTS rate_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL,
    request_type TEXT NOT NULL,                 -- captcha, verify, admin, etc.
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    window_duration_minutes INTEGER DEFAULT 60,   -- Time window
    is_blocked BOOLEAN DEFAULT FALSE,             -- If IP is currently blocked
    block_reason TEXT,                            -- Reason for block
    block_expires TIMESTAMP,                      -- When block expires
    total_requests_today INTEGER DEFAULT 0,       -- Daily counter
    failed_attempts_today INTEGER DEFAULT 0,       -- Failed attempts today
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ip_address, request_type, window_start)
);
CREATE INDEX IF NOT EXISTS idx_rate_limit_ip ON rate_limits (ip_address);
CREATE INDEX IF NOT EXISTS idx_rate_limit_blocked ON rate_limits (is_blocked);
CREATE INDEX IF NOT EXISTS idx_rate_limit_expires ON rate_limits (block_expires);

-- System performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL,                  -- cpu, memory, response_time, etc.
    metric_value REAL NOT NULL,
    metric_unit TEXT,                            -- ms, MB, %, etc.
    server_instance TEXT,                        -- For multi-instance deployments
    metadata TEXT,                              -- Additional context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);
CREATE INDEX IF NOT EXISTS idx_performance_type ON performance_metrics (metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_created ON performance_metrics (created_at);

-- Failed attempt analysis for security
CREATE TABLE IF NOT EXISTS security_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,                   -- failed_attempt, suspicious_pattern, etc.
    severity TEXT DEFAULT 'medium',             -- low, medium, high, critical
    ip_address TEXT,
    session_id TEXT,
    event_details TEXT,                         -- JSON with full details
    threat_score REAL DEFAULT 0.0,            -- Calculated threat level
    is_resolved BOOLEAN DEFAULT FALSE,
    resolution_action TEXT,                      -- Action taken (blocked, monitored, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES captcha_sessions (id) ON DELETE SET NULL,
);
CREATE INDEX IF NOT EXISTS idx_security_event_type ON security_events (event_type);
CREATE INDEX IF NOT EXISTS idx_security_severity ON security_events (severity);
CREATE INDEX IF NOT EXISTS idx_security_ip ON security_events (ip_address);
CREATE INDEX IF NOT EXISTS idx_security_threat ON security_events (threat_score);
CREATE INDEX IF NOT EXISTS idx_security_created ON security_events (created_at);

-- Create views for common analytics queries
CREATE VIEW IF NOT EXISTS session_summary AS
SELECT 
    cs.id,
    cs.created_at,
    cs.ip_address,
    cs.difficulty_level,
    up.is_human,
    up.confidence_score,
    up.solve_time,
    up.velocity_variance,
    up.direction_changes,
    up.jitter_magnitude,
    up.wall_touches,
    CASE 
        WHEN up.is_human = TRUE THEN 'Human'
        WHEN up.is_human = FALSE THEN 'Bot'
        ELSE 'Uncertain'
    END as classification
FROM captcha_sessions cs
LEFT JOIN user_paths up ON cs.id = up.session_id;

CREATE VIEW IF NOT EXISTS daily_stats AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_sessions,
    COUNT(CASE WHEN is_human = TRUE THEN 1 END) as human_count,
    COUNT(CASE WHEN is_human = FALSE THEN 1 END) as bot_count,
    AVG(solve_time) as avg_solve_time,
    AVG(confidence_score) as avg_confidence
FROM session_summary
GROUP BY DATE(created_at);

CREATE VIEW IF NOT EXISTS bot_indicator_summary AS
SELECT 
    DATE(bi.created_at) as date,
    bi.indicator_type,
    COUNT(*) as total_flags,
    AVG(bi.value) as avg_value,
    AVG(bi.threshold) as avg_threshold,
    COUNT(CASE WHEN bi.is_flagged = TRUE THEN 1 END) as flagged_count
FROM bot_indicators bi
GROUP BY DATE(bi.created_at), bi.indicator_type;

-- Triggers for automatic maintenance
CREATE TRIGGER IF NOT EXISTS cleanup_expired_sessions
AFTER INSERT ON captcha_sessions
BEGIN
    DELETE FROM captcha_sessions 
    WHERE expires_at < datetime('now') AND created_at < datetime('now', '-7 days');
END;

CREATE TRIGGER IF NOT EXISTS update_rate_limits
AFTER UPDATE ON rate_limits
BEGIN
    UPDATE rate_limits 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS log_analytics_on_verification
AFTER UPDATE ON captcha_sessions
WHEN NEW.is_verified = TRUE AND OLD.is_verified = FALSE
BEGIN
    INSERT INTO analytics (event_type, session_id, event_data)
    VALUES ('session_verified', NEW.id, 
            json_object('verification_result', NEW.verification_result));
END;