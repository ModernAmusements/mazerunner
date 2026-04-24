# Behavioral Maze CAPTCHA System - Architecture

## System Overview

The Behavioral Maze CAPTCHA is a Flask-based web application that uses procedural maze generation with behavioral analysis to distinguish between human users and bots.

## System Architecture

```mermaid
graph TD
    A[User Browser] --> B[Flask Web Server]
    B --> C[API: /api/captcha]
    B --> D[API: /api/bot-simulate]
    B --> E[API: /api/verify]
    B --> F[API: /api/analytics]
    B --> G[API: /api/health]
    
    C --> H[Maze Generation]
    C --> I[Session Storage]
    C --> J[Database Storage]
    
    D --> K[Bot Path Generation]
    D --> L[Path Validation]
    D --> M[Analysis Engine]
    
    H --> N[Procedural Maze Algorithm]
    H --> O[Pathfinding Algorithm]
    H --> P[Maze Renderer]
    
    I --> Q[Flask Session Management]
    J --> R[SQLite Database]
    
    K --> S[Direct Path Algorithm]
    K --> T[Path Randomization]
    
    L --> U[Start Point Validation]
    L --> V[End Point Validation]
    L --> W[Wall Collision Detection]
    
    M --> X[Solve Time Analysis]
    M --> Y[Path Pattern Analysis]
    M --> Z[Confidence Scoring]
    
    E --> U
    E --> V
    E --> W
    E --> M
    
    F --> AA[Analytics Engine]
    AA --> AB[Session Stats]
    AA --> AC[Detection Stats]
    AA --> AD[Performance Metrics]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#c8e6c9
    style D fill:#ffebee
    style E fill:#fff3e0
    style F fill:#f1f8e9
    style G fill:#e8f5e9
    style H fill:#bbdefb
    style N fill:#c8e6c9
    style O fill:#a5d6a7
    style P fill:#81c784
    style K fill:#ffcc80
    style S fill:#ff8a80
    style L fill:#b2ebf2
    style U fill:#e1bee7
    style M fill:#9575cd
```

## Request Flow

```mermaid
sequenceDiagram
    participant U as User
    participant B as Browser
    participant S as Flask Server
    participant DB as SQLite Database
    participant H as Human Verification
    participant Bot as Bot Simulation
    participant A as Analytics
    
    U->>B: Request Homepage
    B->>S: GET /
    S->>B: Serve production_index.html
    
    U->>B: Load Human Captcha
    B->>S: GET /api/captcha
    S->>DB: Generate session ID
    S->>H: Generate maze
    H->>S: Return maze data
    S->>DB: Store session data
    S->>B: Return captcha with image
    
    U->>B: Draw human path
    B->>S: POST /api/verify
    S->>S: Validate path
    S->>S: Analyze behavior
    S->>DB: Record attempt
    S->>B: Return result
    
    U->>B: Trigger Bot Simulation
    B->>S: POST /api/bot-simulate
    S->>Bot: Generate bot path
    Bot->>S: Return bot path
    S->>S: Analyze bot behavior
    S->>DB: Record bot attempt
    S->>B: Return analysis
    
    B->>S: GET /api/analytics
    S->>DB: Query stats
    S->>B: Return analytics data
    
    Note over S,B: Dual CAPTCHA System<br/>Human + Bot Challenge
```

## Component Details

```mermaid
graph TD
    subgraph Maze_Generation
        G1[Generate Maze Grid] --> G2[Apply DFS Pathfinding]
        G2 --> G3[Select Start Point]
        G3 --> G4[Select End Point]
        G4 --> G5[Find Solution Path]
        G5 --> G6[Render as PNG Image]
    end
    
    subgraph Bot_Simulation
        B1[Get Session Data] --> B2[Validate Start/End Points]
        B2 --> B3[Generate Direct Path]
        B3 --> B4[Add Randomness]
        B4 --> B5[Calculate Solve Time]
        B5 --> B6[Return Bot Result]
    end
    
    subgraph Verification
        V1[Get User Path] --> V2[Validate Start Point]
        V2 --> V3[Validate End Point]
        V3 --> V4[Check Path Validity]
        V4 --> V5[Analyze Behavior]
        V5 --> V6[Classify Human/Bot]
    end
    
    subgraph Behavior_Analysis
        A1[Solve Time Analysis] --> A2[Too Fast = Bot]
        A1 --> A3[Normal Range = Human]
        A2 --> A4[Confidence Scoring]
        A3 --> A4
        A4 --> A5[Return Analysis Result]
    end
    
    G6 --> V1
    B6 --> A1
    
    style G1 fill:#c8e6c9
    style G6 fill:#81c784
    style B1 fill:#ffebee
    style B6 fill:#ff8a80
    style V1 fill:#e1f5fe
    style V6 fill:#0066ff
    style A1 fill:#e1bee7
    style A5 fill:#9575cd
```

## Database Schema

```mermaid
erDiagram
    captcha_sessions {
        string id PK
        string maze_data
        string solution_path
        string start_point
        string end_point
        timestamp created_at
        boolean is_verified
    }
    
    user_paths {
        int id PK
        string session_id FK
        string coordinates
        float solve_time
        boolean is_human
        float confidence_score
        timestamp created_at
    }
    
    captcha_sessions ||--o{ user_paths : "has"
```

## API Endpoints

```mermaid
graph LR
    A[GET /] --> B[Homepage]
    C[GET /api/captcha] --> D[Generate new CAPTCHA]
    E[POST /api/verify] --> F[Verify user's solution]
    G[POST /api/bot-simulate] --> H[Simulate bot behavior]
    I[GET /api/analytics] --> J[Get analytics data]
    K[GET /api/health] --> L[Health check]
    
    style A fill:#e1f5fe
    style C fill:#c8e6c9
    style E fill:#fff3e0
    style G fill:#ffebee
    style I fill:#f1f8e9
    style K fill:#e8f5e9
```

## Key Features

1. **Procedural Maze Generation** - Uses recursive backtracking (DFS) to generate random solvable mazes
2. **Behavioral Analysis** - Detects bots by analyzing solve time and path patterns
3. **Dual Challenge System** - Side-by-side human and bot captchas
4. **Real-time Analytics** - Tracks detection stats and performance metrics
5. **SQLite Persistence** - Stores sessions and attempts for analysis

## Tech Stack

- **Backend**: Flask (Python web framework)
- **Database**: SQLite3
- **Image Processing**: OpenCV, NumPy
- **Frontend**: Vanilla JavaScript, Canvas API, Chart.js
- **Styling**: Custom CSS with Inter font