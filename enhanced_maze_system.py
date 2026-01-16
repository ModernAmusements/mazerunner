#!/usr/bin/env python3
"""
Enhanced Behavioral Maze CAPTCHA System with A* Pathfinding and Sophisticated Anti-Bot Analytics
Implements advanced behavioral biometrics for human vs bot detection
"""

import numpy as np
import cv2
import hashlib
import time
import math
import json
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
from dataclasses import dataclass
from enum import Enum
import sqlite3

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/maze_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DetectionResult(Enum):
    """Enum for detection results"""
    HUMAN = "human"
    BOT = "bot"
    UNCERTAIN = "uncertain"

@dataclass
class PathMetrics:
    """Data class for path analysis metrics"""
    solve_time: float
    velocity_variance: float
    acceleration_variance: float
    direction_changes: int
    jitter_magnitude: float
    path_deviation: float
    straight_line_ratio: float
    pause_count: int
    avg_velocity: float
    max_velocity: float
    instant_turn_count: int

class MazeGenerator:
    """Enhanced maze generator with A* pathfinding"""
    
    def __init__(self, size: int = 21, cell_size: int = 20):
        self.size = size if size % 2 == 1 else size + 1  # Ensure odd size
        self.cell_size = cell_size
        self.maze = None
        self.start = None
        self.end = None
        self.solution_path = None
        
    def generate_maze(self) -> np.ndarray:
        """Generate maze using recursive backtracking algorithm"""
        try:
            # Initialize maze with walls
            self.maze = np.zeros((self.size, self.size), dtype=np.uint8)
            
            # Start position (green)
            self.start = (1, 1)
            # End position (red) - ensure it's on opposite side
            self.end = (self.size - 2, self.size - 2)
            
            # Generate paths using recursive backtracking
            self._carve_paths(self.start[0], self.start[1])
            
            # Ensure start and end are accessible
            self.maze[self.start] = 1
            self.maze[self.end] = 1
            
            # Calculate solution using A*
            self.solution_path = self.astar_pathfinding()
            
            if not self.solution_path:
                logger.warning("No solution found, regenerating maze")
                return self.generate_maze()  # Recursive regeneration
                
            logger.info(f"Maze generated successfully: {self.size}x{self.size}, path length: {len(self.solution_path)}")
            return self.maze.copy()
            
        except Exception as e:
            logger.error(f"Error generating maze: {str(e)}")
            raise
    
    def _carve_paths(self, r: int, c: int) -> None:
        """Recursive backtracking maze generation"""
        self.maze[r, c] = 1  # Mark as path
        
        # Randomize directions
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        np.random.shuffle(directions)
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 < nr < self.size - 1 and 0 < nc < self.size - 1 and 
                self.maze[nr, nc] == 0):
                # Carve path between current and next cell
                self.maze[r + dr // 2, c + dc // 2] = 1
                self._carve_paths(nr, nc)
    
    def astar_pathfinding(self) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm for optimal solution"""
        try:
            if self.start is None or self.end is None:
                return None
                
            # Priority queue: (f_score, counter, position)
            counter = 0
            open_set = [(0, counter, self.start)]
            came_from = {}
            g_score = {self.start: 0}
            f_score = {self.start: self._heuristic(self.start, self.end)}
            closed_set = set()
            
            while open_set:
                current = open_set[0][2]  # Get position with lowest f_score
                
                if current == self.end:
                    # Reconstruct path
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    path.append(self.start)
                    return path[::-1]
                
                open_set.pop(0)
                closed_set.add(current)
                
                # Check neighbors
                for neighbor in self._get_neighbors(current):
                    if neighbor in closed_set:
                        continue
                        
                    tentative_g = g_score[current] + 1  # Cost is always 1 for maze
                    
                    if neighbor not in [item[2] for item in open_set]:
                        counter += 1
                        open_set.append((tentative_g + self._heuristic(neighbor, self.end), 
                                      counter, neighbor))
                    elif tentative_g >= g_score.get(neighbor, float('inf')):
                        continue
                    
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, self.end)
                
                # Sort open set by f_score
                open_set.sort(key=lambda x: x[0])
            
            return None
            
        except Exception as e:
            logger.error(f"A* pathfinding error: {str(e)}")
            return None
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells"""
        r, c = pos
        neighbors = []
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.size and 0 <= nc < self.size and 
                self.maze[nr, nc] == 1):
                neighbors.append((nr, nc))
        
        return neighbors
    
    def render_maze(self) -> np.ndarray:
        """Render maze as image with start (green) and end (red) markers"""
        if self.maze is None:
            raise ValueError("Maze not generated")
            
        rows, cols = self.maze.shape
        img = np.zeros((rows * self.cell_size, cols * self.cell_size, 3), dtype=np.uint8)
        
        # Draw maze
        for r in range(rows):
            for c in range(cols):
                if self.maze[r, c] == 1:
                    color = (255, 255, 255)  # White path
                else:
                    color = (0, 0, 0)  # Black wall
                    
                cv2.rectangle(img, 
                           (c * self.cell_size, r * self.cell_size),
                           ((c + 1) * self.cell_size, (r + 1) * self.cell_size),
                           color, -1)
        
        # Draw start point (green)
        if self.start:
            cv2.rectangle(img,
                        (self.start[1] * self.cell_size, self.start[0] * self.cell_size),
                        ((self.start[1] + 1) * self.cell_size, (self.start[0] + 1) * self.cell_size),
                        (0, 255, 0), -1)
        
        # Draw end point (red)
        if self.end:
            cv2.rectangle(img,
                        (self.end[1] * self.cell_size, self.end[0] * self.cell_size),
                        ((self.end[1] + 1) * self.cell_size, (self.end[0] + 1) * self.cell_size),
                        (0, 0, 255), -1)
        
        return img

class BehavioralAnalyzer:
    """Sophisticated anti-bot analytics with behavioral biometrics"""
    
    def __init__(self):
        self.human_patterns = []
        self.thresholds = {
            'solve_time_min': 2.0,      # seconds
            'solve_time_max': 60.0,     # seconds
            'velocity_variance_min': 50.0, # pixels²/s²
            'direction_changes_min': 3,     # minimum for human
            'jitter_threshold': 2.0,     # pixels
            'instant_turn_threshold': 0.1, # seconds for near-instant turns
            'straight_line_threshold': 0.95, # ratio for too-perfect paths
            'pause_velocity_threshold': 5.0  # pixels/s
        }
        
        # Load learned patterns from database
        self._load_learned_patterns()
    
    def analyze_behavior(self, mouse_data: List[Dict], path_data: List[Tuple[int, int]], 
                      solve_time: float) -> Tuple[DetectionResult, float, Dict[str, Any]]:
        """Analyze user behavior and determine if human or bot"""
        try:
            if len(mouse_data) < 10:
                return DetectionResult.BOT, 0.0, {"reason": "Insufficient data points"}
            
            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(mouse_data, path_data, solve_time)
            
            # Analyze each behavioral indicator
            indicators = self._analyze_indicators(metrics)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(indicators, metrics)
            
            # Make determination
            if confidence >= 0.8:
                result = DetectionResult.HUMAN
            elif confidence <= 0.3:
                result = DetectionResult.BOT
            else:
                result = DetectionResult.UNCERTAIN
            
            # Store successful human patterns for learning
            if result == DetectionResult.HUMAN:
                self._store_human_pattern(metrics)
            
            analysis_details = {
                "metrics": metrics.__dict__,
                "indicators": indicators,
                "confidence": confidence,
                "determination": result.value
            }
            
            logger.info(f"Behavior analysis: {result.value} (confidence: {confidence:.2f})")
            return result, confidence, analysis_details
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {str(e)}")
            return DetectionResult.BOT, 0.0, {"reason": "Analysis error"}
    
    def _calculate_metrics(self, mouse_data: List[Dict], path_data: List[Tuple[int, int]], 
                         solve_time: float) -> PathMetrics:
        """Calculate comprehensive behavioral metrics"""
        # Calculate velocities and accelerations
        velocities = []
        accelerations = []
        
        for i in range(1, len(mouse_data)):
            prev = mouse_data[i-1]
            curr = mouse_data[i]
            dt = (curr['timestamp'] - prev['timestamp']) / 1000.0  # Convert to seconds
            
            if dt > 0:
                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                distance = math.sqrt(dx**2 + dy**2)
                velocity = distance / dt
                velocities.append(velocity)
                
                # Calculate acceleration if we have previous velocity
                if i > 1 and len(velocities) > 1:
                    acceleration = abs(velocities[-1] - velocities[-2]) / dt
                    accelerations.append(acceleration)
        
        # Calculate direction changes
        direction_changes = 0
        instant_turns = 0
        if len(path_data) > 2:
            for i in range(2, len(path_data)):
                dir1 = (path_data[i-1][0] - path_data[i-2][0], path_data[i-1][1] - path_data[i-2][1])
                dir2 = (path_data[i][0] - path_data[i-1][0], path_data[i][1] - path_data[i-1][1])
                
                if dir1 != dir2:
                    direction_changes += 1
                    
                    # Check for instant turns (bot-like behavior)
                    if i < len(mouse_data):
                        time_diff = mouse_data[i]['timestamp'] - mouse_data[i-1]['timestamp']
                        if time_diff < self.thresholds['instant_turn_threshold'] * 1000:
                            instant_turns += 1
        
        # Calculate jitter (natural hand tremors)
        jitter_magnitude = self._calculate_jitter(mouse_data)
        
        # Calculate path deviation from optimal
        path_deviation = self._calculate_path_deviation(path_data)
        
        # Calculate straight-line ratio (how direct the path is)
        straight_line_ratio = self._calculate_straight_line_ratio(path_data)
        
        # Count pauses (human-like behavior)
        pause_count = sum(1 for v in velocities if v < self.thresholds['pause_velocity_threshold'])
        
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = np.max(velocities) if velocities else 0
        velocity_variance = np.var(velocities) if velocities else 0
        acceleration_variance = np.var(accelerations) if accelerations else 0
        
        return PathMetrics(
            solve_time=solve_time,
            velocity_variance=velocity_variance,
            acceleration_variance=acceleration_variance,
            direction_changes=direction_changes,
            jitter_magnitude=jitter_magnitude,
            path_deviation=path_deviation,
            straight_line_ratio=straight_line_ratio,
            pause_count=pause_count,
            avg_velocity=avg_velocity,
            max_velocity=max_velocity,
            instant_turn_count=instant_turns
        )
    
    def _calculate_jitter(self, mouse_data: List[Dict]) -> float:
        """Calculate natural hand jitter magnitude"""
        if len(mouse_data) < 3:
            return 0.0
        
        # Calculate deviations from smooth path
        total_jitter = 0
        count = 0
        
        for i in range(1, len(mouse_data) - 1):
            prev = mouse_data[i-1]
            curr = mouse_data[i]
            next = mouse_data[i+1]
            
            # Expected position on straight line
            if curr['timestamp'] > prev['timestamp']:
                ratio = (curr['timestamp'] - prev['timestamp']) / (next['timestamp'] - prev['timestamp'])
                expected_x = prev['x'] + ratio * (next['x'] - prev['x'])
                expected_y = prev['y'] + ratio * (next['y'] - prev['y'])
                
                # Actual deviation
                deviation = math.sqrt((curr['x'] - expected_x)**2 + (curr['y'] - expected_y)**2)
                total_jitter += deviation
                count += 1
        
        return total_jitter / count if count > 0 else 0.0
    
    def _calculate_path_deviation(self, path_data: List[Tuple[int, int]]) -> float:
        """Calculate deviation from optimal solution"""
        if not path_data:
            return float('inf')
        
        # This would compare with the A* solution if available
        # For now, calculate path efficiency
        total_distance = 0
        for i in range(1, len(path_data)):
            dx = path_data[i][0] - path_data[i-1][0]
            dy = path_data[i][1] - path_data[i-1][1]
            total_distance += math.sqrt(dx**2 + dy**2)
        
        # Optimal distance (straight line from start to end)
        if len(path_data) >= 2:
            optimal = math.sqrt((path_data[-1][0] - path_data[0][0])**2 + 
                             (path_data[-1][1] - path_data[0][1])**2)
            return total_distance / optimal if optimal > 0 else float('inf')
        
        return 0.0
    
    def _calculate_straight_line_ratio(self, path_data: List[Tuple[int, int]]) -> float:
        """Calculate how straight the path is (bot-like = very straight)"""
        if len(path_data) < 2:
            return 0.0
        
        # Calculate total path length
        path_length = 0
        for i in range(1, len(path_data)):
            dx = path_data[i][0] - path_data[i-1][0]
            dy = path_data[i][1] - path_data[i-1][1]
            path_length += math.sqrt(dx**2 + dy**2)
        
        # Calculate direct distance
        direct = math.sqrt((path_data[-1][0] - path_data[0][0])**2 + 
                          (path_data[-1][1] - path_data[0][1])**2)
        
        return direct / path_length if path_length > 0 else 0.0
    
    def _analyze_indicators(self, metrics: PathMetrics) -> Dict[str, bool]:
        """Analyze individual behavioral indicators"""
        indicators = {}
        
        # Time-based indicators
        indicators['valid_solve_time'] = (self.thresholds['solve_time_min'] <= metrics.solve_time <= 
                                       self.thresholds['solve_time_max'])
        
        # Velocity-based indicators
        indicators['human_velocity_variance'] = metrics.velocity_variance >= self.thresholds['velocity_variance_min']
        indicators['reasonable_max_velocity'] = metrics.max_velocity < 500  # pixels per second
        
        # Movement pattern indicators
        indicators['sufficient_direction_changes'] = metrics.direction_changes >= self.thresholds['direction_changes_min']
        indicators['natural_jitter'] = metrics.jitter_magnitude >= self.thresholds['jitter_threshold']
        indicators['human_pauses'] = metrics.pause_count > 0
        indicators['few_instant_turns'] = metrics.instant_turn_count <= 1
        
        # Path efficiency indicators (too perfect = bot)
        indicators['not_perfectly_straight'] = metrics.straight_line_ratio < self.thresholds['straight_line_threshold']
        
        return indicators
    
    def _calculate_confidence(self, indicators: Dict[str, bool], metrics: PathMetrics) -> float:
        """Calculate overall confidence score based on all indicators"""
        human_indicators = [
            'valid_solve_time', 'human_velocity_variance', 'reasonable_max_velocity',
            'sufficient_direction_changes', 'natural_jitter', 'human_pauses',
            'few_instant_turns', 'not_perfectly_straight'
        ]
        
        human_score = sum(indicators.get(ind, False) for ind in human_indicators)
        max_score = len(human_indicators)
        
        base_confidence = human_score / max_score if max_score > 0 else 0
        
        # Apply learned pattern adjustments
        if self.human_patterns:
            pattern_similarity = self._compare_with_patterns(metrics)
            base_confidence = (base_confidence + pattern_similarity) / 2
        
        return base_confidence
    
    def _compare_with_patterns(self, metrics: PathMetrics) -> float:
        """Compare current metrics with learned human patterns"""
        if not self.human_patterns:
            return 0.5  # Neutral if no patterns learned
        
        # Calculate similarity with stored patterns
        similarities = []
        for pattern in self.human_patterns[-50:]:  # Use recent 50 patterns
            similarity = self._calculate_similarity(metrics, pattern)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_similarity(self, metrics1: PathMetrics, metrics2: PathMetrics) -> float:
        """Calculate similarity between two metric sets"""
        # Normalize and compare key metrics
        key_metrics = ['solve_time', 'velocity_variance', 'direction_changes', 
                     'jitter_magnitude', 'pause_count']
        
        similarities = []
        for metric in key_metrics:
            val1 = getattr(metrics1, metric)
            val2 = getattr(metrics2, metric)
            
            # Normalize similarity (0-1 scale)
            if max(val1, val2) > 0:
                similarity = 1 - abs(val1 - val2) / max(val1, val2)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _load_learned_patterns(self) -> None:
        """Load learned human patterns from database"""
        try:
            conn = sqlite3.connect('maze_captcha.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT solve_time, velocity_variance, direction_changes, 
                       jitter_magnitude, pause_count
                FROM user_paths 
                WHERE is_human = TRUE 
                ORDER BY created_at DESC 
                LIMIT 100
            """)
            
            patterns = cursor.fetchall()
            for pattern in patterns:
                self.human_patterns.append(PathMetrics(
                    solve_time=pattern[0],
                    velocity_variance=pattern[1],
                    direction_changes=pattern[2],
                    jitter_magnitude=pattern[3],
                    path_deviation=0,  # Not stored in current schema
                    straight_line_ratio=0,
                    pause_count=pattern[4],
                    avg_velocity=0,
                    max_velocity=0,
                    instant_turn_count=0
                ))
            
            conn.close()
            logger.info(f"Loaded {len(self.human_patterns)} learned human patterns")
            
        except Exception as e:
            logger.warning(f"Could not load learned patterns: {str(e)}")
    
    def _store_human_pattern(self, metrics: PathMetrics) -> None:
        """Store successful human pattern for future learning"""
        try:
            conn = sqlite3.connect('maze_captcha.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analytics (event_type, data, timestamp)
                VALUES (?, ?, ?)
            """, ('human_pattern', json.dumps(metrics.__dict__), datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not store human pattern: {str(e)}")

class HumanBuffer:
    """Implements tolerance for human-like behavior with wall collision detection"""
    
    def __init__(self, wall_touch_tolerance: float = 0.2):
        self.wall_touch_tolerance = wall_touch_tolerance  # 20% of path can touch walls
        self.min_wall_touches_allowed = 2  # Minimum wall touches for human behavior
        
    def validate_path(self, maze: np.ndarray, path: List[Tuple[int, int]], 
                    start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[bool, str, Dict]:
        """Validate path with human-like tolerance"""
        try:
            validation_result = {
                'is_valid': False,
                'wall_touches': 0,
                'out_of_bounds': 0,
                'path_length': len(path),
                'max_allowed_wall_touches': 0
            }
            
            # Check start and end points
            if not path or path[0] != start:
                return False, "Path must start at green point", validation_result
            
            if path[-1] != end:
                return False, "Path must end at red point", validation_result
            
            # Validate each path point
            wall_touches = 0
            out_of_bounds = 0
            
            for i, (r, c) in enumerate(path):
                # Check bounds
                if (r < 0 or r >= maze.shape[0] or 
                    c < 0 or c >= maze.shape[1]):
                    out_of_bounds += 1
                    continue
                
                # Check if on wall
                if maze[r, c] != 1:  # 0 = wall, 1 = path
                    wall_touches += 1
            
            # Calculate allowed wall touches
            max_allowed = max(self.min_wall_touches_allowed, 
                           int(len(path) * self.wall_touch_tolerance))
            
            validation_result.update({
                'wall_touches': wall_touches,
                'out_of_bounds': out_of_bounds,
                'max_allowed_wall_touches': max_allowed
            })
            
            # Determine validity
            if out_of_bounds > 0:
                return False, f"Path goes out of bounds ({out_of_bounds} times)", validation_result
            
            if wall_touches > max_allowed:
                return False, f"Too many wall touches ({wall_touches} > {max_allowed})", validation_result
            
            # Perfect paths (0 wall touches) are suspicious
            if wall_touches == 0:
                validation_result['suspicious_perfect'] = True
                return True, "Perfect path (suspicious)", validation_result
            
            validation_result['is_valid'] = True
            return True, f"Valid path ({wall_touches} wall touches)", validation_result
            
        except Exception as e:
            logger.error(f"Path validation error: {str(e)}")
            return False, f"Validation error: {str(e)}", {}

# Test the components
if __name__ == "__main__":
    logger.info("Testing Enhanced Behavioral Maze CAPTCHA System")
    
    # Test maze generation
    generator = MazeGenerator(size=15)
    maze = generator.generate_maze()
    logger.info(f"Generated maze: {maze.shape}")
    
    # Test pathfinding
    if generator.solution_path:
        logger.info(f"Solution found with {len(generator.solution_path)} steps")
    else:
        logger.error("No solution found!")
    
    # Test behavioral analyzer
    analyzer = BehavioralAnalyzer()
    logger.info("Behavioral analyzer initialized")
    
    # Test human buffer
    buffer = HumanBuffer()
    logger.info("Human buffer initialized")
    
    logger.info("All components initialized successfully!")