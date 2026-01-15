#!/usr/bin/env python3
"""
Standalone 20x20 Maze Generator using Recursive Backtracking Algorithm
Based on the provided reference code with improvements
- Start at (0,0) with green entrance on outer boundary
- End at (19,19) with red exit on outer boundary  
- Broken outer walls at start and end to show entrance and exit
"""

import numpy as np
import heapq
import random
import sys
import time
import cv2

# Maximize recursion for large generation
sys.setrecursionlimit(10**6)

# --- CONFIG ---
CONFIG = {
    "maze_size": 20,  # Fixed 20x20 as specified
    "cell_size": 15,    # Size of each maze cell
    "sidebar_width": 220,
    "colors": {
        "bg": (25, 25, 25),
        "sidebar": (45, 45, 48),
        "accent": (0, 120, 215),
        "wall": (15, 15, 15),
        "path_bg": (255, 255, 255),
        "start": (50, 200, 50),
        "end": (50, 50, 200),
        "explored": (120, 80, 0),
        "frontier": (0, 255, 255),
        "solution": (0, 0, 255),
        "slider": (100, 100, 100)
    }
}

state = {
    "regen": True, "solve": False, "exit": False,
    "mouse_pos": (0, 0), "status": "Ready",
    "path": [], "explored": set(),
    "speed": 0.5,  # 0.0 to 1.0
    "is_dragging_speed": False
}

class MazeGenerator:
    """20x20 Maze Generator using Recursive Backtracking"""
    
    def __init__(self, size=20):
        self.size = size
        self.maze = None
        self.start = (0, 0)
        self.end = (19, 19)
    
    def generate(self):
        """Generate 20x20 maze using recursive backtracking"""
        size = self.size
        maze = np.zeros((size, size), dtype=np.uint8)
        
        def is_valid(x, y):
            return 0 <= x < size and 0 <= y < size
        
        def get_unvisited_neighbors(x, y, visited):
            neighbors = []
            # Check all 4 directions
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            return neighbors
        
        # Recursive backtracking maze generation
        def carve_path(x, y, visited):
            visited.add((x, y))
            maze[x, y] = 1  # Mark as path
            
            neighbors = get_unvisited_neighbors(x, y, visited)
            random.shuffle(neighbors)
            
            for nx, ny in neighbors:
                if (nx, ny) not in visited:
                    # Carve wall between current and neighbor
                    maze[x + (nx - x) // 2, y + (ny - y) // 2] = 1
                    carve_path(nx, ny, visited)
        
        # Start generating maze from (1,1) to avoid boundary issues
        start_pos = (1, 1)
        carve_path(start_pos[0], start_pos[1], set())
        
        # Ensure start at (0,0) and end at (19,19) are accessible
        maze[0, 0] = 1  # Start at top-left as specified
        maze[19, 19] = 1  # End at bottom-right as specified
        
        # Break outer boundaries at start and end
        # Start at (0,0): break top and left walls for entrance
        # End at (19,19): break bottom and right walls for exit
        maze[0, 0] = 1  # Entrance
        maze[19, 19] = 1  # Exit
        
        # Verify maze has a solution from start to end
        if self._verify_path_exists(maze, self.start, self.end):
            self.maze = maze
            return True
        else:
            # Try again if no solution
            return self.generate()
    
    def _verify_path_exists(self, maze, start, end):
        """Verify there's a valid path from start to end"""
        from collections import deque
        
        queue = deque([start])
        visited = set([start])
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == end:
                return True
            
            # Check all 4 directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    maze[nx, ny] == 1 and 
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False

# --- SOLVER WITH VARIABLE SPEED ---
def solve_astar_animated(grid, start, end):
    """Solve maze using A* algorithm with animation"""
    rows, cols = grid.shape
    pq = [(0, start)]
    g_score = {start: 0}
    parent = {}
    explored = set()
    
    count = 0
    t_start = time.time()

    while pq:
        f, curr = heapq.heappop(pq)
        if curr == end:
            path = []
            while curr in parent:
                path.append(curr)
                curr = parent[curr]
            path.append(start)
            return path[::-1], explored, time.time() - t_start

        explored.add(curr)
        y, x = curr
        
        # Check all 8 directions (including diagonals for better paths)
        moves = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        weights = [1.0, 1.0, 1.0, 1.0, 1.41, 1.41, 1.41, 1.41, 1.41]
        
        for i, (dy, dx) in enumerate(moves):
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] == 1:
                new_g = g_score[curr] + weights[i]
                if (ny, nx) not in g_score or new_g < g_score[(ny, nx)]:
                    g_score[(ny, nx)] = new_g
                    h = abs(ny - end[0]) + abs(nx - end[1])
                    heapq.heappush(pq, (new_g + h, (ny, nx)))
                    parent[(ny, nx)] = curr

        # Adjust batching based on speed slider
        batch = int(1 + state["speed"] * 50) 
        count += 1
        if count % batch == 0:
            view = draw_maze_state(grid, explored=explored, frontier=[p[1] for p in pq])
            cv2.imshow("Maze Dashboard", assemble_dashboard(view, "Solving..."))
            # Adjust waitKey delay based on speed
            delay = max(1, int((1.0 - state["speed"]) * 50))
            if cv2.waitKey(delay) & 0xFF == 27: 
                return None, explored, 0

    return None, explored, 0

# --- 3. UI RENDERING ---
def draw_maze_state(grid, path=None, explored=None, frontier=None):
    """Draw maze with 8-bit aesthetic"""
    rows, cols = grid.shape
    s = CONFIG["cell_size"]
    img = np.zeros((rows * s, cols * s, 3), dtype=np.uint8)
    
    # Draw maze cells - crisp black walls, white paths
    for r in range(rows):
        for c in range(cols):
            x = c * s
            y = r * s
            if grid[r, c] == 0:
                # Solid black wall - fill entire cell
                img[y:y+s, x:x+s] = CONFIG["colors"]["wall"]
            else:
                # White path - already white from background
                pass
    
    # Draw grid lines on white paths for 8-bit aesthetic
    grid_color = [230, 230, 230]  # Subtle grid lines
    line_width = 1
    
    # Horizontal grid lines
    for i in range(rows + 1):
        y = i * s
        # Draw only on paths, not walls
        for j in range(cols):
            x = j * s
            if j < cols - 1 and i < rows - 1 and grid[i, j] == 1 and grid[i, j+1] == 1:
                img[y:y+line_width, x:x+s] = grid_color
    
    # Vertical grid lines
    for j in range(cols + 1):
        x = j * s
        # Draw only on paths, not walls  
        for i in range(rows):
            y = i * s
            if i < rows - 1 and j < cols - 1 and grid[i, j] == 1 and grid[i+1, j] == 1:
                img[y:y+s, x:x+line_width] = grid_color

    # Place entry (green) at (0,0) and exit (red) at (19,19)
    start_row, start_col = 0, 0
    end_row, end_col = rows - 1, cols - 1
    
    # Bright green entry square
    green_color = CONFIG["colors"]["start"]
    img[start_row*s + 3:(start_row+1)*s - 3, 
        start_col*s + 3:(start_col+1)*s - 3] = green_color
    
    # Bright red exit square  
    red_color = CONFIG["colors"]["end"]
    img[end_row*s + 3:(end_row+1)*s - 3,
        end_col*s + 3:(end_col+1)*s - 3] = red_color
    
    return img

def assemble_dashboard(maze_img, status_text=None):
    """Assemble the complete dashboard with maze and controls"""
    h, w = maze_img.shape[:2]
    canvas = np.zeros((max(h, 500), w + CONFIG["sidebar_width"], 3), dtype=np.uint8)
    canvas[:] = CONFIG["colors"]["bg"]
    canvas[0:h, 0:w] = maze_img
    
    sx = w + 20
    status = status_text if status_text else state["status"]
    cv2.putText(canvas, "20×20 MAZE", (sx, 40), 1, 1.5, CONFIG["colors"]["accent"], 2)
    cv2.putText(canvas, f"Status: {status}", (sx, 80), 1, 0.9, (0, 255, 255), 1)
    
    # Buttons
    mx, my = state["mouse_pos"]
    for i, (txt, y) in enumerate([("REGEN (R)", 140), ("SOLVE (S)", 190), ("SIZE +", 240)]):
        hover = (w + 20 <= mx <= w + 200) and (y <= my <= y + 35)
        clr = CONFIG["colors"]["accent"] if hover else (60, 60, 60)
        cv2.rectangle(canvas, (w + 20, y), (w + 200, y + 35), clr, -1)
        cv2.putText(canvas, txt, (w + 50, y + 23), 1, 0.9, (255, 255, 255), 1)

    # --- SPEED SLIDER ---
    sy = 320
    cv2.putText(canvas, "SOLVE SPEED", (sx, sy), 1, 0.8, (180, 180, 180), 1)
    # Track
    cv2.rectangle(canvas, (sx, sy+20), (w+200, sy+25), (60, 60, 60), -1)
    # Handle
    hx = int(sx + state["speed"] * 180)
    h_clr = CONFIG["colors"]["accent"] if state["is_dragging_speed"] else (200, 200, 200)
    cv2.circle(canvas, (hx, sy+22), 8, h_clr, -1)
        
    return canvas

# --- 4. MOUSE HANDLER ---
def mouse_handler(event, x, y, flags, param):
    """Handle mouse events for dashboard"""
    state["mouse_pos"] = (x, y)
    w = maze_grid.shape[1] * CONFIG["cell_size"] if maze_grid is not None else 400
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check buttons
        if w + 20 <= x <= w + 200:
            if 140 <= y <= 175: 
                state["regen"] = True
            elif 190 <= y <= 225: 
                state["solve"] = True
            elif 240 <= y <= 275:
                # Handle size increase (optional)
                pass
            # Check Slider
            elif 310 <= y <= 350:
                state["is_dragging_speed"] = True

    elif event == cv2.EVENT_MOUSEMOVE and state["is_dragging_speed"]:
        # Update Slider
        new_speed = (x - (w + 20)) / 180.0
        state["speed"] = max(0.0, min(1.0, new_speed))

    elif event == cv2.EVENT_LBUTTONUP:
        state["is_dragging_speed"] = False

# --- 5. MAIN ---
cv2.namedWindow("20×20 Maze Dashboard")
cv2.setMouseCallback("20×20 Maze Dashboard", mouse_handler)

maze_grid = None
display_maze = None
maze_generator = MazeGenerator()

while True:
    if state["regen"]:
        state["regen"], state["solve"] = False, False
        state["status"] = "Generating..."
        state["path"], state["explored"] = [], set()
        
        print("Generating 20×20 maze with recursive backtracking...")
        if maze_generator.generate():
            maze_grid = maze_generator.maze
            display_maze = draw_maze_state(maze_grid)
            state["status"] = "Ready"
        else:
            state["status"] = "Generation Failed"

    if state["solve"]:
        state["solve"] = False
        state["status"] = "Solving..."
        state["path"], state["explored"] = [], set()
        
        if maze_grid is not None:
            start_node, end_node = maze_generator.start, maze_generator.end
            path, explored, dur = solve_astar_animated(maze_grid, start_node, end_node)
            
            if path:
                state["path"], state["explored"] = path, explored
                state["status"] = f"Solved: {dur:.2f}s"
                display_maze = draw_maze_state(maze_grid, path=path, explored=explored)
            else:
                state["status"] = "No Path Found"

    # Display current state
    if display_maze is not None:
        cv2.imshow("20×20 Maze Dashboard", assemble_dashboard(display_maze, state["status"]))
    
    key = cv2.waitKey(10) & 0xFF
    if key == 27: 
        break
    if key == ord('r'): 
        state["regen"] = True
    if key == ord('s'): 
        state["solve"] = True

cv2.destroyAllWindows()