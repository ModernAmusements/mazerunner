import cv2
import numpy as np
import heapq
import random
import sys
import time

# Maximize recursion for large generation
sys.setrecursionlimit(10**6)

# --- CONFIG ---
CONFIG = {
    "maze_size": 41,
    "cell_size": 15,
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
    "speed": 0.5, # 0.0 to 1.0
    "is_dragging_speed": False
}

# --- 1. GENERATOR ---
def generate_maze(rows, cols):
    rows, cols = (rows // 2 * 2 + 1), (cols // 2 * 2 + 1)
    maze = np.zeros((rows, cols), dtype=np.uint8)
    def walk(r, c):
        maze[r, c] = 1
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows-1 and 0 < nc < cols-1 and maze[nr, nc] == 0:
                maze[r + dr // 2, c + dc // 2] = 1
                walk(nr, nc)
    walk(1, 1)
    return maze

# --- 2. SOLVER WITH VARIABLE SPEED ---
def solve_astar_animated(grid, start, end):
    rows, cols = grid.shape
    pq = [(0, start)]
    g_score = {start: 0}
    parent = {}
    explored = set()
    
    moves = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    weights = [1.0, 1.0, 1.0, 1.0, 1.41, 1.41, 1.41, 1.41]
    
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
            if cv2.waitKey(delay) & 0xFF == 27: return None, explored, 0

    return None, explored, 0

# --- 3. UI RENDERING ---
def draw_maze_state(grid, path=None, explored=None, frontier=None):
    rows, cols = grid.shape
    s = CONFIG["cell_size"]
    img = np.zeros((rows * s, cols * s, 3), dtype=np.uint8)
    
    for r in range(rows):
        for c in range(cols):
            color = CONFIG["colors"]["path_bg"] if grid[r, c] == 1 else CONFIG["colors"]["wall"]
            cv2.rectangle(img, (c*s, r*s), ((c+1)*s, (r+1)*s), color, -1)

    if explored:
        for (r, c) in explored:
            cv2.rectangle(img, (c*s+1, r*s+1), ((c+1)*s-1, (r+1)*s-1), CONFIG["colors"]["explored"], -1)
            
    if frontier:
        for (r, c) in frontier:
            cv2.rectangle(img, (c*s+1, r*s+1), ((c+1)*s-1, (r+1)*s-1), CONFIG["colors"]["frontier"], -1)

    cv2.rectangle(img, (1*s, 1*s), (2*s, 2*s), CONFIG["colors"]["start"], -1)
    cv2.rectangle(img, ((cols-2)*s, (rows-2)*s), ((cols-1)*s, (rows-1)*s), CONFIG["colors"]["end"], -1)

    if path:
        for i in range(len(path)-1):
            p1 = (path[i][1]*s + s//2, path[i][0]*s + s//2)
            p2 = (path[i+1][1]*s + s//2, path[i+1][0]*s + s//2)
            cv2.line(img, p1, p2, CONFIG["colors"]["solution"], 2, cv2.LINE_AA)
    return img

def assemble_dashboard(maze_img, status_text=None):
    h, w = maze_img.shape[:2]
    canvas = np.zeros((max(h, 500), w + CONFIG["sidebar_width"], 3), dtype=np.uint8)
    canvas[:] = CONFIG["colors"]["bg"]
    canvas[0:h, 0:w] = maze_img
    
    sx = w + 20
    status = status_text if status_text else state["status"]
    cv2.putText(canvas, "A* MAZE PRO", (sx, 40), 1, 1.5, CONFIG["colors"]["accent"], 2)
    cv2.putText(canvas, f"Status: {status}", (sx, 80), 1, 0.9, (0, 255, 255), 1)
    
    # Buttons
    mx, my = state["mouse_pos"]
    for i, (txt, y) in enumerate([("REGEN (R)", 140), ("SOLVE (S)", 190), ("SIZE +", 240)]):
        hover = (w + 20 <= mx <= w + 200) and (y <= my <= y + 35)
        clr = CONFIG["colors"]["accent"] if hover else (60, 60, 60)
        cv2.rectangle(canvas, (w + 20, y), (w + 200, y + 35), clr, -1)
        cv2.putText(canvas, txt, (w + 50, y + 23), 1, 0.9, (255,255,255), 1)

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
    state["mouse_pos"] = (x, y)
    w = maze_grid.shape[1] * CONFIG["cell_size"]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check buttons
        if w + 20 <= x <= w + 200:
            if 140 <= y <= 175: state["regen"] = True
            elif 190 <= y <= 225: state["solve"] = True
            elif 240 <= y <= 275:
                CONFIG["maze_size"] = min(CONFIG["maze_size"] + 20, 101)
                state["regen"] = True
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
cv2.namedWindow("Maze Dashboard")
cv2.setMouseCallback("Maze Dashboard", mouse_handler)

maze_grid = None
display_maze = None

while True:
    if state["regen"]:
        state["regen"], state["solve"] = False, False
        state["status"] = "Ready"
        state["path"], state["explored"] = [], set()
        maze_grid = generate_maze(CONFIG["maze_size"], CONFIG["maze_size"])
        display_maze = draw_maze_state(maze_grid)

    if state["solve"]:
        state["solve"] = False
        start_node, end_node = (1, 1), (maze_grid.shape[0]-2, maze_grid.shape[1]-2)
        path, explored, dur = solve_astar_animated(maze_grid, start_node, end_node)
        
        if path:
            state["path"], state["explored"] = path, explored
            state["status"] = f"Solved: {dur:.2f}s"
            display_maze = draw_maze_state(maze_grid, path=path, explored=explored)
        else:
            state["status"] = "No Path Found"

    cv2.imshow("Maze Dashboard", assemble_dashboard(display_maze))
    key = cv2.waitKey(10) & 0xFF
    if key == 27: break
    if key == ord('r'): state["regen"] = True
    if key == ord('s'): state["solve"] = True

cv2.destroyAllWindows()