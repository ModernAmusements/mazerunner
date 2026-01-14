import cv2
import numpy as np
import heapq
import random
import sys
import time

# Maximize recursion
sys.setrecursionlimit(10**6)

CONFIG = {
    "maze_size": 41,
    "cell_size": 15,
    "sidebar_width": 220,
    "colors": {
        "bg": (25, 25, 25),
        "sidebar": (45, 45, 48),
        "accent": (0, 120, 215),
        "start": (50, 255, 50),
        "end": (50, 50, 255),
        "solution": (255, 255, 255),
        # Map Weights
        "highway": (0, 150, 0),    # Cost 1
        "road": (0, 150, 255),     # Cost 4
        "traffic": (0, 0, 150),    # Cost 15
        "building": (15, 15, 15)   # Impassable
    }
}

state = {
    "regen": True, "solve": False, "mode": "MAP", # "MAZE" or "MAP"
    "mouse_pos": (0, 0), "status": "Ready",
    "speed": 0.6, "is_dragging_speed": False,
    "path": [], "explored": set()
}

# --- 1. GENERATORS ---
def generate_map_grid(rows, cols):
    # 0: Building, 1: Highway, 5: Road, 15: Traffic
    grid = np.full((rows, cols), 5, dtype=np.uint8) 
    
    # Add random buildings (walls)
    for _ in range(int(rows*cols*0.1)):
        grid[random.randint(0, rows-1), random.randint(0, cols-1)] = 0
        
    # Add "Highways" (Fast routes)
    for _ in range(4):
        r = random.randint(0, rows-1)
        grid[r, :] = 1
        c = random.randint(0, cols-1)
        grid[:, c] = 1

    # Add "Traffic Jams" (High cost)
    for _ in range(5):
        r, c = random.randint(5, rows-5), random.randint(5, cols-5)
        grid[r-2:r+2, c-2:c+2] = 15
        
    grid[1, 1] = 1 # Ensure start is clear
    grid[rows-2, cols-2] = 1 # Ensure end is clear
    return grid

# --- 2. THE WEIGHTED SOLVER ---
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
            if 0 <= ny < rows and 0 <= nx < cols:
                cell_weight = grid[ny, nx]
                if cell_weight == 0: continue # Building/Wall
                
                # COST = Distance Weight * Terrain Weight
                move_cost = weights[i] * cell_weight
                new_g = g_score[curr] + move_cost
                
                if (ny, nx) not in g_score or new_g < g_score[(ny, nx)]:
                    g_score[(ny, nx)] = new_g
                    h = (abs(ny - end[0]) + abs(nx - end[1])) # Manhattan
                    heapq.heappush(pq, (new_g + h, (ny, nx)))
                    parent[(ny, nx)] = curr

        batch = int(1 + state["speed"] * 60)
        count += 1
        if count % batch == 0:
            view = draw_map_state(grid, explored=explored, frontier=[p[1] for p in pq])
            cv2.imshow("Route Simulator", assemble_dashboard(view, "Routing..."))
            delay = max(1, int((1.0 - state["speed"]) * 40))
            if cv2.waitKey(delay) & 0xFF == 27: return None, explored, 0

    return None, explored, 0

# --- 3. RENDERING ---
def draw_map_state(grid, path=None, explored=None, frontier=None):
    rows, cols = grid.shape
    s = CONFIG["cell_size"]
    img = np.zeros((rows * s, cols * s, 3), dtype=np.uint8)
    
    color_map = {1: CONFIG["colors"]["highway"], 5: CONFIG["colors"]["road"], 
                 15: CONFIG["colors"]["traffic"], 0: CONFIG["colors"]["building"]}

    for r in range(rows):
        for c in range(cols):
            cv2.rectangle(img, (c*s, r*s), ((c+1)*s, (r+1)*s), color_map[grid[r, c]], -1)

    if explored:
        for (r, c) in explored:
            overlay = img[r*s:(r+1)*s, c*s:(c+1)*s].copy()
            cv2.rectangle(overlay, (0,0), (s,s), (255, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, img[r*s:(r+1)*s, c*s:(c+1)*s], 0.7, 0, img[r*s:(r+1)*s, c*s:(c+1)*s])
            
    cv2.rectangle(img, (1*s, 1*s), (2*s, 2*s), CONFIG["colors"]["start"], -1)
    cv2.rectangle(img, ((cols-2)*s, (rows-2)*s), ((cols-1)*s, (rows-1)*s), CONFIG["colors"]["end"], -1)

    if path:
        for i in range(len(path)-1):
            p1 = (path[i][1]*s + s//2, path[i][0]*s + s//2)
            p2 = (path[i+1][1]*s + s//2, path[i+1][0]*s + s//2)
            cv2.line(img, p1, p2, CONFIG["colors"]["solution"], 3, cv2.LINE_AA)
    return img

def assemble_dashboard(maze_img, status_text=None):
    h, w = maze_img.shape[:2]
    canvas = np.zeros((max(h, 550), w + CONFIG["sidebar_width"], 3), dtype=np.uint8)
    canvas[:] = CONFIG["colors"]["bg"]
    canvas[0:h, 0:w] = maze_img
    
    sx = w + 20
    cv2.putText(canvas, "GPS SIMULATOR", (sx, 40), 1, 1.4, CONFIG["colors"]["accent"], 2)
    cv2.putText(canvas, f"Mode: {state['mode']}", (sx, 75), 1, 0.9, (200, 200, 200), 1)
    cv2.putText(canvas, f"Status: {status_text if status_text else state['status']}", (sx, 100), 1, 0.8, (0, 255, 255), 1)
    
    # Buttons
    mx, my = state["mouse_pos"]
    for i, (txt, y) in enumerate([("NEW MAP (R)", 140), ("FIND ROUTE (S)", 190), ("TOGGLE MODE (M)", 240)]):
        hover = (w + 20 <= mx <= w + 200) and (y <= my <= y + 35)
        clr = CONFIG["colors"]["accent"] if hover else (60, 60, 60)
        cv2.rectangle(canvas, (w + 20, y), (w + 200, y + 35), clr, -1)
        cv2.putText(canvas, txt, (w + 35, y + 23), 1, 0.8, (255,255,255), 1)

    # Legend
    ly = 300
    for txt, clr in [("Highway (Fast)", "highway"), ("Road (Normal)", "road"), ("Traffic (Slow)", "traffic")]:
        cv2.rectangle(canvas, (sx, ly), (sx+15, ly+15), CONFIG["colors"][clr], -1)
        cv2.putText(canvas, txt, (sx+25, ly+13), 1, 0.7, (180, 180, 180), 1)
        ly += 25

    # Speed Slider
    sy = 420
    cv2.putText(canvas, "SIM SPEED", (sx, sy), 1, 0.8, (180, 180, 180), 1)
    cv2.rectangle(canvas, (sx, sy+20), (w+200, sy+25), (60, 60, 60), -1)
    hx = int(sx + state["speed"] * 180)
    cv2.circle(canvas, (hx, sy+22), 8, CONFIG["colors"]["accent"] if state["is_dragging_speed"] else (200, 200, 200), -1)
    return canvas

# --- 4. INTERACTION ---
def mouse_handler(event, x, y, flags, param):
    state["mouse_pos"] = (x, y)
    w = maze_grid.shape[1] * CONFIG["cell_size"]
    if event == cv2.EVENT_LBUTTONDOWN:
        if w + 20 <= x <= w + 200:
            if 140 <= y <= 175: state["regen"] = True
            elif 190 <= y <= 225: state["solve"] = True
            elif 240 <= y <= 275: 
                state["mode"] = "MAZE" if state["mode"] == "MAP" else "MAP"
                state["regen"] = True
            elif 410 <= y <= 450: state["is_dragging_speed"] = True
    elif event == cv2.EVENT_MOUSEMOVE and state["is_dragging_speed"]:
        state["speed"] = max(0.0, min(1.0, (x - (w + 20)) / 180.0))
    elif event == cv2.EVENT_LBUTTONUP:
        state["is_dragging_speed"] = False

# --- 5. MAIN ---
cv2.namedWindow("Route Simulator")
cv2.setMouseCallback("Route Simulator", mouse_handler)
maze_grid = None
display_maze = None

while True:
    if state["regen"]:
        state["regen"], state["solve"] = False, False
        state["status"] = "Ready"
        maze_grid = generate_map_grid(CONFIG["maze_size"], CONFIG["maze_size"])
        display_maze = draw_map_state(maze_grid)

    if state["solve"]:
        state["solve"] = False
        start_node, end_node = (1, 1), (maze_grid.shape[0]-2, maze_grid.shape[1]-2)
        path, explored, dur = solve_astar_animated(maze_grid, start_node, end_node)
        if path:
            state["status"] = f"Time: {dur:.2f}s"
            display_maze = draw_map_state(maze_grid, path=path, explored=explored)

    cv2.imshow("Route Simulator", assemble_dashboard(display_maze))
    key = cv2.waitKey(10) & 0xFF
    if key == 27: break
    if key == ord('m'): 
        state["mode"] = "MAZE" if state["mode"] == "MAP" else "MAP"
        state["regen"] = True

cv2.destroyAllWindows()