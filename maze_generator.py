#!/usr/bin/env python3
"""
20x20 Maze Generator using Recursive Backtracking Algorithm
- Start at (0,0) with green entrance on outer boundary
- End at (19,19) with red exit on outer boundary
- Renders to matplotlib window with visual highlighting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from typing import List, Tuple, Optional

class Cell:
    """Represents a maze cell with walls"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.visited = False
        self.walls = {
            'top': True,
            'right': True,
            'bottom': True,
            'left': True
        }
    
    def __repr__(self):
        return f"Cell({self.x},{self.y})"

class MazeGenerator:
    """20x20 Maze Generator using Recursive Backtracking"""
    
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.grid = [[Cell(x, y) for y in range(height)] for x in range(width)]
        self.start = (0, 0)
        self.end = (19, 19)
        
    def get_cell(self, x: int, y: int) -> Optional[Cell]:
        """Get cell at coordinates, return None if out of bounds"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y]
        return None
    
    def get_unvisited_neighbors(self, cell: Cell) -> List[Tuple[Cell, str]]:
        """Get unvisited neighbors with wall direction"""
        neighbors = []
        x, y = cell.x, cell.y
        
        # Check all four directions
        directions = [
            ((x, y - 1), 'top'),      # North
            ((x + 1, y), 'right'),    # East  
            ((x, y + 1), 'bottom'),   # South
            ((x - 1, y), 'left')      # West
        ]
        
        for (nx, ny), direction in directions:
            neighbor = self.get_cell(nx, ny)
            if neighbor and not neighbor.visited:
                neighbors.append((neighbor, direction))
        
        return neighbors
    
    def remove_wall(self, cell1: Cell, cell2: Cell, direction: str):
        """Remove wall between two adjacent cells"""
        opposite = {
            'top': 'bottom',
            'right': 'left', 
            'bottom': 'top',
            'left': 'right'
        }
        
        cell1.walls[direction] = False
        cell2.walls[opposite[direction]] = False
    
    def generate(self):
        """Generate maze using recursive backtracking"""
        # Reset all cells
        for row in self.grid:
            for cell in row:
                cell.visited = False
                cell.walls = {
                    'top': True,
                    'right': True,
                    'bottom': True,
                    'left': True
                }
        
        # Start recursive generation from (0,0)
        self.carve_path(0, 0)
        
        # Break outer boundaries at start and end
        self.create_entrances()
    
    def carve_path(self, x: int, y: int):
        """Recursively carve path through maze"""
        current = self.get_cell(x, y)
        if not current:
            return
            
        current.visited = True
        
        # Get unvisited neighbors and shuffle for randomness
        neighbors = self.get_unvisited_neighbors(current)
        random.shuffle(neighbors)
        
        # Visit each neighbor
        for neighbor, direction in neighbors:
            if not neighbor.visited:
                # Remove wall between current and neighbor
                self.remove_wall(current, neighbor, direction)
                
                # Recursively visit neighbor
                self.carve_path(neighbor.x, neighbor.y)
    
    def create_entrances(self):
        """Break outer walls at start and end points"""
        start_cell = self.get_cell(*self.start)
        end_cell = self.get_cell(*self.end)
        
        if start_cell:
            start_cell.walls['left'] = False  # Entrance on left boundary
        if end_cell:
            end_cell.walls['right'] = False   # Exit on right boundary


class MazeRenderer:
    """Renders maze to matplotlib window"""
    
    def __init__(self, maze: MazeGenerator, cell_size: float = 1.0):
        self.maze = maze
        self.cell_size = cell_size
        
        # Colors
        self.colors = {
            'background': 'white',
            'wall': 'black',
            'start': 'green',
            'end': 'red',
        }
    
    def draw_cell_walls(self, ax, cell: Cell):
        """Draw walls for a single cell"""
        x = cell.x * self.cell_size
        y = cell.y * self.cell_size
        size = self.cell_size
        
        wall_color = self.colors['wall']
        wall_width = 2
        
        # Draw each wall if it exists
        if cell.walls['top']:
            ax.plot([x, x + size], [y, y], color=wall_color, linewidth=wall_width)
        
        if cell.walls['right']:
            ax.plot([x + size, x + size], [y, y + size], 
                   color=wall_color, linewidth=wall_width)
        
        if cell.walls['bottom']:
            ax.plot([x + size, x], [y + size, y + size], 
                   color=wall_color, linewidth=wall_width)
        
        if cell.walls['left']:
            ax.plot([x, x], [y + size, y], 
                   color=wall_color, linewidth=wall_width)
    
    def draw_start_end(self, ax):
        """Highlight start and end cells"""
        size = self.cell_size
        
        # Draw start cell in green (0,0)
        start_x = self.maze.start[0] * size
        start_y = self.maze.start[1] * size
        start_rect = patches.Rectangle(
            (start_x + 0.1, start_y + 0.1), size - 0.2, size - 0.2,
            linewidth=0, facecolor=self.colors['start'], alpha=0.7
        )
        ax.add_patch(start_rect)
        
        # Draw end cell in red (19,19)
        end_x = self.maze.end[0] * size
        end_y = self.maze.end[1] * size
        end_rect = patches.Rectangle(
            (end_x + 0.1, end_y + 0.1), size - 0.2, size - 0.2,
            linewidth=0, facecolor=self.colors['end'], alpha=0.7
        )
        ax.add_patch(end_rect)
    
    def draw_outer_boundary(self, ax):
        """Draw outer boundary of the maze with openings"""
        x = 0
        y = 0
        width = self.maze.width * self.cell_size
        height = self.maze.height * self.cell_size
        
        wall_color = self.colors['wall']
        wall_width = 3
        
        # Top wall (full)
        ax.plot([x, x + width], [y, y], color=wall_color, linewidth=wall_width)
        
        # Bottom wall (full)
        ax.plot([x, x + width], [y + height, y + height], 
               color=wall_color, linewidth=wall_width)
        
        # Left wall (with opening at start)
        start_y = y + self.maze.start[1] * self.cell_size
        ax.plot([x, x], [y, start_y], color=wall_color, linewidth=wall_width)
        ax.plot([x, x], [start_y + self.cell_size, y + height], 
               color=wall_color, linewidth=wall_width)
        
        # Right wall (with opening at end)
        end_y = y + self.maze.end[1] * self.cell_size
        ax.plot([x + width, x + width], [y, end_y], 
               color=wall_color, linewidth=wall_width)
        ax.plot([x + width, x + width], [end_y + self.cell_size, y + height], 
               color=wall_color, linewidth=wall_width)
    
    def render(self, ax):
        """Render complete maze to axes"""
        # Clear and set background
        ax.clear()
        ax.set_facecolor(self.colors['background'])
        
        # Draw outer boundary with entrances
        self.draw_outer_boundary(ax)
        
        # Draw cell walls
        for row in self.maze.grid:
            for cell in row:
                self.draw_cell_walls(ax, cell)
        
        # Highlight start and end cells
        self.draw_start_end(ax)
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, self.maze.width * self.cell_size - 0.5)
        ax.set_ylim(-0.5, self.maze.height * self.cell_size - 0.5)
        ax.set_title(f"20x20 Maze - Recursive Backtracking\n"
                    f"Start: {self.maze.start} (Green) | End: {self.maze.end} (Red)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")


def main():
    """Main function to generate and display maze"""
    print("Generating 20x20 maze using Recursive Backtracking...")
    
    # Create maze generator
    maze = MazeGenerator(20, 20)
    print(f"Start: {maze.start} (Green)")
    print(f"End: {maze.end} (Red)")
    
    # Generate maze
    maze.generate()
    print("Maze generated successfully!")
    print("Press SPACE to generate new maze, Close window to exit")
    
    # Create renderer and display
    renderer = MazeRenderer(maze, cell_size=1.0)
    
    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def on_key(event):
        if event.key == ' ':
            # Generate new maze
            print("Generating new maze...")
            maze.generate()
            renderer.render(ax)
            plt.draw()
    
    # Connect keyboard event
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial render
    renderer.render(ax)
    plt.show()


if __name__ == "__main__":
    main()