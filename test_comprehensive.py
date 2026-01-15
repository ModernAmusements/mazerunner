#!/usr/bin/env python3
"""
Comprehensive test suite for maze captcha system
"""
import sys
import os
sys.path.append('.')

def test_maze_generation():
    """Test maze generation function"""
    print("=== Testing Maze Generation ===")
    try:
        from working_maze_server import generate_maze
        maze, solution_path = generate_maze()
        
        # Check maze properties
        assert maze.shape == (20, 20), f"Expected (20,20), got {maze.shape}"
        assert maze.any(), "Maze is all zeros"
        assert len(solution_path) == 2, f"Expected 2 points, got {len(solution_path)}"
        assert solution_path[0] == (0, 0), f"Expected start (0,0), got {solution_path[0]}"
        assert solution_path[1] == (19, 19), f"Expected end (19,19), got {solution_path[1]}"
        
        print("✅ Maze generation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Maze generation: FAILED - {e}")
        return False

def test_maze_generator_class():
    """Test MazeGenerator class"""
    print("\n=== Testing MazeGenerator Class ===")
    try:
        from maze_generator import MazeGenerator
        maze = MazeGenerator(20, 20)
        maze.generate()
        
        # Check properties
        assert maze.width == 20, f"Expected width 20, got {maze.width}"
        assert maze.height == 20, f"Expected height 20, got {maze.height}"
        assert maze.start == (0, 0), f"Expected start (0,0), got {maze.start}"
        assert maze.end == (19, 19), f"Expected end (19,19), got {maze.end}"
        
        # Check grid has cells
        assert len(maze.grid) == 20, f"Expected 20 rows, got {len(maze.grid)}"
        assert len(maze.grid[0]) == 20, f"Expected 20 cols, got {len(maze.grid[0])}"
        
        # Check visited cells during generation
        visited_count = sum(1 for row in maze.grid for cell in row if cell.visited)
        assert visited_count > 100, f"Too few visited cells: {visited_count}"
        
        print("✅ MazeGenerator class: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ MazeGenerator class: FAILED - {e}")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("\n=== Testing Dependencies ===")
    required_modules = [
        'flask', 'numpy', 'cv2', 'requests', 'matplotlib', 
        'base64', 'json', 'random', 'time'
    ]
    
    failed = []
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'matplotlib':
                import matplotlib
            else:
                __import__(module)
        except ImportError:
            failed.append(module)
    
    if failed:
        print(f"❌ Dependencies: FAILED - Missing {failed}")
        return False
    else:
        print("✅ Dependencies: PASSED")
        return True

def test_api_endpoints():
    """Test API endpoints"""
    print("\n=== Testing API Endpoints ===")
    try:
        import requests
        import time
        
        # Test captcha generation
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        assert response.status_code == 200, f"Captcha API failed: {response.status_code}"
        
        data = response.json()
        required_fields = ['captcha_id', 'maze_image', 'start', 'end', 'canvas_start', 'canvas_end', 'difficulty']
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Test analytics
        response = requests.get("http://127.0.0.1:8080/api/analytics")
        assert response.status_code == 200, f"Analytics API failed: {response.status_code}"
        
        print("✅ API endpoints: PASSED")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ API endpoints: FAILED - Server not running")
        return False
    except Exception as e:
        print(f"❌ API endpoints: FAILED - {e}")
        return False

def test_coordinate_conversions():
    """Test coordinate system conversions"""
    print("\n=== Testing Coordinate Conversions ===")
    try:
        from working_maze_server import generate_maze
        
        # Test maze to canvas coordinate conversion
        maze, solution_path = generate_maze()
        cell_size = 20
        
        # Expected canvas coordinates
        expected_canvas_start = [0 * cell_size + cell_size//2, 0 * cell_size + cell_size//2]
        expected_canvas_end = [19 * cell_size + cell_size//2, 19 * cell_size + cell_size//2]
        
        # Test calculations
        start_canvas_x = solution_path[0][0] * cell_size + cell_size//2
        start_canvas_y = solution_path[0][1] * cell_size + cell_size//2
        end_canvas_x = solution_path[1][0] * cell_size + cell_size//2  
        end_canvas_y = solution_path[1][1] * cell_size + cell_size//2
        
        calculated_start = [start_canvas_x, start_canvas_y]
        calculated_end = [end_canvas_x, end_canvas_y]
        
        assert calculated_start == expected_canvas_start, f"Start calculation wrong: {calculated_start} vs {expected_canvas_start}"
        assert calculated_end == expected_canvas_end, f"End calculation wrong: {calculated_end} vs {expected_canvas_end}"
        
        print("✅ Coordinate conversions: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Coordinate conversions: FAILED - {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Comprehensive Maze Captcha Tests")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_maze_generation,
        test_maze_generator_class,
        test_coordinate_conversions,
        test_api_endpoints
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests PASSED! Maze captcha system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests FAILED. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())