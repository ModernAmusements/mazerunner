#!/usr/bin/env python3
"""
Test script to verify maze generation
"""
import requests
import json

def test_maze():
    """Test maze generation"""
    try:
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        data = response.json()
        
        print("=== MAZE GENERATION TEST ===")
        print(f"Captcha ID: {data['captcha_id']}")
        print(f"Start coordinates (maze): {data['start']}")
        print(f"End coordinates (maze): {data['end']}")
        print(f"Canvas start: {data['canvas_start']}")
        print(f"Canvas end: {data['canvas_end']}")
        
        # Verify coordinates are correct for 20x20 maze
        expected_start = [0, 0]
        expected_end = [19, 19]
        
        print(f"\nExpected start: {expected_start}")
        print(f"Expected end: {expected_end}")
        print(f"Maze size: 20x20")
        
        # Check if coordinates match expectations
        start_match = data['start'] == expected_start
        end_match = data['end'] == expected_end
        
        print(f"\nStart match: {start_match}")
        print(f"End match: {end_match}")
        
        if start_match and end_match:
            print("✅ MAZE GENERATION: SUCCESS - 20x20 maze with correct start/end positions")
        else:
            print("❌ MAZE GENERATION: FAILED - incorrect coordinates")
            
        return start_match and end_match
        
    except Exception as e:
        print(f"Error testing maze: {e}")
        return False

if __name__ == "__main__":
    test_maze()