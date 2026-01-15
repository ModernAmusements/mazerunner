#!/usr/bin/env python3
"""
Debug bot detection scores
"""
import sys
import sys
sys.path.append('.')

def debug_bot_detection():
    """Debug what bot scores are being generated"""
    print("=== Debugging Bot Detection Scores ===")
    try:
        import requests
        import json
        
        # Test case 1: Perfect straight line (should get high bot score)
        print("Test 1: Perfect straight line")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        perfect_path = []
        for i in range(15):  # 15 points
            x = 10 + i * (380/14)  # Perfect linear progression
            y = 10 + i * (380/14)  # Perfect linear progression
            perfect_path.append([int(x), int(y)])
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': perfect_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        print(f"Message: {result.get('message')}")
        print(f"Success: {result.get('success')}")
        print(f"Analysis: {result.get('analysis')}")
        
        # Test case 2: Very short path (should get high bot score)
        print("\nTest 2: Very short path")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        short_path = [[10, 10], [20, 20], [390, 390]]  # Only 3 points
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': short_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        print(f"Message: {result.get('message')}")
        print(f"Success: {result.get('success')}")
        print(f"Analysis: {result.get('analysis')}")
        
        # Test case 3: Wrong coordinates (should get high bot score)
        print("\nTest 3: Wrong coordinates")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        wrong_path = []
        for i in range(10):
            wrong_path.append([50 + i * 30, 50 + i * 30])  # Starts at 50, not 10
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': wrong_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        print(f"Message: {result.get('message')}")
        print(f"Success: {result.get('success')}")
        print(f"Analysis: {result.get('analysis')}")
        
    except Exception as e:
        print(f"Debug failed: {e}")

if __name__ == "__main__":
    debug_bot_detection()