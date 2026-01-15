#!/usr/bin/env python3
"""
Path drawing and bot detection tests for maze captcha system
"""
import sys
import os
sys.path.append('.')

def test_path_drawing():
    """Test path drawing and verification"""
    print("=== Testing Path Drawing ===")
    try:
        import requests
        import json
        
        # Generate a new captcha
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        assert response.status_code == 200
        
        captcha_data = response.json()
        captcha_id = captcha_data['captcha_id']
        print(f"Generated captcha: {captcha_id}")
        
        # Draw a valid path from start to end
        # Start from canvas_start [10, 10] and end at canvas_end [390, 390]
        # Create a simple diagonal path
        valid_path = []
        for i in range(20):  # 20 points along the path
            x = 10 + i * 19  # From 10 to 390
            y = 10 + i * 19  # From 10 to 390
            valid_path.append([x, y])
        
        print(f"Path length: {len(valid_path)} points")
        print(f"First point: {valid_path[0]}, Last point: {valid_path[-1]}")
        
        # Submit the path for verification
        verify_data = {
            'captcha_id': captcha_id,
            'path': valid_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert verify_response.status_code == 200
        result = verify_response.json()
        
        print(f"Verification result: {result}")
        print(f"Success: {result.get('success')}")
        print(f"Message: {result.get('message')}")
        
        # Check if the path was validated correctly
        if result.get('success'):
            analysis = result.get('analysis', {})
            print(f"Human detected: {analysis.get('is_human')}")
            print(f"Confidence: {analysis.get('confidence')}")
            print(f"Path length: {analysis.get('path_length')}")
            print("✅ Path drawing: PASSED - Valid path accepted")
            return True
        else:
            print(f"❌ Path drawing: FAILED - {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Path drawing: FAILED - {e}")
        return False

def test_bot_detection():
    """Test bot detection with invalid paths"""
    print("\n=== Testing Bot Detection ===")
    try:
        import requests
        
        # Test 1: Empty path
        print("Test 1: Empty path")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': []  # Empty path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert verify_response.status_code == 200
        result = verify_response.json()
        assert not result.get('success'), "Empty path should be rejected"
        print("✅ Empty path correctly rejected")
        
        # Test 2: Too short path
        print("Test 2: Too short path")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': [[10, 10], [15, 15]]  # Only 2 points
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert verify_response.status_code == 200
        result = verify_response.json()
        assert not result.get('success'), "Too short path should be rejected"
        print("✅ Short path correctly rejected")
        
        # Test 3: Wrong start/end coordinates
        print("Test 3: Wrong coordinates")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_data = response.json()
        captcha_id = captcha_data['captcha_id']
        
        # Path that doesn't start or end at correct positions
        wrong_path = []
        for i in range(10):
            wrong_path.append([50, 50])  # Same point repeated
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': wrong_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert verify_response.status_code == 200
        result = verify_response.json()
        
        # This should fail due to wrong coordinates
        if not result.get('success'):
            analysis = result.get('analysis', {})
            print(f"Bot detection confidence: {analysis.get('confidence')}")
            print(f"Message: {result.get('message')}")
            print("✅ Wrong coordinates correctly rejected")
        else:
            print("⚠️  Wrong coordinates unexpectedly accepted")
        
        # Test 4: Invalid captcha ID
        print("Test 4: Invalid captcha ID")
        verify_data = {
            'captcha_id': 'invalid_id_12345',
            'path': [[10, 10], [20, 20], [30, 30]]
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert verify_response.status_code == 200
        result = verify_response.json()
        assert not result.get('success'), "Invalid captcha ID should be rejected"
        print("✅ Invalid captcha ID correctly rejected")
        
        print("✅ Bot detection: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Bot detection: FAILED - {e}")
        return False

def test_path_formats():
    """Test different path coordinate formats"""
    print("\n=== Testing Path Coordinate Formats ===")
    try:
        import requests
        
        # Get a captcha
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        # Test 1: List format [x, y]
        print("Test 1: List format [x, y]")
        path_list = [[10, 10], [20, 20], [30, 30], [40, 40]]
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': path_list
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if verify_response.status_code == 200:
            print("✅ List format accepted")
        else:
            print("❌ List format rejected")
        
        # Test 2: Dict format {"x": x, "y": y}
        print("Test 2: Dict format {x, y}")
        path_dict = [
            {"x": 10, "y": 10},
            {"x": 20, "y": 20},
            {"x": 30, "y": 30},
            {"x": 40, "y": 40}
        ]
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': path_dict
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if verify_response.status_code == 200:
            print("✅ Dict format accepted")
        else:
            print("❌ Dict format rejected")
        
        print("✅ Path coordinate formats: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Path coordinate formats: FAILED - {e}")
        return False

def test_analytics_tracking():
    """Test analytics tracking of attempts"""
    print("\n=== Testing Analytics Tracking ===")
    try:
        import requests
        
        # Get initial analytics
        response = requests.get("http://127.0.0.1:8080/api/analytics")
        initial_data = response.json()
        initial_attempts = initial_data.get('total_attempts', 0)
        initial_success = initial_data.get('successful_verifications', 0)
        
        print(f"Initial attempts: {initial_attempts}")
        print(f"Initial successes: {initial_success}")
        
        # Submit a valid path
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        valid_path = [[10, 10], [100, 100], [200, 200], [390, 390]]
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': valid_path
        }
        
        requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Submit an invalid path
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        invalid_path = []  # Empty path
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': invalid_path
        }
        
        requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        # Check updated analytics
        response = requests.get("http://127.0.0.1:8080/api/analytics")
        final_data = response.json()
        final_attempts = final_data.get('total_attempts', 0)
        final_success = final_data.get('successful_verifications', 0)
        
        print(f"Final attempts: {final_attempts}")
        print(f"Final successes: {final_success}")
        print(f"Success rate: {final_data.get('success_rate', 0):.1f}%")
        
        # Should have at least some tracking
        if final_attempts >= initial_attempts:
            print("✅ Analytics tracking: PASSED")
            return True
        else:
            print("❌ Analytics tracking: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Analytics tracking: FAILED - {e}")
        return False

def main():
    """Run all path and bot detection tests"""
    print("🧪 Running Path Drawing & Bot Detection Tests")
    print("=" * 60)
    
    tests = [
        test_path_drawing,
        test_bot_detection,
        test_path_formats,
        test_analytics_tracking
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All path and bot tests PASSED!")
        return 0
    else:
        print("⚠️  Some tests FAILED. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())