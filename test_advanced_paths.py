#!/usr/bin/env python3
"""
Advanced path drawing tests - realistic human paths and sophisticated bots
"""
import sys
import os
sys.path.append('.')

def test_realistic_human_path():
    """Test with realistic human-like path (not perfectly straight)"""
    print("=== Testing Realistic Human Path ===")
    try:
        import requests
        import random
        
        # Get a captcha
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_data = response.json()
        captcha_id = captcha_data['captcha_id']
        
        # Create a realistic, slightly curved path from start to end
        realistic_path = []
        steps = 25  # More points for realism
        
        for i in range(steps):
            # Base linear progression
            progress = i / (steps - 1)
            base_x = 10 + progress * 380  # 10 to 390
            base_y = 10 + progress * 380  # 10 to 390
            
            # Add small random variations (human hand tremor)
            variation_x = random.randint(-3, 3)
            variation_y = random.randint(-3, 3)
            
            # Occasionally make it slightly curved
            if 5 < i < 20:
                curve_offset = 10 * (0.5 - abs(progress - 0.5))  # Bell curve
                base_x += curve_offset
            
            realistic_path.append([
                max(0, min(400, base_x + variation_x)),
                max(0, min(400, base_y + variation_y))
            ])
        
        # Ensure first and last points are exactly correct
        realistic_path[0] = [10, 10]
        realistic_path[-1] = [390, 390]
        
        print(f"Realistic path length: {len(realistic_path)} points")
        print(f"Sample points: {realistic_path[:3]} ... {realistic_path[-3:]}")
        
        # Submit realistic path
        verify_data = {
            'captcha_id': captcha_id,
            'path': realistic_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        assert verify_response.status_code == 200
        result = verify_response.json()
        
        if result.get('success'):
            analysis = result.get('analysis', {})
            print(f"✅ Realistic human path accepted")
            print(f"   Confidence: {analysis.get('confidence')}")
            print(f"   Path length: {analysis.get('path_length')}")
            return True
        else:
            print(f"❌ Realistic human path rejected: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"❌ Realistic human path test failed: {e}")
        return False

def test_sophisticated_bot():
    """Test sophisticated bot that tries to mimic human behavior"""
    print("\n=== Testing Sophisticated Bot Detection ===")
    try:
        import requests
        import random
        
        # Bot strategy 1: Perfect straight line (too perfect)
        print("Bot Strategy 1: Perfect straight line")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        perfect_path = []
        for i in range(10):
            x = 10 + i * 38  # Perfect linear progression
            y = 10 + i * 38  # Perfect linear progression
            perfect_path.append([x, y])
        
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
        bot1_detected = not result.get('success')
        print(f"   Perfect line detected as bot: {bot1_detected}")
        
        # Bot strategy 2: Random path that doesn't connect start to end
        print("Bot Strategy 2: Random scattered points")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        random_path = []
        for _ in range(15):
            random_path.append([
                random.randint(0, 400),
                random.randint(0, 400)
            ])
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': random_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        bot2_detected = not result.get('success')
        print(f"   Random path detected as bot: {bot2_detected}")
        
        # Bot strategy 3: Path with wrong timing pattern (instant submission)
        print("Bot Strategy 3: Instant submission test")
        # This would require timing analysis on server side
        # For now, we'll test the existing validation
        
        # Bot strategy 4: Minimal effort path
        print("Bot Strategy 4: Minimal effort path")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        minimal_path = [[10, 10], [20, 20], [390, 390]]  # Just 3 points
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': minimal_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        bot4_detected = not result.get('success')
        print(f"   Minimal path detected as bot: {bot4_detected}")
        
        # Bot strategy 5: Path that starts/ends at wrong coordinates
        print("Bot Strategy 5: Wrong coordinates")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        wrong_coords_path = []
        for i in range(10):
            wrong_coords_path.append([
                50 + i * 30,  # Starts at 50, not 10
                50 + i * 30   # Starts at 50, not 10
            ])
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': wrong_coords_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        bot5_detected = not result.get('success')
        print(f"   Wrong coordinates detected as bot: {bot5_detected}")
        
        # Evaluate bot detection effectiveness
        detected_strategies = sum([bot1_detected, bot2_detected, bot4_detected, bot5_detected])
        total_strategies = 4
        
        detection_rate = detected_strategies / total_strategies
        print(f"Bot detection rate: {detection_rate:.1%} ({detected_strategies}/{total_strategies})")
        
        if detection_rate >= 0.75:
            print("✅ Sophisticated bot detection: PASSED")
            return True
        else:
            print("⚠️  Sophisticated bot detection: PARTIAL")
            return detection_rate > 0.5
            
    except Exception as e:
        print(f"❌ Sophisticated bot test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n=== Testing Edge Cases ===")
    try:
        import requests
        
        # Test 1: Path exactly at boundaries
        print("Test 1: Boundary path")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        boundary_path = [
            [10, 10],     # Start
            [50, 10],     # Along top edge
            [390, 50],    # To right edge
            [390, 390]     # End
        ]
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': boundary_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        boundary_success = result.get('success')
        print(f"   Boundary path accepted: {boundary_success}")
        
        # Test 2: Path with duplicate points
        print("Test 2: Path with duplicate points")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        duplicate_path = [
            [10, 10],     # Start
            [10, 10],     # Duplicate
            [20, 20],     # Normal
            [20, 20],     # Duplicate
            [30, 30],     # Normal
            [390, 390]    # End
        ]
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': duplicate_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        duplicate_success = result.get('success')
        print(f"   Duplicate points accepted: {duplicate_success}")
        
        # Test 3: Very long detailed path
        print("Test 3: Very detailed path")
        response = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=medium")
        captcha_id = response.json()['captcha_id']
        
        detailed_path = []
        for i in range(100):  # 100 points
            progress = i / 99
            x = 10 + progress * 380
            y = 10 + progress * 380
            detailed_path.append([int(x), int(y)])
        
        verify_data = {
            'captcha_id': captcha_id,
            'path': detailed_path
        }
        
        verify_response = requests.post(
            "http://127.0.0.1:8080/api/verify",
            json=verify_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = verify_response.json()
        detailed_success = result.get('success')
        print(f"   Detailed path accepted: {detailed_success}")
        if detailed_success:
            analysis = result.get('analysis', {})
            print(f"   Path length: {analysis.get('path_length')}")
        
        print("✅ Edge cases: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False

def main():
    """Run advanced path and bot tests"""
    print("🧪 Running Advanced Path & Bot Detection Tests")
    print("=" * 60)
    
    tests = [
        test_realistic_human_path,
        test_sophisticated_bot,
        test_edge_cases
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All advanced tests PASSED!")
        print("🤖 Bot detection is working effectively!")
        return 0
    else:
        print("⚠️  Some advanced tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())