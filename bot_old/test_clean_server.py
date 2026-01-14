import requests
import json

# Test the clean version
print("Testing clean production maze captcha...")

# Get captcha
try:
    captcha_resp = requests.get("http://127.0.0.1:8080/api/captcha?difficulty=easy")
    captcha_data = captcha_resp.json()
    captcha_id = captcha_data['captcha_id']
    print(f"Got captcha ID: {captcha_id}")
except Exception as e:
    print(f"Error getting captcha: {e}")
    exit(1)

# Test bot simulation
try:
    bot_resp = requests.post(
        "http://127.0.0.1:8080/api/bot-simulate",
        json={"captcha_id": captcha_id},
        cookies=captcha_resp.cookies
    )
    print("Bot simulation response:")
    print(json.dumps(bot_resp.json(), indent=2))
except Exception as e:
    print(f"Error with bot simulation: {e}")

# Check analytics
try:
    analytics_resp = requests.get("http://127.0.0.1:8080/api/analytics")
    analytics_data = analytics_resp.json()
    print("Analytics:")
    print(f"  Human detected: {analytics_data.get('human_detected', 0)}")
    print(f"  Bot detected: {analytics_data.get('bot_detected', 0)}")
    print(f"  Total attempts: {analytics_data.get('total_attempts', 0)}")
    print(f"  Learned patterns: {analytics_data.get('learning_status', {}).get('behaviors_learned', 0)}")
except Exception as e:
    print(f"Error getting analytics: {e}")