import requests

# Test session persistence
session = requests.Session()

# Get captcha
print("Getting captcha...")
captcha_resp = session.get("http://127.0.0.1:8080/api/captcha?difficulty=easy")
print(f"Status: {captcha_resp.status_code}")
print(f"Headers: {dict(captcha_resp.headers)}")
print(f"Raw response (first 200 chars): {captcha_resp.text[:200]}")

try:
    captcha_data = captcha_resp.json()
    captcha_id = captcha_data['captcha_id']
    print(f"Captcha ID: {captcha_id}")
except Exception as e:
    print(f"JSON parse error: {e}")
    exit(1)

# Simulate bot
print("Simulating bot...")
bot_resp = session.post(
    "http://127.0.0.1:8080/api/bot-simulate",
    json={"captcha_id": captcha_id}
)

print(f"Bot simulation status: {bot_resp.status_code}")
print(f"Bot simulation raw response: {bot_resp.text}")