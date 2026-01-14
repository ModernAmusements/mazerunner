import requests

# Test session persistence
session = requests.Session()

# Get captcha
print("Getting captcha...")
captcha_resp = session.get("http://127.0.0.1:8080/api/captcha?difficulty=easy")
captcha_data = captcha_resp.json()
captcha_id = captcha_data['captcha_id']
print(f"Captcha ID: {captcha_id}")

# Simulate bot
print("Simulating bot...")
bot_resp = session.post(
    "http://127.0.0.1:8080/api/bot-simulate",
    json={"captcha_id": captcha_id}
)

print("Bot simulation result:")
print(bot_resp.json())