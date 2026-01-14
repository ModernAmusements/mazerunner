import requests
import json

# Get new captcha
response = requests.get("http://localhost:8080/api/captcha?difficulty=easy")
captcha_data = response.json()
captcha_id = captcha_data['captcha_id']
print(f"New captcha ID: {captcha_id}")

# Test bot simulation
bot_response = requests.post(
    "http://localhost:8080/api/bot-simulate",
    json={"captcha_id": captcha_id},
    cookies=response.cookies
)
print("Bot simulation result:")
print(json.dumps(bot_response.json(), indent=2))