import requests

from dotenv import load_dotenv
import os
load_dotenv()


# Replace with your bot token and chat ID
bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")    
chat_id = os.environ.get("TELEGRAM_CHAT_ID")
test_message = "Hello, test message from bot"

# Telegram Bot API URL
url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

# Payload to send the message
payload = {
    "chat_id": chat_id,
    "text": test_message
}

try:
    # Send the request to Telegram
    response = requests.post(url, json=payload)
    # Check the response
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print(f"Failed to send message. Error: {response.status_code}, {response.text}")
except Exception as e:
    print(f"An error occurred: {e}")
