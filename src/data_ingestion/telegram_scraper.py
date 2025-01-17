from telethon import TelegramClient
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")

# Initialize the Telegram client
client = TelegramClient('scraper', API_ID, API_HASH)

async def scrape_channel(channel_name, save_path):
    """Fetch messages from a Telegram channel and save them to a CSV file."""
    async with client:
        messages = []
        async for message in client.iter_messages(channel_name, limit=100):  # Fetch up to 100 messages
            messages.append({
                "text": message.text,
                "timestamp": message.date,
                "sender": message.sender_id,
                "media": message.media if message.media else None,
            })
        # Save messages to a CSV file
        pd.DataFrame(messages).to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
