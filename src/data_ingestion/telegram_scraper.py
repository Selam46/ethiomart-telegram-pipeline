# from telethon import TelegramClient
# import os
# import pandas as pd
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# API_ID = os.getenv("TELEGRAM_API_ID")
# API_HASH = os.getenv("TELEGRAM_API_HASH")

# # Initialize the Telegram client
# client = TelegramClient('scraper', API_ID, API_HASH)

# async def scrape_channel(channel_name, save_path):
#     """Fetch messages from a Telegram channel and save them to a CSV file."""
#     async with client:
#         messages = []
#         async for message in client.iter_messages(channel_name, limit=100):  # Fetch up to 100 messages
#             messages.append({
#                 "text": message.text,
#                 "timestamp": message.date,
#                 "sender": message.sender_id,
#                 "media": message.media if message.media else None,
#             })
#         # Save messages to a CSV file
#         pd.DataFrame(messages).to_csv(save_path, index=False)
#         print(f"Data saved to {save_path}")



from telethon import TelegramClient
import os
import pandas as pd
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")

# Initialize Telegram client
client = TelegramClient("scraper", API_ID, API_HASH)

def is_amharic(text):
    """Check if text contains Amharic characters."""
    if not isinstance(text, str):
        return False
    amharic_pattern = re.compile(r"[\u1200-\u137F]+")
    return bool(amharic_pattern.search(text))

async def scrape_channel(channel_name, save_text_path, save_images_folder):
    """Fetch messages from a Telegram channel and save text and media."""
    async with client:
        messages = []
        os.makedirs(save_images_folder, exist_ok=True)  # Create folder for images

        async for message in client.iter_messages(channel_name, limit=100):  # Fetch 100 messages
            if message.text and is_amharic(message.text):
                messages.append({
                    "text": message.text,
                    "timestamp": message.date,
                    "sender": message.sender_id,
                })
            
            if message.media:  # If there's media, save it
                file_name = os.path.join(save_images_folder, f"{message.id}.jpg")
                await message.download_media(file_name)

        # Save messages to a CSV file
        pd.DataFrame(messages).to_csv(save_text_path, index=False, encoding="utf-8")
        print(f"Text data saved to {save_text_path}")
        print(f"Images saved to {save_images_folder}")
