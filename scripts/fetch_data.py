import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.telegram_scraper import scrape_channel

if __name__ == "__main__":
    # Focus on @marakibrand only
    channel_name = "@marakibrand"
    save_path = "data/raw/marakibrand.csv"
    asyncio.run(scrape_channel(channel_name, save_path))
