
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.telegram_scraper import scrape_channel

if __name__ == "__main__":
    # Scrape only from @marakibrand
    channel_name = "@marakibrand"
    save_text_path = "data/raw/marakibrand.csv"
    save_images_folder = "data/raw/marakibrand_images"

    asyncio.run(scrape_channel(channel_name, save_text_path, save_images_folder))
