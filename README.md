# EthioMart Data Ingestion and Preprocessing

## Overview
EthioMart aims to centralize Telegram-based e-commerce activities in Ethiopia by consolidating data from multiple Telegram channels. This will simplify product discovery and vendor interactions for customers while creating a seamless platform. The project involves fine-tuning a large language model (LLM) for Amharic Named Entity Recognition (NER) to extract critical business information such as product names, prices, and locations.

## Key Objectives
- Real-time data collection from Telegram e-commerce channels.
- Fine-tune LLM for extracting entities like product names, prices, and locations.
- Populate a centralized database for a unified e-commerce experience.

## Task 1: Data Ingestion and Preprocessing
### Steps
1. **Identify and Connect to Channels**:
   - Select at least 5 Ethiopian-based Telegram channels.
   - Use a custom scraper to fetch data.
2. **Message Ingestion**:
   - Collect text, images, and documents in real-time.
3. **Preprocess Text Data**:
   - Tokenize and normalize text.
   - Handle Amharic linguistic features.
4. **Clean and Structure Data**:
   - Separate metadata from content.
   - Store data in a unified format (e.g., JSON or CSV).

### Output
- Preprocessed data ready for NER.
- A reusable pipeline for continuous ingestion and preprocessing.

## Repository Structure
```
project/
├── README.md               # Documentation
├── requirements.txt        # Python dependencies
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code for scraping and preprocessing
├── scripts/                # Automation scripts
└── tests/                  # Unit tests
```

## Usage
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Data Ingestion**:
   ```bash
   python scripts/fetch_data.py
   ```
3. **Run Preprocessing**:
   ```bash
   python scripts/preprocess_data.py
   ```

## License
This project is licensed under the MIT License.

## Contact
For support, contact **support@ethiomart.com**.
