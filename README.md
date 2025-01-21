# EthioMart Telegram Pipeline

This project builds an **Amharic Named Entity Recognition (NER) pipeline** to extract business-critical entities such as product names, prices, and locations from Ethiopian Telegram e-commerce channels. The pipeline handles data ingestion, preprocessing, model fine-tuning, comparison, and interpretability.

## Folder Structure
```plaintext
ethiomart-telegram-pipeline/
├── data/
│   ├── raw/                     # Raw data from Telegram channels
│   ├── processed/               # Preprocessed and labeled data
├── scripts/
│   ├── fetch_data.py            # Ingest Telegram data
│   ├── preprocess_data.py       # Preprocess raw data
│   ├── label_data.py            # Label data in CoNLL format
│   ├── fine_tune_models.py      # Fine-tune multiple models
│   ├── compare_models.py        # Compare model performances
│   ├── interpret_model.py       # Interpret model predictions
├── src/
│   ├── data_ingestion/          # Telegram scraping logic
│   ├── preprocessing/           # Text preprocessing
│   ├── evaluation/              # Metrics and interpretability helpers
├── results/                     # Fine-tuned models and evaluation outputs
├── logs/                        # Training logs
├── requirements.txt             # Python dependencies
├── README.md                    # Project description
```

## Tasks Overview

### Task 1: Data Ingestion and Preprocessing
- **Goal**: Collect messages from Telegram e-commerce channels and preprocess them for NER.
- **Steps**:
  1. Scrape messages, metadata, and media using `fetch_data.py`.
  2. Preprocess text: normalize Amharic text, tokenize, and clean data using `preprocess_data.py`.
- **Outputs**:
  - Raw data (`data/raw/`)
  - Preprocessed data (`data/processed/`)

### Task 2: Label Data in CoNLL Format
- **Goal**: Annotate a subset of the dataset with entity labels (e.g., `B-LOC`, `B-PRICE`).
- **Steps**:
  1. Use `label_data.py` to tokenize and manually label at least 30–50 messages.
  2. Save in CoNLL format (`data/processed/marakibrand_conll.txt`).
- **Outputs**:
  - Annotated dataset in CoNLL format.

### Task 3: Fine-Tune NER Model
- **Goal**: Fine-tune a pre-trained model to extract entities.
- **Steps**:
  1. Fine-tune models like `xlm-roberta-base` or `distilbert-base-multilingual-cased` using `fine_tune_models.py`.
  2. Save fine-tuned models (`results/<model_name>/`).
- **Outputs**:
  - Fine-tuned models.
  - Evaluation metrics (precision, recall, F1-score).

### Task 4: Model Comparison & Selection
- **Goal**: Compare different models for performance, speed, and robustness.
- **Steps**:
  1. Evaluate models using `compare_models.py`.
  2. Save metrics for each model in `results/comparison_results.json`.
- **Outputs**:
  - Comparison table (accuracy, F1-score).
  - Selected best model for production.

### Task 5: Model Interpretability
- **Goal**: Explain how the model identifies entities using SHAP and LIME.
- **Steps**:
  1. Use SHAP and LIME in `interpret_model.py` to analyze predictions.
  2. Identify misclassified or ambiguous cases.
  3. Generate visual reports.
- **Outputs**:
  - SHAP and LIME visualizations.
  - Actionable insights for model improvement.

## How to Run

### 1. Set Up Environment
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Ingestion and Preprocessing
Run the data ingestion pipeline:
```bash
python scripts/fetch_data.py
```
Preprocess the raw data:
```bash
python scripts/preprocess_data.py
```

### 3. Label Data
Annotate a subset of the data:
```bash
python scripts/label_data.py
```

### 4. Fine-Tune Models
Fine-tune NER models:
```bash
python scripts/fine_tune_models.py
```

### 5. Compare Models
Compare model performances:
```bash
python scripts/compare_models.py
```

### 6. Interpret Models
Generate SHAP and LIME visualizations:
```bash
python scripts/interpret_model.py
```


---


