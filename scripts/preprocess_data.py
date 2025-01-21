import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.text_preprocessor import preprocess_csv

if __name__ == "__main__":
    input_path = "data/raw/marakibrand.csv"
    output_path = "data/processed/marakibrand_processed.csv"
    preprocess_csv(input_path, output_path)
