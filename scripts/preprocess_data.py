from src.preprocessing.text_preprocessor import preprocess_csv

if __name__ == "__main__":
    input_path = "data/raw/marakibrand.csv"
    output_path = "data/processed/marakibrand_processed.csv"
    preprocess_csv(input_path, output_path)
