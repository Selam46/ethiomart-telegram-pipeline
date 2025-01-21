# import re
# import pandas as pd
# import nltk
# nltk.download('punkt')

# # Tokenizer and Normalizer
# def tokenize(text):
#     """Tokenize Amharic text using NLTK."""
#     return nltk.word_tokenize(text)

# def normalize(text):
#     """Normalize Amharic text."""
#     replacements = {"ሀ": "ሃ", "ወጋበር": "ወጋቤር"}  # Add your replacements here
#     for old, new in replacements.items():
#         text = text.replace(old, new)
#     text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
#     return text

# def preprocess_text(text):
#     """Normalize and tokenize text."""
#     text = normalize(text)
#     tokens = tokenize(text)
#     return " ".join(tokens)  # Return as string for saving

# def preprocess_csv(input_path, output_path):
#     """Preprocess all messages in a CSV file and save the structured data."""
#     df = pd.read_csv(input_path)
#     df["processed_text"] = df["text"].apply(lambda x: preprocess_text(x) if isinstance(x, str) else "")
#     df[["sender", "timestamp", "processed_text"]].to_csv(output_path, index=False)
#     print(f"Preprocessed data saved to {output_path}")


import pandas as pd
import re

def normalize(text):
    """Normalize Amharic text."""
    replacements = {
        "ሀ": "ሃ", "ሐ": "ሃ", "ኅ": "ሃ", "ኸ": "ሀ",
        "ሰ": "ሠ", "ቃ": "ቀ", "ወጋበር": "ወጋቤር",
        "ኢትዮ": "ኢትዮጵያ", "ብርሆ": "ብርሃን",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

def preprocess_text(text):
    """Preprocess text by normalizing and keeping only Amharic tokens."""
    amharic_pattern = re.compile(r"[\u1200-\u137F]+")  # Match Amharic script
    tokens = amharic_pattern.findall(text)
    normalized_tokens = [normalize(token) for token in tokens]
    return " ".join(normalized_tokens)

def preprocess_csv(input_path, output_path):
    """Preprocess Amharic text data in a CSV file."""
    df = pd.read_csv(input_path)
    df["processed_text"] = df["text"].apply(lambda x: preprocess_text(x) if isinstance(x, str) else "")
    df[["sender", "timestamp", "processed_text"]].to_csv(output_path, index=False, encoding="utf-8")
    print(f"Preprocessed data saved to {output_path}")
