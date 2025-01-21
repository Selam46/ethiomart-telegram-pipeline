import pandas as pd
import re

def tokenize(text):
    """Simple tokenizer for splitting text into words."""
    return text.split()

def label_tokens(tokens):
    """
    Manually label tokens with entities for a subset of the dataset.
    Example:
        Input: ["አዲስ", "አበባ", "ዋጋ", "1000", "ብር"]
        Output: [("አዲስ", "B-LOC"), ("አበባ", "I-LOC"), ("ዋጋ", "B-PRICE"), ("1000", "I-PRICE"), ("ብር", "I-PRICE")]
    """
    labels = []
    for token in tokens:
        if re.match(r"አዲስ|ቦሌ|አበባ", token):  # Example location keywords
            labels.append((token, "B-LOC" if len(labels) == 0 or labels[-1][1] != "B-LOC" else "I-LOC"))
        elif re.match(r"ዋጋ|ብር|በ", token):  # Example price keywords
            labels.append((token, "B-PRICE" if len(labels) == 0 or labels[-1][1] != "B-PRICE" else "I-PRICE"))
        elif re.match(r"ቤት|መጠጥ|ጫማ", token):  # Example product keywords
            labels.append((token, "B-Product" if len(labels) == 0 or labels[-1][1] != "B-Product" else "I-Product"))
        else:
            labels.append((token, "O"))
    return labels

def save_conll_format(data, output_path):
    """
    Save labeled data in CoNLL format.
    Format: Each token on its own line, followed by its label, with blank lines between messages.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for message in data:
            for token, label in message:
                f.write(f"{token} {label}\n")
            f.write("\n")  # Blank line to separate messages

if __name__ == "__main__":
    # Load the processed data from Task 1
    input_path = "data/processed/marakibrand_processed.csv"
    output_path = "data/processed/marakibrand_conll.txt"
    df = pd.read_csv(input_path)

    # Limit to 30–50 messages for labeling
    sample_data = df["processed_text"].dropna().head(50)

    # Label each message
    labeled_data = []
    for message in sample_data:
        tokens = tokenize(message)
        labeled_message = label_tokens(tokens)
        labeled_data.append(labeled_message)

    # Save the labeled data in CoNLL format
    save_conll_format(labeled_data, output_path)
    print(f"Labeled data saved in CoNLL format to {output_path}")
