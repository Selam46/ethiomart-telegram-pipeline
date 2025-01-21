
from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from evaluate import load
import numpy as np

# Task: Compare models for NER
models_to_compare = {
    "XLM-Roberta": "xlm-roberta-base",
    "DistilBERT": "distilbert-base-multilingual-cased",
    "mBERT": "bert-base-multilingual-cased"
}


cleaned_dataset = DatasetDict({
    "train": Dataset.from_dict({"tokens": [["Hello", "world"]], "ner_tags": [["O", "B-LOC"]]}),
    "validation": Dataset.from_dict({"tokens": [["Hello", "Ethiopia"]], "ner_tags": [["O", "B-LOC"]]}),
    "test": Dataset.from_dict({"tokens": [["Welcome", "to", "Africa"]], "ner_tags": [["O", "O", "B-LOC"]]}),
})

# Extract unique labels dynamically from the dataset
all_labels = set(
    label for split in cleaned_dataset for example in cleaned_dataset[split] for label in example["ner_tags"]
)
label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels))}
id_to_label = {idx: label for label, idx in label_to_id.items()}

print(f"label_to_id: {label_to_id}")

# Function to map labels to IDs
def map_labels(example):
    example["ner_tags"] = [label_to_id[label] for label in example["ner_tags"]]
    return example

# Apply label mapping to all splits
for split in ["train", "validation", "test"]:
    cleaned_dataset[split] = cleaned_dataset[split].map(map_labels)

# Initialize metric
metric = load("seqeval")

# Function to compute metrics
def compute_metrics(predictions, labels):
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

# Fine-tune and evaluate each model
results = {}
for model_name, model_checkpoint in models_to_compare.items():
    print(f"Fine-tuning {model_name}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(label_to_id)
    )

    # Tokenize dataset
    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
        )
        labels = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(example["ner_tags"][word_idx])
            else:
                labels.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = cleaned_dataset.map(tokenize_and_align_labels, batched=True)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs/{model_name}",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p.predictions, p.label_ids),
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    # Store results
    results[model_name] = eval_results
    print(f"{model_name} Results: {eval_results}")

# Compare Results
print("\nFinal Comparison:")
for model_name, result in results.items():
    print(f"{model_name}:")
    print(f"  Precision: {result['eval_precision']:.4f}")
    print(f"  Recall: {result['eval_recall']:.4f}")
    print(f"  F1 Score: {result['eval_f1']:.4f}")
    print(f"  Accuracy: {result['eval_accuracy']:.4f}")
