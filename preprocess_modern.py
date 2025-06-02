# preprocess.py
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from datasets import DatasetDict, Dataset
import os
import json

def preprocess_meld_data_for_kfold(
    train_csv_path: str,
    dev_csv_path: str, # This will be merged into train
    test_csv_path: str,
    model_name: str = "answerdotai/ModernBERT-large",
    output_dir: str = "./processed_data_kfold" # Changed output directory name for clarity
):
    """
    Preprocesses the MELD dataset (from local CSVs) for emotion classification,
    merging train and dev sets for k-fold cross-validation.

    Args:
        train_csv_path (str): Path to the MELD train CSV file.
        dev_csv_path (str): Path to the MELD dev (validation) CSV file.
        test_csv_path (str): Path to the MELD test CSV file.
        model_name (str): The Hugging Face model name for the tokenizer.
        output_dir (str): Directory to save the processed datasets and label mappings.
    """
    print(f"Loading data from local CSVs:")
    print(f"  Train: {train_csv_path}")
    print(f"  Dev (will be merged into train): {dev_csv_path}")
    print(f"  Test: {test_csv_path}")

    # Load datasets from local CSV files
    try:
        train_df = pd.read_csv(train_csv_path)
        dev_df = pd.read_csv(dev_csv_path)
        test_df = pd.read_csv(test_csv_path)
    except FileNotFoundError as e:
        print(f"Error: One or more MELD CSV files not found. Please ensure the paths are correct.")
        print(f"Error details: {e}")
        return

    # Clean the data function, adjusted for 'Utterance' and 'Emotion' columns
    def clean_dataframe(df):
        # Drop rows where 'Utterance' or 'Emotion' is missing
        df = df.dropna(subset=['Utterance', 'Emotion'])
        # Select only the relevant columns
        df = df[['Utterance', 'Emotion']]
        # Strip whitespace and replace specific newlines in 'Utterance'
        df['Utterance'] = df['Utterance'].str.strip().str.replace(r'\r\n', ' ', regex=True)
        # Rename 'Emotion' to 'label' for Hugging Face Trainer compatibility
        df['label'] = df['Emotion']
        return df

    print("Cleaning and preparing dataframes...")
    train_data_cleaned = clean_dataframe(train_df)
    dev_data_cleaned = clean_dataframe(dev_df)
    test_data_cleaned = clean_dataframe(test_df)

    # MERGE TRAIN AND DEV DATASETS FOR K-FOLD CROSS-VALIDATION
    print("Merging train and dev datasets for k-fold cross-validation...")
    full_train_data = pd.concat([train_data_cleaned, dev_data_cleaned], ignore_index=True)

    # Get unique labels from the merged training data to ensure consistency
    label_list = sorted(full_train_data['label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Discovered emotions: {label_list}")
    print(f"Label to ID mapping: {label2id}")

    # Initialize tokenizer for ModernBERT-large
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to tokenize utterances and map labels
    def tokenize_function(examples):
        # Tokenize the 'Utterance' column
        tokenized_inputs = tokenizer(
            examples["Utterance"],
            padding="max_length", # Pad to max_length
            truncation=True,      # Truncate if longer than max_length
            max_length=tokenizer.model_max_length # Use model's max length if available, default 512
        )
        # Map string labels to numerical IDs
        tokenized_inputs["labels"] = [label2id[e] for e in examples["label"]]
        return tokenized_inputs

    print("Converting pandas DataFrames to Hugging Face Datasets...")
    # Convert DataFrames to Hugging Face Datasets, now with merged train set
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(full_train_data),
        "test": Dataset.from_pandas(test_data_cleaned)
    })

    print("Tokenizing and mapping labels for the dataset splits...")
    # Apply tokenization and label mapping to all splits
    # CORRECTED: Removed '__index_level_0__' from remove_columns
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=['Utterance', 'Emotion', 'label']
    )

    # Save the processed datasets and label mappings
    os.makedirs(output_dir, exist_ok=True)
    for split in tokenized_datasets.keys():
        save_path = os.path.join(output_dir, split)
        tokenized_datasets[split].save_to_disk(save_path)
        print(f"Saved {split} dataset to {save_path}")

    # Save label mappings for use in train.py and inference
    with open(os.path.join(output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)
    with open(os.path.join(output_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f)
    print(f"Saved label mappings to {output_dir}")

if __name__ == "__main__":
    # Define paths to your local MELD CSV files
    base_data_path = "data/MELD" # Relative to where you run the script

    preprocess_meld_data_for_kfold(
        train_csv_path=os.path.join(base_data_path, "train_sent_emo.csv"),
        dev_csv_path=os.path.join(base_data_path, "dev_sent_emo.csv"),
        test_csv_path=os.path.join(base_data_path, "test_sent_emo.csv"),
        model_name="answerdotai/ModernBERT-large",
        output_dir="./processed_data_kfold" # Output directory for k-fold ready data
    )