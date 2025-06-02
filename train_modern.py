# train.py
import argparse
import json
import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import EarlyStoppingCallback

def compute_metrics(p):
    """
    Computes evaluation metrics: F1-macro, accuracy, precision-macro, and recall-macro.
    """
    predictions = p.predictions.argmax(axis=1)
    labels = p.label_ids
    f1_macro = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    return {
        "f1": f1_macro,
        "accuracy": accuracy,
        "precision": precision_macro,
        "recall": recall_macro
    }

def train_modernbert_kfold(
    model_name: str = "answerdotai/ModernBERT-large",
    processed_data_dir: str = "./modernBERT/processed_data_kfold", # Updated to k-fold output
    output_dir: str = "./modernBERT_model/model_output_kfold",       # Updated output directory
    num_epochs: int = 3,
    batch_size: int = 8,  # Adjusted batch size for larger model and quantization
    learning_rate: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    early_stopping_patience: int = 3,
):
    """
    Trains ModernBERT-large for emotion classification using PEFT (LoRA)
    and 4-bit quantization, ready for k-fold cross-validation.
    """
    print(f"Loading processed data from {processed_data_dir}...")
    try:
        # Load the merged train dataset and the test dataset
        train_dataset = load_from_disk(os.path.join(processed_data_dir, "train"))
        test_dataset = load_from_disk(os.path.join(processed_data_dir, "test"))
    except Exception as e:
        print(f"Error loading processed datasets. Make sure '{processed_data_dir}' contains 'train' and 'test' directories.")
        print(f"Did you run preprocess.py (for k-fold) first? Error: {e}")
        return

    # Load label mappings
    try:
        with open(os.path.join(processed_data_dir, "label2id.json"), "r") as f:
            label2id = json.load(f)
        with open(os.path.join(processed_data_dir, "id2label.json"), "r") as f:
            id2label = json.load(f)
    except FileNotFoundError:
        print(f"Label mappings not found in {processed_data_dir}. Ensure preprocess.py created them.")
        return

    num_labels = len(label2id)
    print(f"Number of emotion labels: {num_labels}")

    print("Configuring 4-bit quantization...")
    # BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    print(f"Loading pre-trained model: {model_name} with {num_labels} classification heads and quantization...")
    # Load ModernBERT-large with a classification head and quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, # Use bfloat16 for computation if supported by GPU
    )

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"], # Common target modules for BERT-like models
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS, # Specify sequence classification task
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, # For evaluation on test set after training
        learning_rate=learning_rate,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=500,
        evaluation_strategy="no", # Evaluation will be handled by k-fold loop
        save_strategy="no",       # Saving will be handled manually or by k-fold loop
        report_to="none", # You can set this to "tensorboard" or "wandb" for logging
        fp16=True if torch.cuda.is_available() else False, # Enable mixed precision if CUDA is available
        load_best_model_at_end=False, # Best model selection will be handled by k-fold logic
        # disable_tqdm=True, # Uncomment to disable tqdm progress bar
    )

    print("Initializing Trainer for the full training set (before k-fold logic)...")
    # Note: This train.py will train on the *entire* merged dataset.
    # The k-fold cross-validation logic will need to be implemented
    # by you in a separate script or by extending this one to
    # iterate through folds and create train/validation splits dynamically.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # This is now the merged train+dev dataset
        eval_dataset=test_dataset,   # This will be used for final evaluation after k-fold
        compute_metrics=compute_metrics,
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    print("Starting training on the full merged dataset...")
    # Perform training on the full merged dataset.
    # If you intend to use k-fold *during* this script, you'll need
    # to add a loop here that manually creates folds and trains models for each.
    train_results = trainer.train()

    print("Training finished. Evaluating the final model on the test set...")
    # Evaluate on the dedicated test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Evaluation Results: {test_results}")

    print(f"Saving the final trained model to {output_dir}/final_model")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model")) # Save tokenizer with the model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ModernBERT-large for emotion classification using PEFT and 4-bit quantization.")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-large",
                        help="Hugging Face model name or path of the pre-trained ModernBERT model.")
    parser.add_argument("--processed_data_dir", type=str, default="./processed_data_kfold",
                        help="Directory where the preprocessed dataset (for k-fold) is saved.")
    parser.add_argument("--output_dir", type=str, default="./model_output_kfold",
                        help="Directory to save the trained model checkpoints and results.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, # Adjusted default
                        help="Training batch size per device. Adjust based on GPU memory.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA r parameter (rank).")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate.")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of evaluation steps to wait before early stopping.")
    args = parser.parse_args()

    train_modernbert_kfold(
        model_name=args.model_name,
        processed_data_dir=args.processed_data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        early_stopping_patience=args.early_stopping_patience,
    )