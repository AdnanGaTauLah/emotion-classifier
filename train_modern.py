# train_modern_single_run_quantized.py
import argparse
import json
import os
import numpy as np
import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# Removed: from sklearn.model_selection import KFold # No longer needed for single run

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

def train_modernbert_single_run_quantized(
    model_name: str = "answerdotai/ModernBERT-large",
    processed_data_dir: str = "./processed_data_kfold", # Data from preprocess_kfold
    output_dir: str = "./model_output_single_run_quantized", # New output directory
    num_epochs: int = 1,  # Set to 1 epoch as requested
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    seed: int = 42 # Still useful for reproducibility of weights/shuffling
):
    """
    Trains ModernBERT-large for emotion classification in a single run
    with LoRA and 4-bit quantization (no k-fold cross-validation).
    """
    print(f"Loading processed data from {processed_data_dir}...")
    try:
        # full_train_dataset is now the main training set
        train_dataset = load_from_disk(os.path.join(processed_data_dir, "train"))
        eval_dataset = load_from_disk(os.path.join(processed_data_dir, "test")) # Use test as evaluation during training
        test_dataset = load_from_disk(os.path.join(processed_data_dir, "test")) # Separate for final eval
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

    # Configure 4-bit quantization
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value", "key", "dense"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    # Initialize model
    print(f"Loading ModernBERT-large model: {model_name} with quantization and LoRA...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16, # Using float16 for compatibility
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Set up training arguments for a single run
    os.makedirs(output_dir, exist_ok=True) # Ensure main output directory exists

    training_args = TrainingArguments(
        output_dir=output_dir, # Output directly to the main directory
        num_train_epochs=num_epochs, # Set to 1 epoch
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=500,
        eval_strategy="epoch", # Evaluate at the end of each epoch on eval_dataset
        save_strategy="epoch", # Save checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model based on evaluation metric
        metric_for_best_model="f1",  # Use F1-score to determine the best model
        report_to="none",
        fp16=False, # Keep False when using bitsandbytes quantization
        save_total_limit=1, # Save only the best model
        seed=seed, # Set seed for reproducibility
    )

    tokenizer_for_trainer = AutoTokenizer.from_pretrained(model_name)

    # Initialize Trainer for the single training run
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # The entire merged train+dev dataset
        eval_dataset=eval_dataset,   # The test dataset, used for evaluation during training
        compute_metrics=compute_metrics,
        tokenizer=tokenizer_for_trainer,
    )

    print("Starting training on the full merged dataset...")
    trainer.train()

    print("Training finished. Evaluating the best model on the test set (final evaluation)...")
    final_test_results = trainer.evaluate(test_dataset) # Use the separate test_dataset for final evaluation
    print(f"\nFinal Test Set Evaluation Results: {final_test_results}")

    # Save final results
    with open(os.path.join(output_dir, "final_test_results.json"), "w") as f:
        json.dump(final_test_results, f, indent=4)
    print(f"Final test results saved to {os.path.join(output_dir, 'final_test_results.json')}")

    print(f"\nSaving the final best model to {output_dir}/final_best_model")
    trainer.save_model(os.path.join(output_dir, "final_best_model"))
    trainer.tokenizer.save_pretrained(os.path.join(output_dir, "final_best_model"))
    print("Training and evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ModernBERT-large for emotion classification with 4-bit quantization and LoRA (single run).")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-large",
                        help="Hugging Face model name or path of the pre-trained ModernBERT model.")
    parser.add_argument("--processed_data_dir", type=str, default="./processed_data_kfold",
                        help="Directory where the preprocessed dataset (from k-fold preprocess) is saved.")
    parser.add_argument("--output_dir", type=str, default="./model_output_single_run_quantized",
                        help="Directory to save the trained model checkpoints and results.")
    parser.add_argument("--num_epochs", type=int, default=1, # Default to 1 epoch
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size per device. Adjust based on GPU memory.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA r parameter (rank).")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    train_modernbert_single_run_quantized(
        model_name=args.model_name,
        processed_data_dir=args.processed_data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )