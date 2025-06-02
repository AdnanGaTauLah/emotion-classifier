# train.py (Updated for fp16 RuntimeError fix with bitsandbytes)
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
from sklearn.model_selection import KFold

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
    processed_data_dir: str = "./processed_data_kfold",
    output_dir: str = "./model_output_kfold",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    num_folds: int = 5,
    seed: int = 42
):
    """
    Trains ModernBERT-large for emotion classification using PEFT (LoRA)
    and 4-bit quantization with k-fold cross-validation.
    Includes enhanced logging per epoch/fold and saves the best model for each fold.
    """
    print(f"Loading processed data from {processed_data_dir}...")
    try:
        full_train_dataset = load_from_disk(os.path.join(processed_data_dir, "train"))
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

    # Configure 4-bit quantization (same for all folds)
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Configure LoRA (same for all folds)
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value", "key", "dense"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_metrics = []
    tokenizer_for_trainer = AutoTokenizer.from_pretrained(model_name)

    print(f"\nStarting {num_folds}-fold cross-validation...")
    for fold, (train_index, val_index) in enumerate(kf.split(full_train_dataset)):
        print(f"\n--- Fold {fold + 1}/{num_folds} ---")

        # Create train and validation subsets for the current fold
        train_subset = full_train_dataset.select(train_index)
        val_subset = full_train_dataset.select(val_index)

        print(f"Fold {fold + 1} training samples: {len(train_subset)}")
        print(f"Fold {fold + 1} validation samples: {len(val_subset)}")

        # Initialize a NEW model for each fold to ensure fresh start
        print(f"Loading fresh model for Fold {fold + 1}...")
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

        # Set up training arguments for the current fold
        fold_output_dir = os.path.join(output_dir, f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.06,
            weight_decay=0.01,
            logging_dir=os.path.join(fold_output_dir, "logs"),
            logging_steps=500,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            # CORRECTED: Set fp16 to False when using bitsandbytes quantization
            fp16=False,
            save_total_limit=1,
        )

        # Initialize Trainer for the current fold
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer_for_trainer,
        )

        print(f"Starting training for Fold {fold + 1}...")
        trainer.train()

        print(f"Evaluating Fold {fold + 1} on its validation set (best model)...")
        eval_results = trainer.evaluate()
        print(f"Fold {fold + 1} Validation Results: {eval_results}")
        fold_metrics.append(eval_results)

        # Explicitly save the best model for this fold
        best_model_path = os.path.join(fold_output_dir, "best_model_fold")
        trainer.save_model(best_model_path)
        trainer.tokenizer.save_pretrained(best_model_path)
        print(f"Best model for Fold {fold + 1} saved to {best_model_path}")

    print("\n--- K-Fold Cross-Validation Complete ---")

    # Aggregate and print average metrics
    avg_metrics = {
        "avg_f1": np.mean([m['eval_f1'] for m in fold_metrics]),
        "avg_accuracy": np.mean([m['eval_accuracy'] for m in fold_metrics]),
        "avg_precision": np.mean([m['eval_precision'] for m in fold_metrics]),
        "avg_recall": np.mean([m['eval_recall'] for m in fold_metrics]),
        "avg_runtime": np.mean([m['eval_runtime'] for m in fold_metrics]),
        "avg_samples_per_second": np.mean([m['eval_samples_per_second'] for m in fold_metrics]),
        "avg_steps_per_second": np.mean([m['eval_steps_per_second'] for m in fold_metrics]),
        "avg_loss": np.mean([m['eval_loss'] for m in fold_metrics])
    }
    print("\nAverage K-Fold Validation Metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save aggregated metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "kfold_average_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"Average K-Fold metrics saved to {os.path.join(output_dir, 'kfold_average_metrics.json')}")

    print("\nEvaluating the test set with the best model from the *last* fold.")
    final_model_path = os.path.join(output_dir, f"fold_{num_folds}", "best_model_fold")
    if os.path.exists(final_model_path):
        print(f"Loading best model from the last fold ({final_model_path}) for final test evaluation...")
        model_for_test = AutoModelForSequenceClassification.from_pretrained(
            final_model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        
        test_trainer = Trainer(
            model=model_for_test,
            args=TrainingArguments(
                output_dir=os.path.join(output_dir, "final_test_evaluation"),
                per_device_eval_batch_size=batch_size,
                report_to="none",
                fp16=False, # Keep false for consistency with bitsandbytes during test evaluation
            ),
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer_for_trainer,
        )
        test_results = test_trainer.evaluate()
        print(f"\nFinal Test Set Evaluation Results (using last fold's best model): {test_results}")
        with open(os.path.join(output_dir, "final_test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)
        print(f"Final test results saved to {os.path.join(output_dir, 'final_test_results.json')}")
    else:
        print(f"Could not find best model from the last fold at {final_model_path} to perform final test evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ModernBERT-large for emotion classification using PEFT and 4-bit quantization with k-fold cross-validation.")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-large",
                        help="Hugging Face model name or path of the pre-trained ModernBERT model.")
    parser.add_argument("--processed_data_dir", type=str, default="./processed_data_kfold",
                        help="Directory where the preprocessed dataset (for k-fold) is saved.")
    parser.add_argument("--output_dir", type=str, default="./model_output_kfold",
                        help="Directory to save the trained model checkpoints and results.")
    parser.add_argument("--num_epochs", type=int, default=3,
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
    parser.add_argument("--num_folds", type=int, default=5,
                        help="Number of folds for k-fold cross-validation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for k-fold shuffling.")
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
        num_folds=args.num_folds,
        seed=args.seed,
    )