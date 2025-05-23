import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import argparse
from datetime import datetime

# Config
DEFAULT_MODEL_NAME = "roberta-base"
NUM_FOLDS = 10
SEED = 42
EARLY_STOPPING_PATIENCE = 2

def compute_metrics_builder(metrics_log, fold):
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_score(p.label_ids, preds)
        prec = precision_score(p.label_ids, preds, average="weighted", zero_division=0)
        rec = recall_score(p.label_ids, preds, average="weighted", zero_division=0)
        f1 = f1_score(p.label_ids, preds, average="weighted")

        epoch = trainer.state.epoch if trainer.state.epoch is not None else -1

        metrics_log.append({
            "fold": fold,
            "epoch": epoch,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
    return compute_metrics

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    if device == "cpu":
        print("Warning: Running on CPU. This will be significantly slower than GPU training.")
        # Adjust batch size for CPU
        args.batch_size = min(args.batch_size, 8)

    # Load data
    train_data = load_from_disk("data/processed/train")
    test_data = load_from_disk("data/processed/test")
    label_list = np.load("data/label_list.npy", allow_pickle=True).tolist()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    # Map string labels to IDs
    train_data = train_data.map(lambda e: {"label": label2id[e["label"]]})
    test_data = test_data.map(lambda e: {"label": label2id[e["label"]]})

    labels = train_data["label"]
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    splits = skf.split(np.zeros(len(labels)), labels)

    os.makedirs("logs/metrics", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n=== Training Fold {fold+1}/{NUM_FOLDS} ===")

        fold_train = train_data.select(train_idx)
        fold_val = train_data.select(val_idx)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )

        metrics_log = []

        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=64,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"./logs/fold_{fold}",
            seed=SEED,
            report_to="none",  # Disable W&B
            run_name=f"fold-{fold}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
            push_to_hub=args.push_to_hub,
            hub_model_id=f"{args.model_name}-meld-fold-{fold}"
        )

        global trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=fold_train,
            eval_dataset=fold_val,
            compute_metrics=compute_metrics_builder(metrics_log, fold),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        )

        trainer.train()

        # Save metrics log
        pd.DataFrame(metrics_log).to_csv(f"logs/metrics/fold_{fold}_metrics.csv", index=False)

        # Save model/tokenizer
        trainer.save_model(f"./models/fold_{fold}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.save_pretrained(f"./models/fold_{fold}")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train()
