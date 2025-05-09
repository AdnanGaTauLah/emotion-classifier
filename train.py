import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import argparse
import wandb
from datetime import datetime

# Config
DEFAULT_MODEL_NAME = "roberta-base"
NUM_FOLDS = 2
SEED = 42
EARLY_STOPPING_PATIENCE = 2

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    # Load data
    train_data = load_from_disk("data/processed/train")
    test_data = load_from_disk("data/processed/test")
    label_list = np.load("data/label_list.npy", allow_pickle=True).tolist()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    # Map string labels to IDs
    train_data = train_data.map(lambda e: {"label": label2id[e["label"]]})
    test_data = test_data.map(lambda e: {"label": label2id[e["label"]]})

    # Cross-validation setup
    labels = train_data["label"]
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    splits = skf.split(np.zeros(len(labels)), labels)

    for fold, (train_idx, val_idx) in enumerate(splits):
        wandb.init(
            project="emotion-classifier",
            config={
                "model": args.model_name,
                "dataset": "MELD",
                "epochs": args.epochs,
                "fold": fold
            },
            name=f"{args.model_name}-fold-{fold}-{datetime.now().strftime('%m%d-%H%M')}",
            reinit=True
        )

        fold_train = train_data.select(train_idx)
        fold_val = train_data.select(val_idx)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"./logs/fold_{fold}",
            seed=SEED,
            report_to="wandb",
            run_name=f"fold-{fold}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
            push_to_hub=args.push_to_hub,
            hub_model_id=f"{args.model_name}-meld-fold-{fold}"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=fold_train,
            eval_dataset=fold_val,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        )

        trainer.train()
        trainer.save_model(f"./models/fold_{fold}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.save_pretrained(f"./models/fold_{fold}")
        wandb.finish()

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train()
