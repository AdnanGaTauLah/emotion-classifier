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
from datetime import datetime  # Correct import

# Configuration
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
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    # Load processed data
    train_data = load_from_disk("data/processed/train")
    test_data = load_from_disk("data/processed/test")
    labels = train_data["label"]

    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    splits = skf.split(np.zeros(len(labels)), labels)

    # Training loop
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Initialize wandb for each fold
        wandb.init(
            project="emotion-classifier",
            entity="adnanfatawi-electronic-engineering-polytechnic-institute",
            config={
                "model": args.model_name,
                "dataset": "MELD",
                "epochs": args.epochs,
                "batch_size": 32,
                "fold": fold
            },
            name=f"{args.model_name}-fold-{fold}-{datetime.now().strftime('%m%d-%H%M')}",
            reinit=True
        )

        print(f"\n{'='*40}")
        print(f"Training Fold {fold+1}/{NUM_FOLDS}")
        print(f"{'='*40}")

        # Create fold datasets
        fold_train = train_data.select(train_idx)
        fold_val = train_data.select(val_idx)
        
        # Load fresh model for each fold
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(np.unique(labels))
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # Training arguments
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
            hub_model_id=f"{args.model_name}-meld-fold-{fold}",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=fold_train,
            eval_dataset=fold_val,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)] # Fixed parameter name
        )

        # Train and save best model
        trainer.train()
        trainer.save_model(f"./models/fold_{fold}")

        # Save tokenizer
        tokenizer.save_pretrained(f"./models/fold_{fold}")

        # Evaluate on test set
        test_results = trainer.evaluate(test_data)
        print(f"\nFold {fold+1} Test Results:")
        print(f"Loss: {test_results['eval_loss']:.4f}")
        print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
        print(f"F1 Score: {test_results['eval_f1']:.4f}")

        # Push to Hub if enabled
        if args.push_to_hub:
            trainer.push_to_hub(commit_message=f"Add fold {fold} model")

        # Finish wandb run
        wandb.finish()

    print("\nTraining completed for all folds!")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train()