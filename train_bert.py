import argparse
import torch
import numpy as np
import random
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from load_datasets import create_balanced_dataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='binary'),
        "precision": precision_score(labels, preds, average='binary'),
        "recall": recall_score(labels, preds, average='binary'),
    }
def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = f"./{args.base_model.replace('/', '-')}-fine-tune"

    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        # Optimizer settings
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,

        # Training Stability
        max_grad_norm=args.max_grad_norm,

        # Evaluation and logging
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,

        # Saving strategy
        save_strategy="epoch",
        load_best_model_at_end=True,

        # Metric
        metric_for_best_model="accuracy",
        greater_is_better=True,

        # Other
        seed=args.seed,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    train_result = trainer.train()

    print("\nTraining completed!")
    print(f"Training metrics: {train_result.metrics}")

    return trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BERT Model Training Script")

    parser.add_argument(
        "--base_model",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Base model to use for fine-tuning"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Sample size for each dataset"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = "cuda"
    print(f"Using device: {device}")

    print(f"Loading dataset")
    ds_train, ds_val = create_balanced_dataset(sample_size=args.sample_size, seed=args.seed)
    print(f"Train size: {len(ds_train)}, Validation size: {len(ds_val)}")

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading model from {args.base_model}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
    )
    model.to(device)

    print("Tokenizing datasets...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_tokenized = ds_train.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    eval_tokenized = ds_val.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    trainer = train_model(
        model, tokenizer, train_tokenized, eval_tokenized, args
    )

    print("\nEvaluating model...")
    eval_results = trainer.evaluate(eval_tokenized)
    print(f"Evaluation results: {eval_results}")

    output_dir = f"./{args.base_model.replace('/', '-')}-fine-tune"
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()