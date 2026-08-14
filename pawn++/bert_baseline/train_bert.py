import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from dataset_module import TextDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _running_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


class BertTextCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, float]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch)

        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


def compute_metrics(eval_pred):
    logits = np.asarray(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids).reshape(-1).astype(int)

    preds = logits.argmax(-1)
    ai_scores = logits[:, 1]
    
    ai_recall = recall_score(labels, preds, zero_division=0)
    human_recall = recall_score(1 - labels, 1 - preds, zero_division=0)
    avg_recall = 0.5 * (ai_recall + human_recall)

    return {
          "accuracy": accuracy_score(labels, preds),

          "ai_f1": f1_score(labels, preds, zero_division=0),
          "ai_precision": precision_score(labels, preds, zero_division=0),
          "ai_recall": ai_recall,

          "human_f1": f1_score(1 - labels, 1 - preds, zero_division=0),
          "human_precision": precision_score(1 - labels, 1 - preds, zero_division=0),
          "human_recall": human_recall,

          "avg_recall": avg_recall,

          "roc_auc": roc_auc_score(labels, ai_scores),
          "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
      }


class BertTrainer(Trainer):
    def __init__(
        self,
        *args,
        label_smoothing: float = 0.0,
        pos_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_smoothing = float(label_smoothing)
        self._pos_weight = float(pos_weight)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        binary_logits = logits[:, 1] - logits[:, 0]
        smoothed = labels.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        pos_weight = torch.tensor(self._pos_weight, device=binary_logits.device, dtype=binary_logits.dtype)
        loss = F.binary_cross_entropy_with_logits(binary_logits, smoothed, pos_weight=pos_weight)

        if return_outputs:
            return (loss, outputs)
        return loss


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    data_collator = BertTextCollator(tokenizer=tokenizer, max_length=args.max_length)

    logging_kwargs = {
        "report_to": args.report_to,
    }
    if args.report_to == "tensorboard":
        logging_kwargs["logging_dir"] = os.path.join(args.output_dir, "tensorboard")
    elif args.report_to == "mlflow":
        logging_kwargs["run_name"] = args.output_dir

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        
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
        ddp_find_unused_parameters=False if _running_distributed() else None,

        # Evaluation and logging
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,

        # Saving strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,

        # Metric
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,

        # Other
        seed=args.seed,
        **logging_kwargs,
    )

    trainer = BertTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        label_smoothing=args.label_smoothing,
        pos_weight=args.pos_weight,
    )

    if _is_main_process():
        print("Starting training...")
    train_result = trainer.train()

    if trainer.is_world_process_zero():
        print("\nTraining completed!")
        print(f"Training metrics: {train_result.metrics}")

    return trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BERT Model Training Script")

    parser.add_argument("--train_dataset", type=str, required=True, help="Path to train CSV.")
    parser.add_argument("--valid_dataset", type=str, required=True, help="Path to validation CSV.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to test CSV.")
    parser.add_argument("--output_dir", type=str, default="bert_output", help="Output directory.")
    parser.add_argument(
        "--report_to",
        type=str,
        choices=["tensorboard", "mlflow", "none"],
        default="tensorboard",
        help="Metrics logging backend for Hugging Face Trainer.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="FacebookAI/roberta-base",
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
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing value for BCE loss. Same formula as PAWN.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Positive class weight for BCE loss. Positive class is label 1 / AI.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="avg_recall",
        help="Metric for selecting the best model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    if _is_main_process():
        print(f"Using device: {_default_device()}")

    if _is_main_process():
        print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if _is_main_process():
        print(f"Loading model from {args.base_model}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
    )

    if _is_main_process():
        print("Loading CSV datasets...")
    train_dataset = TextDataset(args.train_dataset)
    eval_dataset = TextDataset(args.valid_dataset)
    test_dataset = TextDataset(args.test_dataset)
    if _is_main_process():
        print(f"Train size: {len(train_dataset)}, Validation size: {len(eval_dataset)}, Test size: {len(test_dataset)}")

    if args.report_to == "mlflow":
        import mlflow

        mlflow.set_experiment("bert")

    trainer = train_model(
        model, tokenizer, train_dataset, eval_dataset, args
    )

    if trainer.is_world_process_zero():
        print("\nEvaluating model...")
    eval_results = trainer.evaluate(eval_dataset)
    if trainer.is_world_process_zero():
        print(f"Evaluation results: {eval_results}")

    if trainer.is_world_process_zero():
        print("\nPredicting test dataset...")
    prediction_output = trainer.predict(test_dataset)
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump(prediction_output.metrics, f, indent=2)

        print(f"\nSaving model to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
