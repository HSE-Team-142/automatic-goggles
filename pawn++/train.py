import os
import torch 
import argparse
import json
import yaml

import numpy as np

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from configs import ExperimentConfig, build_experiment_config

from dataset_module import TextDataset, collate_text_batch
from model import PAWN


class PAWNTrainer(Trainer):
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
        logits = model(**inputs)

        smoothed = labels.float() * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        pos_weight = torch.tensor(self._pos_weight, device=logits.device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pos_weight)

        if return_outputs:
          return (loss, {"logits": logits})
        return loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def compute_metrics(eval_pred):
      logits = np.asarray(eval_pred.predictions).reshape(-1)
      labels = np.asarray(eval_pred.label_ids).reshape(-1).astype(int)

      preds = (logits >= 0).astype(int)

      return {
          "accuracy": accuracy_score(labels, preds),

          "ai_f1": f1_score(labels, preds, zero_division=0),
          "ai_precision": precision_score(labels, preds, zero_division=0),
          "ai_recall": recall_score(labels, preds, zero_division=0),

          "human_f1": f1_score(1 - labels, 1 - preds, zero_division=0),
          "human_precision": precision_score(1 - labels, 1 - preds, zero_division=0),
          "human_recall": recall_score(1 - labels, 1 - preds, zero_division=0),

          "roc_auc": roc_auc_score(labels, logits),
          "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
      }


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_training_args(args):
    config = args.config
    optimizer_config = config.optimizer
    trainer_config = config.trainer
    data_config = config.data


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
        num_train_epochs=trainer_config.epochs,
        per_device_train_batch_size=data_config.batch_size,
        per_device_eval_batch_size=data_config.eval_batch_size,

        # Optimizer settings
        learning_rate=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,

        # Training Stability
        max_grad_norm=optimizer_config.max_grad_norm,
        gradient_accumulation_steps=optimizer_config.gradient_accumulation_steps,

        # Scheduler
        lr_scheduler_type="cosine",
        warmup_steps=0,

        # Evaluation
        eval_strategy="epoch",

        # Saving strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,

        # Metric
        metric_for_best_model="roc_auc",
        greater_is_better=True,

        # Logging
        logging_strategy="steps",
        logging_steps=10,
        **logging_kwargs,

        # Seed
        seed=trainer_config.seed,
    )

    return training_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PAWN.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML file with training parameters.")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to train CSV.")
    parser.add_argument("--valid_dataset", type=str, required=True, help="Path to validation CSV.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to test CSV.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory override. Defaults to trainer.output_dir from the YAML config.",)
    parser.add_argument("--report_to", type=str, choices=["tensorboard", "mlflow"], default="mlflow", help="Metrics logging backend for Hugging Face Trainer.")

    args = parser.parse_args()
    try:
        config = load_yaml_config(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    args.config = config
    return args


def load_yaml_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"YAML config must contain a mapping at the top level: {path}")

    return build_experiment_config(raw_config)


def main() -> None:
    args = parse_args()
    config = args.config
    model_config = config.model
    optimizer_config = config.optimizer

    device = _default_device()

    train_dataset = TextDataset(args.train_dataset)

    eval_dataset = TextDataset(args.valid_dataset)

    test_dataset = TextDataset(args.test_dataset)

    model = PAWN(model_config).to(device)

    training_args = _get_training_args(args)

    trainer = PAWNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_text_batch,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=5,
        )],
        label_smoothing=optimizer_config.label_smoothing,
        pos_weight=optimizer_config.pos_weight,
    )

    if args.report_to == "mlflow":
        import mlflow

        mlflow.set_experiment("pawn")

    trainer.train()

    prediction_output = trainer.predict(test_dataset)

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(prediction_output.metrics, f, indent=2)


if __name__ == "__main__":
    main()
