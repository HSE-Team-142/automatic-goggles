import os
import torch 
import argparse
import json
import yaml

import numpy as np

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from tqdm import tqdm
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
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("feature_extractor.")
        }
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


class CachedFeatureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: TextDataset,
        model: PAWN,
        batch_size: int,
        description: str,
    ) -> None:
        self.samples = []
        feature_extractor = model.feature_extractor
        feature_extractor.eval()

        for start in tqdm(range(0, len(dataset), batch_size), desc=description):
            texts = dataset.texts[start : start + batch_size]
            labels = dataset.labels[start : start + batch_size]
            features = feature_extractor(texts)
            attention_mask = features["attention_mask"].detach().cpu()
            lengths = attention_mask.sum(dim=1).tolist()

            for index, label in enumerate(labels):
                token_length = int(lengths[index])
                metric_length = max(token_length - 1, 0)

                sample_features = {
                    "metrics": features["metrics"][index, :metric_length].detach().cpu(),
                    "agg_metrics": (
                        features["agg_metrics"][index].detach().cpu()
                        if features["agg_metrics"] is not None
                        else None
                    ),
                    "primary_hidden_states": features["primary_hidden_states"][index, :token_length].detach().cpu(),
                    "second_hidden_states": (
                        features["second_hidden_states"][index, :token_length].detach().cpu()
                        if features["second_hidden_states"] is not None
                        else None
                    ),
                    "attention_mask": attention_mask[index, :token_length],
                }
                self.samples.append((sample_features, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate_feature_batch(batch: list[tuple[dict[str, torch.Tensor], float]]) -> dict[str, torch.Tensor]:
    sample_features, labels = zip(*batch)
    features = {
        "metrics": torch.nn.utils.rnn.pad_sequence(
            [sample["metrics"] for sample in sample_features],
            batch_first=True,
        ),
        "agg_metrics": None,
        "primary_hidden_states": torch.nn.utils.rnn.pad_sequence(
            [sample["primary_hidden_states"] for sample in sample_features],
            batch_first=True,
        ),
        "second_hidden_states": None,
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [sample["attention_mask"] for sample in sample_features],
            batch_first=True,
        ),
    }
    if sample_features[0]["agg_metrics"] is not None:
        features["agg_metrics"] = torch.stack([sample["agg_metrics"] for sample in sample_features])
    if sample_features[0]["second_hidden_states"] is not None:
        features["second_hidden_states"] = torch.nn.utils.rnn.pad_sequence(
            [sample["second_hidden_states"] for sample in sample_features],
            batch_first=True,
        )

    return {
        "features": features,
        "labels": torch.tensor(labels, dtype=torch.float32),
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
        remove_unused_columns=False,
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
    parser.add_argument(
        "--precompute_features",
        action="store_true",
        help="Cache frozen LM features in memory before training. Faster for multi-epoch runs, but can use a lot of RAM.",
    )

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
    data_collator = collate_text_batch
    if args.precompute_features:
        train_dataset = CachedFeatureDataset(
            train_dataset,
            model,
            batch_size=config.data.batch_size,
            description="precompute train features",
        )
        eval_dataset = CachedFeatureDataset(
            eval_dataset,
            model,
            batch_size=config.data.eval_batch_size,
            description="precompute validation features",
        )
        test_dataset = CachedFeatureDataset(
            test_dataset,
            model,
            batch_size=config.data.eval_batch_size,
            description="precompute test features",
        )
        data_collator = collate_feature_batch

    trainer = PAWNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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
