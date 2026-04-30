from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from tqdm.auto import tqdm

from checkpoint_6.dataset_module import TextDataset, collate_text_batch
from model import PAWN, PAWNConfig


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device else default_device())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    train_loader = DataLoader(
        TextDataset(args.train_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_text_batch,
    )
    eval_loader = DataLoader(
        TextDataset(args.validation_dataset),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_text_batch,
    )
    test_loader = DataLoader(
        TextDataset(args.test_dataset),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_text_batch,
    )

    config = PAWNConfig(
        max_length=args.max_length,
        metric_features=args.metric_features,
        gates=args.gates,
        mlp_hidden_features=args.mlp_hidden_features,
        mlp_hidden_layers=args.mlp_hidden_layers,
        mlp_dropout=args.mlp_dropout,
        token_dropout=args.token_dropout,
        model_name=args.model_name,
    )
    model = PAWN(config).to(device)

    train(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=device,
        output_dir=output_dir,
    )

    test_metrics = evaluate(model, test_loader, device)
    print(
        "test "
        f"loss={test_metrics['loss']:.4f} "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )

    final_path = output_dir / "pawn_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "args": vars(args),
            "test_metrics": test_metrics,
        },
        final_path,
    )
    print(f"saved final checkpoint to {final_path}")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float | None,
    device: torch.device,
    output_dir: Path,
) -> None:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_f1 = -math.inf
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            grad_clip=grad_clip,
            device=device,
            epoch=epoch,
            epochs=epochs,
        )
        metrics = evaluate(model, eval_loader, device, desc=f"eval epoch {epoch}/{epochs}")
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"eval_loss={metrics['loss']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            checkpoint_path = output_dir / "pawn_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_metrics": metrics,
                },
                checkpoint_path,
            )
            print(f"saved best checkpoint to {checkpoint_path}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float | None,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    progress = tqdm(loader, desc=f"train epoch {epoch}/{epochs}", leave=False)
    for batch in progress:
        texts = batch["texts"]
        labels = batch["labels"].to(device)
        logits = model(texts).squeeze(-1)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / max(total_examples, 1))

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "eval",
) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    accuracy_metric = BinaryAccuracy().to(device)
    macro_f1_metric = BinaryF1Score().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    roc_auc_metric = BinaryAUROC().to(device)

    total_loss = 0.0
    total_examples = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        texts = batch["texts"]
        labels = batch["labels"].to(device)
        logits = model(texts).squeeze(-1)
        loss = criterion(logits, labels)
        targets = labels.long()

        accuracy_metric.update(logits, targets)
        macro_f1_metric.update(logits, targets)
        precision_metric.update(logits, targets)
        recall_metric.update(logits, targets)
        roc_auc_metric.update(logits, targets)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / max(total_examples, 1))

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": float(accuracy_metric.compute().item()),
        "macro_f1": float(macro_f1_metric.compute().item()),
        "precision": float(precision_metric.compute().item()),
        "recall": float(recall_metric.compute().item()),
        "roc_auc": float(roc_auc_metric.compute().item()),
    }


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Path to a YAML file with training parameters.")
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Train checkpoint_6 PAWN on text/label CSV datasets.")
    parser.add_argument("--config", default=None, help="Path to a YAML file with training parameters.")
    dataset_paths_required = config_args.config is None
    parser.add_argument("--train_dataset", type=str, required=dataset_paths_required, help="Path to train CSV.")
    parser.add_argument("--validation_dataset", type=str, required=dataset_paths_required, help="Path to validation CSV.")
    parser.add_argument("--test_dataset", type=str, required=dataset_paths_required, help="Path to test CSV.")
    parser.add_argument("--model_name", default="openai-community/gpt2", help="Frozen causal LM backbone.")
    parser.add_argument("--output_dir", default=os.path.join("checkpoint_6", "outputs"))
    parser.add_argument("--device", default=None, help="Device override, for example cuda, mps, or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--metric_features", type=int, default=256)
    parser.add_argument("--gates", type=int, default=256)
    parser.add_argument("--mlp_hidden_features", type=int, default=256)
    parser.add_argument("--mlp_hidden_layers", type=int, default=3)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--token_dropout", type=float, default=0.15)

    if config_args.config is not None:
        yaml_config = load_yaml_config(config_args.config)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(yaml_config) - valid_keys)
        if unknown_keys:
            parser.error(f"unknown YAML config keys: {unknown_keys}")
        parser.set_defaults(**yaml_config)

    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping at the top level: {path}")

    return flatten_config(config)


def flatten_config(config: dict) -> dict:
    values = {}
    for key, value in config.items():
        if isinstance(value, dict):
            values.update(flatten_config(value))
        else:
            values[key] = value
    return values


if __name__ == "__main__":
    main()
