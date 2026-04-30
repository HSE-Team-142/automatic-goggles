from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

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

from configs import ExperimentConfig, build_experiment_config
from dataset_module import TextDataset, collate_text_batch
from model import PAWN


def main() -> None:
    args = parse_args()
    config = args.config
    model_config = config.model
    optimizer_config = config.optimizer
    trainer_config = config.trainer
    data_config = config.data

    set_seed(trainer_config.seed)

    device_name = trainer_config.device or default_device()
    device = torch.device(device_name)
    output_dir = Path(trainer_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        TextDataset(args.train_dataset),
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        collate_fn=collate_text_batch,
    )
    eval_loader = DataLoader(
        TextDataset(args.valid_dataset),
        batch_size=data_config.eval_batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        collate_fn=collate_text_batch,
    )
    test_loader = DataLoader(
        TextDataset(args.test_dataset),
        batch_size=data_config.eval_batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        collate_fn=collate_text_batch,
    )

    model = PAWN(model_config).to(device)

    train(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=trainer_config.epochs,
        learning_rate=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        grad_clip=optimizer_config.grad_clip,
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
    parser = argparse.ArgumentParser(description="Train checkpoint_6 PAWN on text/label CSV datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML file with training parameters.")
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to train CSV.")
    parser.add_argument("--valid_dataset", type=str, required=True, help="Path to validation CSV.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to test CSV.")

    args = parser.parse_args()
    try:
        config = load_yaml_config(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    args.config = config
    return SimpleNamespace(**config)


def load_yaml_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"YAML config must contain a mapping at the top level: {path}")

    return build_experiment_config(raw_config)


if __name__ == "__main__":
    main()
