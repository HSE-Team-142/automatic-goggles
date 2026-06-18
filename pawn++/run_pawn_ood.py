import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import build_experiment_config
from dataset_module import TextDataset, collate_text_batch
from model import PAWN


DEFAULT_CONFIG = (
    "pawn++/experiments/MAGE/configs/two_models/"
    "mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full.yaml"
)
DEFAULT_CHECKPOINT = (
    "mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full/"
    "checkpoint-39884/pytorch_model.bin"
)
DEFAULT_DATASETS = [
    "mage/testbeds/test_ood_gpt_para.csv",
    "mage/testbeds/test_ood_gpt.csv",
]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    return build_experiment_config(raw_config)


def load_pawn_checkpoint(model: PAWN, checkpoint_path: Path) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    trainable_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("feature_extractor.")
    }
    if not trainable_state_dict:
        raise ValueError(f"No trainable PAWN weights found in checkpoint: {checkpoint_path}")

    missing_keys, unexpected_keys = model.load_state_dict(trainable_state_dict, strict=False)
    unexpected_trainable = [
        key for key in unexpected_keys
        if not key.startswith("feature_extractor.")
    ]
    missing_trainable = [
        key for key in missing_keys
        if not key.startswith("feature_extractor.")
    ]
    if unexpected_trainable:
        raise ValueError(f"Unexpected trainable checkpoint keys: {unexpected_trainable}")
    if missing_trainable:
        raise ValueError(f"Missing trainable model keys: {missing_trainable}")


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
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


def evaluate_dataset(model: PAWN, dataset_path: Path, batch_size: int, output_dir: Path) -> dict[str, float]:
    dataset = TextDataset(str(dataset_path))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_text_batch,
    )

    all_logits = []
    all_labels = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"evaluate {dataset_path.name}"):
            labels = batch["labels"].cpu()
            logits = model(texts=batch["texts"]).detach().cpu()
            all_labels.append(labels)
            all_logits.append(logits)

    labels = torch.cat(all_labels).numpy().astype(int)
    logits = torch.cat(all_logits).numpy().reshape(-1)
    metrics = compute_metrics(labels, logits)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    with (output_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    pl.DataFrame(
        {
            "label": labels,
            "logit": logits,
            "probability": 1.0 / (1.0 + np.exp(-logits)),
            "prediction": (logits >= 0).astype(int),
        }
    ).write_csv(output_dir / f"{stem}_predictions.csv")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PAWN++ checkpoint on MAGE OOD GPT testbeds.")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--datasets", type=Path, nargs="+", default=[Path(path) for path in DEFAULT_DATASETS])
    parser.add_argument("--output_dir", type=Path, default=Path("mage_ood_eval"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = args.device or default_device()
    batch_size = args.batch_size or config.data.eval_batch_size

    model = PAWN(config.model).to(device)
    load_pawn_checkpoint(model, args.checkpoint)

    all_metrics = {}
    for dataset_path in args.datasets:
        metrics = evaluate_dataset(model, dataset_path, batch_size, args.output_dir)
        all_metrics[str(dataset_path)] = metrics
        print(f"\n{dataset_path}")
        print(json.dumps(metrics, indent=2))

    with (args.output_dir / "all_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)


if __name__ == "__main__":
    main()
