import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset_module import TextDataset
from train_bert import BertTextCollator


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


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    ai_scores = logits[:, 1] - logits[:, 0]
    preds = (ai_scores >= 0).astype(int)

    ai_recall = recall_score(labels, preds, zero_division=0)
    human_recall = recall_score(1 - labels, 1 - preds, zero_division=0)

    return {
        "accuracy": accuracy_score(labels, preds),
        "ai_f1": f1_score(labels, preds, zero_division=0),
        "ai_precision": precision_score(labels, preds, zero_division=0),
        "ai_recall": ai_recall,
        "human_f1": f1_score(1 - labels, 1 - preds, zero_division=0),
        "human_precision": precision_score(1 - labels, 1 - preds, zero_division=0),
        "human_recall": human_recall,
        "avg_recall": 0.5 * (ai_recall + human_recall),
        "roc_auc": roc_auc_score(labels, ai_scores),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def evaluate_dataset(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    dataset_path: Path,
    batch_size: int,
    max_length: int,
    output_dir: Path,
    device: str,
) -> dict[str, float]:
    dataset = TextDataset(str(dataset_path))
    collator = BertTextCollator(tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    all_logits = []
    all_labels = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"evaluate {dataset_path.name}"):
            labels = batch.pop("labels").cpu()
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.detach().cpu()

            all_labels.append(labels)
            all_logits.append(logits)

    labels = torch.cat(all_labels).numpy().astype(int)
    logits = torch.cat(all_logits).numpy()
    metrics = compute_metrics(labels, logits)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    with (output_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    binary_logits = logits[:, 1] - logits[:, 0]
    probabilities = 1.0 / (1.0 + np.exp(-binary_logits))
    preds = (binary_logits >= 0).astype(int)
    pl.DataFrame(
        {
            "label": labels,
            "human_logit": logits[:, 0],
            "ai_logit": logits[:, 1],
            "binary_logit": binary_logits,
            "ai_probability": probabilities,
            "prediction": preds,
        }
    ).write_csv(output_dir / f"{stem}_predictions.csv")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BERT/Longformer checkpoint on MAGE OOD GPT testbeds.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to HF checkpoint directory.")
    parser.add_argument("--base_model", type=str, default=None, help="Tokenizer model name/path. Defaults to checkpoint.")
    parser.add_argument("--datasets", type=Path, nargs="+", default=[Path(path) for path in DEFAULT_DATASETS])
    parser.add_argument("--output_dir", type=Path, default=Path("bert_mage_ood_eval"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or default_device()
    tokenizer_path = args.base_model or str(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)

    all_metrics = {}
    for dataset_path in args.datasets:
        metrics = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            output_dir=args.output_dir,
            device=device,
        )
        all_metrics[str(dataset_path)] = metrics
        print(f"\n{dataset_path}")
        print(json.dumps(metrics, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "all_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)


if __name__ == "__main__":
    main()
