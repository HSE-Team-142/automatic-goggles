import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs import build_experiment_config
from model import PAWN


DEFAULT_CONFIG = (
    "pawn++/experiments/MAGE/configs/two_models/"
    "mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full.yaml"
)
DEFAULT_CHECKPOINT = (
    "mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full/"
    "checkpoint-39884/pytorch_model.bin"
)
DEFAULT_RAID_SPLITS = ["train", "extra"]
RAID_DATA_URL = "https://dataset.raid-bench.xyz"
LABELED_RAID_SPLITS = {"train", "extra"}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def distributed_context() -> tuple[int, int, int]:
    """Initialize torchrun's process group and return rank, local rank, world size."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed OOD evaluation requires CUDA GPUs.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), local_rank, dist.get_world_size()


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


def combine_distributed_predictions(
    shards: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    dataset_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and order every rank's non-overlapping evaluation shard."""
    completed_shards = [shard for shard in shards if shard is not None]
    indices = np.concatenate([shard[0] for shard in completed_shards])
    labels = np.concatenate([shard[1] for shard in completed_shards])
    logits = np.concatenate([shard[2] for shard in completed_shards])

    expected_indices = np.arange(dataset_size)
    order = np.argsort(indices)
    if not np.array_equal(indices[order], expected_indices):
        raise RuntimeError(
            "Distributed evaluation did not receive exactly one prediction for every RAID row."
        )
    return labels[order], logits[order]


class RAIDTextDataset(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, int, str]:
        row = self.data[index]
        return row["generation"], int(row["model"] != "human"), row["id"]


def collate_raid_batch(batch: list[tuple[str, int, str]]) -> dict:
    texts, labels, ids = zip(*batch)
    return {
        "texts": list(texts),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "ids": list(ids),
    }


def write_predictions(
    file,
    ids: list[str],
    logits: torch.Tensor,
    include_header: bool,
) -> None:
    logits_array = logits.numpy().reshape(-1)
    pl.DataFrame(
        {
            "id": ids,
            "logit": logits_array,
            "probability": 1.0 / (1.0 + np.exp(-logits_array)),
            "prediction": (logits_array >= 0).astype(int),
        }
    ).write_csv(file, include_header=include_header)
    file.flush()
    os.fsync(file.fileno())


def merge_prediction_shards(shard_paths: list[Path], destination: Path) -> None:
    with destination.open("wb") as output_file:
        wrote_header = False
        for shard_path in shard_paths:
            with shard_path.open("rb") as shard_file:
                header = shard_file.readline()
                if not header:
                    continue
                if not wrote_header:
                    output_file.write(header)
                    wrote_header = True
                shutil.copyfileobj(shard_file, output_file)
        output_file.flush()
        os.fsync(output_file.fileno())

    for shard_path in shard_paths:
        shard_path.unlink()


def load_raid_datasets(splits: list[str]) -> dict[str, RAIDTextDataset]:
    unsupported_splits = set(splits) - LABELED_RAID_SPLITS
    if unsupported_splits:
        raise ValueError(
            "RAID OOD evaluation requires labeled splits. "
            f"Choose from {sorted(LABELED_RAID_SPLITS)}, not {sorted(unsupported_splits)}."
        )

    # RAID publishes these smaller, clean partitions specifically for evaluations
    # that do not include adversarial attacks.
    raid = load_dataset(
        "csv",
        data_files={split: f"{RAID_DATA_URL}/{split}_none.csv" for split in splits},
    )
    datasets = {}
    required_columns = {"id", "generation", "model"}
    for split in splits:
        data = raid[split]
        missing_columns = required_columns - set(data.column_names)
        if missing_columns:
            raise ValueError(
                f"RAID {split!r} split is missing required columns: {sorted(missing_columns)}"
            )
        datasets[split] = RAIDTextDataset(data)
    return datasets


def evaluate_dataset(
    model: PAWN,
    dataset: RAIDTextDataset,
    split: str,
    batch_size: int,
    output_dir: Path,
    rank: int,
    world_size: int,
    flush_every_batches: int,
) -> dict[str, float] | None:
    # Unlike DistributedSampler, this does not pad the final shard with duplicate rows.
    shard = Subset(dataset, range(rank, len(dataset), world_size))
    dataloader = DataLoader(
        shard,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_raid_batch,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{split}_predictions.csv"
    shard_path = (
        output_dir / f".{split}_rank{rank}_predictions.csv"
        if world_size > 1
        else prediction_path
    )
    all_logits = []
    all_labels = []
    local_indices = np.arange(rank, len(dataset), world_size)
    pending_ids = []
    pending_logits = []
    has_written_predictions = False

    model.eval()
    with shard_path.open("w", encoding="utf-8", newline="") as prediction_file, torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"RAID {split} | rank {rank}/{world_size} | {len(shard)} rows",
                position=rank,
                leave=True,
                dynamic_ncols=True,
            )
        ):
            labels = batch["labels"].cpu()
            logits = model(texts=batch["texts"]).detach().cpu()
            all_labels.append(labels)
            all_logits.append(logits)
            pending_ids.extend(batch["ids"])
            pending_logits.append(logits)

            if (batch_index + 1) % flush_every_batches == 0:
                write_predictions(
                    prediction_file,
                    pending_ids,
                    torch.cat(pending_logits),
                    include_header=not has_written_predictions,
                )
                pending_ids.clear()
                pending_logits.clear()
                has_written_predictions = True

        if pending_ids:
            write_predictions(
                prediction_file,
                pending_ids,
                torch.cat(pending_logits),
                include_header=not has_written_predictions,
            )

    local_labels = (
        torch.cat(all_labels).numpy().astype(int) if all_labels else np.empty(0, dtype=int)
    )
    local_logits = (
        torch.cat(all_logits).numpy().reshape(-1) if all_logits else np.empty(0, dtype=float)
    )

    if world_size > 1:
        gathered: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * world_size
        dist.gather_object(
            (local_indices, local_labels, local_logits),
            gathered if rank == 0 else None,
            dst=0,
        )
        if rank != 0:
            return None
        labels, logits = combine_distributed_predictions(gathered, len(dataset))
        merge_prediction_shards(
            [output_dir / f".{split}_rank{other_rank}_predictions.csv" for other_rank in range(world_size)],
            prediction_path,
        )
    else:
        labels = local_labels
        logits = local_logits
    metrics = compute_metrics(labels, logits)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = split
    with (output_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PAWN++ checkpoint on the RAID OOD dataset.")
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--splits", nargs="+", default=DEFAULT_RAID_SPLITS, help="RAID splits to evaluate.")
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--flush_every_batches",
        type=int,
        default=100,
        help="Flush prediction rows to disk after this many batches.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.flush_every_batches < 1:
        raise ValueError("--flush_every_batches must be at least 1.")
    rank, local_rank, world_size = distributed_context()
    try:
        config = load_config(args.config)
        device = f"cuda:{local_rank}" if world_size > 1 else args.device or default_device()
        batch_size = args.batch_size or config.data.eval_batch_size

        model = PAWN(config.model).to(device)
        load_pawn_checkpoint(model, args.checkpoint)

        datasets = load_raid_datasets(args.splits)
        all_metrics = {}
        for split, dataset in datasets.items():
            metrics = evaluate_dataset(
                model=model,
                dataset=dataset,
                split=split,
                batch_size=batch_size,
                output_dir=args.output_dir,
                rank=rank,
                world_size=world_size,
                flush_every_batches=args.flush_every_batches,
            )
            if rank == 0:
                all_metrics[split] = metrics
                print(f"\nRAID {split}")
                print(json.dumps(metrics, indent=2))

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "all_metrics.json").open("w", encoding="utf-8") as file:
                json.dump(all_metrics, file, indent=2)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
