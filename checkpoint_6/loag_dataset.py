from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset


def load_mage_dataset(
    dataset_name: str,
    train_samples: int | None,
    eval_samples: int | None,
    test_samples: int | None,
    seed: int,
) -> DatasetDict:
    ds = load_dataset(dataset_name)

    keep_columns = {"text", "label"}
    ds = DatasetDict(
        {
            split: dataset.remove_columns(
                [column for column in dataset.column_names if column not in keep_columns]
            )
            for split, dataset in ds.items()
        }
    )

    def _sample(dataset: Dataset, sample_size: int | None, seed: int) -> Dataset:
        if sample_size is None or sample_size <= 0 or sample_size >= len(dataset):
            return dataset
        return dataset.shuffle(seed=seed).select(range(sample_size))

    return DatasetDict(
        {
            "train": _sample(ds["train"], train_samples, seed),
            "validation": _sample(ds["validation"], eval_samples, seed),
            "test": _sample(ds["test"], test_samples, seed),
        }
    )


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[str, float]:
        row = self.dataset[index]
        return str(row["text"]), float(row["label"])


def collate_text_batch(batch: list[tuple[str, float]]) -> dict[str, Any]:
    texts, labels = zip(*batch)
    return {
        "texts": list(texts),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }
