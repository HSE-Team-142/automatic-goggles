import polars as pl
from typing import Any

import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str) -> None:
        self.data = pl.read_csv(data_path)

    def __len__(self) -> int:
        return self.data.height

    def __getitem__(self, index: int) -> tuple[str, float]:
        row = self.data[index]
        label = row["label"].item()
        text = row["text"].item()

        return text, label

def collate_text_batch(batch: list[tuple[str, float]]) -> dict[str, Any]:
    texts, labels = zip(*batch)
    return {
        "texts": list(texts),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }
