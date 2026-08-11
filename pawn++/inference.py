import os
import argparse
import json

import torch
import yaml

from configs import ModelConfig
from model import PAWN
from dataset_module import TextDataset

LABEL_NAMES = {0: "human", 1: "ai"}


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_config(path: str) -> ModelConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"YAML config must contain a mapping at the top level: {path}")

    # Accept both a full experiment config and a bare model config.
    model_config = raw_config.get("model", raw_config)
    return ModelConfig.model_validate(model_config)


def load_model(
    config_path: str, checkpoint_path: str, device: str | None = None
) -> tuple[PAWN, str]:
    device = device or _default_device()
    config = load_model_config(config_path)

    model = PAWN(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
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
        raise ValueError(f"Missing trainable checkpoint keys: {missing_trainable}")
    model.to(device)
    model.eval()

    return model, device


@torch.no_grad()
def predict(
    model: PAWN,
    texts: list[str],
    device: str,
    batch_size: int = 32,
) -> list[dict]:
    results: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        logits = model(batch)
        probs = torch.sigmoid(logits)
        for text, logit, prob in zip(batch, logits.tolist(), probs.tolist()):
            pred = int(logit >= 0)
            results.append(
                {
                    "text": text,
                    "logit": logit,
                    "prob_human": prob,
                    "prediction": pred,
                    "label": LABEL_NAMES[pred],
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained PAWN model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint directory or a pytorch_model.bin file.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--text", type=str, nargs="+", help="One or more raw texts to classify."
    )
    source.add_argument(
        "--input_csv", type=str, help="Path to a CSV with a 'text' column."
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to write predictions as JSON.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Inference batch size."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (cuda/mps/cpu)."
    )

    return parser.parse_args()


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, "pytorch_model.bin")
    return checkpoint


def main() -> None:
    args = parse_args()

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    model, device = load_model(args.config, checkpoint_path, args.device)

    if args.text is not None:
        texts = args.text
    else:
        dataset = TextDataset(args.input_csv)
        texts = [dataset[i][0] for i in range(len(dataset))]

    results = predict(model, texts, device, batch_size=args.batch_size)

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

    for result in results:
        preview = result["text"][:80].replace("\n", " ")
        print(f"[{result['label']:>5}] p(human)={result['prob_human']:.4f}  {preview}")


if __name__ == "__main__":
    main()
