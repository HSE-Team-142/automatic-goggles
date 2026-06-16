import polars as pl
import argparse
from pathlib import Path


def sample_df(path_to_df: Path, n_samples_per_class: int = 5000, seed: int = 42) -> pl.DataFrame:
    df = pl.read_csv(path_to_df)

    df_human = df.filter(pl.col("label") == 1).sample(n_samples_per_class, seed=seed)
    df_ai = df.filter(pl.col("label") == 0).sample(n_samples_per_class, seed=seed)

    df_sampled = pl.concat([df_human, df_ai]).sample(fraction=1.0, shuffle=True, seed=seed)

    return df_sampled


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to train CSV.")
    parser.add_argument("--valid_dataset", type=str, required=False, help="Path to validation CSV.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    train_path = Path(args.train_dataset)
    output_dir = train_path.parent

    train_sampled = sample_df(args.train_dataset, 2500)
    train_sampled.write_csv(output_dir / "train_sampled.csv")

    if args.valid_dataset is not None:
        valid_path = Path(args.valid_dataset)
        valid_sampled = sample_df(valid_path, 1000)
        valid_sampled.write_csv(output_dir / "valid_sampled.csv")


if __name__ == "__main__":
    main()
