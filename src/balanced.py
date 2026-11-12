"""Create a class-balanced CSV from the seven-dialect dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/my_7dialect_data.csv")
    parser.add_argument("--out", default="data/balanced_833_per_dialect.csv")
    parser.add_argument("--label_col", default="dialect")
    parser.add_argument("--rows_per_class", type=int, default=833)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_path = Path(args.out)

    frame = pd.read_csv(data_path)
    smallest_class = int(frame[args.label_col].value_counts().min())
    if args.rows_per_class > smallest_class:
        raise ValueError(
            f"rows_per_class={args.rows_per_class} exceeds the smallest class "
            f"size ({smallest_class})."
        )

    balanced = (
        frame.groupby(args.label_col, group_keys=False)
        .sample(n=args.rows_per_class, random_state=args.seed)
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(balanced):,} rows to {out_path}")


if __name__ == "__main__":
    main()

