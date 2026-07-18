"""Prepare Mario world-model data and optionally publish it to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from world_model.conversion import ConversionConfig, convert_dataset, publish_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Mario HDF5 transitions into an H100-friendly NumPy cache"
    )
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument(
        "--break-index",
        action="append",
        type=int,
        default=[],
        help="Transition index that starts a new trajectory; repeat as needed",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--hf-repo", type=str, help="Optional Hugging Face dataset repo namespace/name"
    )
    parser.add_argument(
        "--hf-private", action="store_true", help="Create a private dataset repository"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConversionConfig(
        input_path=args.input_h5,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        history=args.history,
        break_indices=tuple(args.break_index),
        split_seed=args.split_seed,
        workers=args.workers,
    )
    output = convert_dataset(config)
    print(f"Validated cache: {output}")
    if args.hf_repo:
        url = publish_cache(output, args.hf_repo, private=args.hf_private)
        print(f"Hugging Face dataset: {url}")


if __name__ == "__main__":
    main()
