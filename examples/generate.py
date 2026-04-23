#!/usr/bin/env python3
"""Dataset generation entry point for M-127 (thyroidus_cine_nodule_track).

Usage:
    python examples/generate.py
    python examples/generate.py --num-samples 5
    python examples/generate.py --output data/my_output
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate M-127 (thyroid US cine nodule-tracking) dataset"
    )
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max clips to process (default = all 192).")
    parser.add_argument("--output", type=str, default="data/questions")
    args = parser.parse_args()

    print("Generating M-127 (thyroidus_cine_nodule_track) dataset...")
    config = TaskConfig(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
    )
    pipeline = TaskPipeline(config)
    pipeline.run()
    print("Done.")


if __name__ == "__main__":
    main()
