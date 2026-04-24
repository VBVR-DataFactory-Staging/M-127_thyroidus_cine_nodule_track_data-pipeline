"""Local example generator."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.config import TaskConfig
from src.pipeline.pipeline import TaskPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--output", type=str, default="data/questions")
    args = p.parse_args()

    cfg = TaskConfig(num_samples=args.num_samples, output_dir=Path(args.output))
    pipe = TaskPipeline(cfg)
    print(f"[M-127_thyroidus_cine_nodule_track] Generating {args.num_samples} sample(s) -> {args.output}")
    samples = pipe.run()
    print(f"[M-127_thyroidus_cine_nodule_track] Wrote {len(samples)} samples")


if __name__ == "__main__":
    main()
