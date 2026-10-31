"""Export a training checkpoint as an inference-only model directory."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggpt.inference.artifact import export_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="latest")
    args = parser.parse_args()
    destination = export_model(args.run, args.output, checkpoint=args.checkpoint)
    print(f"Inference-only GGPT model: {destination}")


if __name__ == "__main__":
    main()
