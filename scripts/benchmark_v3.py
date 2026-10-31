"""Run Benchmark v3 against an exported, inference-only GGPT model."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ggpt.benchmark.v3_dataset import CATEGORIES, load_v3_dataset
from ggpt.benchmark.v3_runner import VARIANTS, run_v3
from ggpt.utils.logger import get_logger, setup_logging

logger = get_logger("benchmark.v3.cli")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Exported inference model directory")
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/georgian/language_benchmark_v3.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--limit", type=int, help="Only first N cases per category (for smoke testing)")
    parser.add_argument("--only-score", action="store_true", help="Skip all text generation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Keep the log next to the output directory: configuring file logging must
    # not create the result directory before run_v3 checks its existence.
    output_path = args.output.resolve()
    log_file = output_path.parent / f"{output_path.name}.log"
    setup_logging(log_file=log_file, console=True)
    logger.info("benchmark log=%s", log_file)
    try:
        suite = load_v3_dataset(args.dataset)
        outcome = run_v3(
            model_dir=args.model, suite=suite, output=args.output, device=args.device,
            variants=tuple(args.variants), categories=tuple(args.categories),
            limit=args.limit, only_score=args.only_score, resume=args.resume,
        )
    except Exception:
        logger.exception("benchmark failed; output=%s", output_path)
        raise
    logger.info("summary=%s", outcome)


if __name__ == "__main__":
    main()
