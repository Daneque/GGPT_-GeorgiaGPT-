"""Unified GGPT training CLI: start / resume / extend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from ggpt.training.experiment import run_extend, run_resume, run_start
from ggpt.training.experiment_config import load_experiment_config
from ggpt.utils.logger import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="New experiment from random weights")
    start.add_argument("--config", required=True, type=Path)
    start.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Temporary stop at an absolute global step",
    )

    resume = commands.add_parser("resume", help="Restart the SAME saved run")
    resume.add_argument("--run", required=True, type=Path)
    resume.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    resume.add_argument("--stop-after", type=int, default=None)

    extend = commands.add_parser("extend", help="New stage with ADDITIONAL steps")
    extend.add_argument("--from", dest="source", required=True, type=Path)
    extend.add_argument("--config", required=True, type=Path)
    extend.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Temporary stop at an absolute global step",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command in {"start", "extend"}:
        config_file = args.config.resolve()
        config = load_experiment_config(
            config_file,
            mode=args.command,
            project_root=ROOT,
        )
        run_name = config.name
    else:
        run_name = args.run.resolve().name

    setup_logging(log_file=ROOT / "runs/logs/training" / f"{run_name}.log", console=True)

    if args.command == "start":
        progress = run_start(config, config_path=config_file, stop_after=args.stop_after)
    elif args.command == "resume":
        progress = run_resume(args.run, device=args.device, stop_after=args.stop_after)
    else:
        progress = run_extend(
            args.source,
            config,
            config_path=config_file,
            stop_after=args.stop_after,
        )

    print("Training state:", progress, flush=True)


if __name__ == "__main__":
    main()
