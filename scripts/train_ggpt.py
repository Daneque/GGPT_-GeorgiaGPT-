"""Train GGPT from scratch or resume an identical run from latest checkpoint."""

import argparse
from pathlib import Path

from ggpt.utils.logger import setup_logging
from ggpt.model import ModelConfig
from ggpt.training.config import OptimizerConfig
from ggpt.training.trainer import TrainerConfig, run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default="runs/ggpt")
    p.add_argument("--train-bin", default="data/tokenized/kat_Geor/train.bin")
    p.add_argument("--val-bin", default="data/tokenized/kat_Geor/val.bin")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/bpe_100k_16384/tokenizer.json")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, default=None,
                   help="Absolute update count; checkpoint and pause before max_steps")
    p.add_argument("--max-steps", type=int, default=60_000)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--train-windows", type=int, default=10_000)
    p.add_argument("--validation-windows", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--generate-every", type=int, default=1000,
                   help="0 disables generation")
    p.add_argument("--generation-max-tokens", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run-name must be a simple directory name")
    cfg = TrainerConfig(
        max_steps=args.max_steps, micro_batch_size=args.micro_batch_size,
        accumulation_steps=args.accumulation_steps, train_windows=args.train_windows,
        validation_windows=args.validation_windows, num_workers=args.workers,
        seed=args.seed, precision=args.precision, warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio, log_every=args.log_every,
        eval_every=args.eval_every, checkpoint_every=args.checkpoint_every,
        generate_every=args.generate_every,
        generation_max_tokens=args.generation_max_tokens,
    )
    setup_logging(
        log_file=Path("runs/logs/training") / f"{args.run_name}.log",
        console=True,
    )
    progress = run_training(
        run_dir=Path(args.runs_dir) / args.run_name,
        train_path=args.train_bin, val_path=args.val_bin,
        tokenizer_path=args.tokenizer, model_config=ModelConfig(),
        optimizer_config=OptimizerConfig(learning_rate=args.lr,
                                         weight_decay=args.weight_decay),
        config=cfg, resume=args.resume, stop_after=args.stop_after,
        device=args.device,
    )
    print("Training state:", progress, flush=True)


if __name__ == "__main__":
    main()
