
"""Train a byte-level BPE tokenizer from scratch."""

import argparse
import hashlib
import json
import resource
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tokenizers

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)

from ggpt.utils.logger import get_logger, setup_logging


ROOT = Path(__file__).resolve().parents[2]

SAMPLE_DIR = ROOT / "data/interim/tokenizer/bpe_sample"
SAMPLE_PATH = SAMPLE_DIR / "sample.parquet"

LOG_FILE = ROOT / "runs/logs/tokenizer/train_bpe.log"

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
]

logger = get_logger("scripts.tokenizer.train_bpe")


def iter_selected_texts(
    sample_path: Path,
    selected_indices: np.ndarray,
    batch_size: int = 1024,
):
    cursor = 0
    offset = 0

    with pq.ParquetFile(sample_path) as parquet:
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["text"],
        ):
            end = offset + batch.num_rows

            next_cursor = int(
                np.searchsorted(selected_indices, end)
            )

            if next_cursor > cursor:
                positions = (
                    selected_indices[cursor:next_cursor] - offset
                )

                texts = batch.column(0).take(
                    pa.array(positions, type=pa.int64())
                )

                if texts.null_count:
                    raise RuntimeError(
                        "Null text in BPE training sample"
                    )

                cursor = next_cursor

                yield texts.to_pylist()

            offset = end

    if cursor != len(selected_indices):
        raise RuntimeError("Incomplete training sample")


def check_roundtrip(tokenizer: Tokenizer) -> None:
    examples = [
        "გამარჯობა, საქართველო!",
        "მე მიყვარს საქართველო.",
        "ქართული ტექსტი\nმეორე ხაზი",
        "  გამარჯობა!  ",
        "Hello, world! Привет 🌍",
        "",
    ]

    for text in examples:
        encoded = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        decoded = tokenizer.decode(
            encoded.ids,
            skip_special_tokens=False,
        )

        if decoded != text:
            raise RuntimeError(
                f"Round-trip failed: {text!r}"
            )

    logger.info(
        "Round-trip tests passed: %d examples",
        len(examples),
    )


def train_bpe(
    n_docs: int,
    seed: int,
    vocab_size: int,
    output_dir: Path,
) -> None:
    if n_docs <= 0 or vocab_size <= 259:
        raise ValueError("Invalid training parameters")

    sample_documents = pq.read_metadata(SAMPLE_PATH).num_rows

    sample_manifest = json.loads(
        (SAMPLE_DIR / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if sample_documents != sample_manifest["sample_documents"]:
        raise RuntimeError("Sample metadata mismatch")

    if n_docs > sample_documents:
        raise ValueError("Requested more documents than available")

    if output_dir.exists():
        raise FileExistsError(output_dir)

    temporary_dir = output_dir.with_name(
        output_dir.name + ".partial"
    )

    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)

    logger.info(
        "Starting BPE: n_docs=%d, seed=%d, vocab_size=%d",
        n_docs,
        seed,
        vocab_size,
    )

    logger.info("Source sample: %s", SAMPLE_PATH)
    logger.info("Output directory: %s", output_dir)

    rng = np.random.default_rng(seed)

    selected_indices = np.sort(
        rng.choice(
            sample_documents,
            size=n_docs,
            replace=False,
        )
    )

    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    logger.info(
        "Configuration: byte-level BPE, min_frequency=2, "
        "no normalization, add_prefix_space=False"
    )

    logger.info("Special tokens: %s", SPECIAL_TOKENS)

    print(
        f"Training BPE: {n_docs:,} documents, "
        f"vocab_size={vocab_size:,}",
        flush=True,
    )

    started = perf_counter()

    tokenizer.train_from_iterator(
        iter_selected_texts(
            SAMPLE_PATH,
            selected_indices,
        ),
        trainer=trainer,
        length=n_docs,
    )

    elapsed = perf_counter() - started

    peak_rss_mb = (
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss / 1024
    )

    actual_vocab_size = tokenizer.get_vocab_size()

    if actual_vocab_size > vocab_size:
        raise RuntimeError("Actual vocabulary exceeds requested size")

    special_token_ids = {
        token: tokenizer.token_to_id(token)
        for token in SPECIAL_TOKENS
    }

    if any(
        token_id is None
        for token_id in special_token_ids.values()
    ):
        raise RuntimeError("Missing special tokens")

    logger.info(
        "Training finished: seconds=%.2f, peak_ram=%.1f MiB, "
        "vocab_size=%d",
        elapsed,
        peak_rss_mb,
        actual_vocab_size,
    )

    logger.info("Special token IDs: %s", special_token_ids)

    check_roundtrip(tokenizer)

    temporary_dir.mkdir(parents=True)

    try:
        tokenizer_path = temporary_dir / "tokenizer.json"

        tokenizer.save(str(tokenizer_path))

        loaded_tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        check_roundtrip(loaded_tokenizer)

        manifest = {
            "algorithm": "byte_level_bpe",
            "library": "tokenizers",
            "library_version": tokenizers.__version__,
            "training_sample": str(
                SAMPLE_PATH.relative_to(ROOT)
            ),
            "training_documents": n_docs,
            "sample_selection_seed": seed,
            "selected_indices_sha256": hashlib.sha256(
                selected_indices.tobytes()
            ).hexdigest(),
            "requested_vocab_size": vocab_size,
            "actual_vocab_size": actual_vocab_size,
            "min_frequency": 2,
            "normalization": None,
            "add_prefix_space": False,
            "special_token_ids": special_token_ids,
            "training_seconds": round(elapsed, 3),
            "peak_process_rss_mb": round(peak_rss_mb, 1),
        }

        (temporary_dir / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_dir.rename(output_dir)

    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    logger.info("Tokenizer successfully saved: %s", output_dir)

    print("BPE training: OK")
    print(f"Training time: {elapsed:.1f} s")
    print(f"Peak RAM: {peak_rss_mb:.1f} MiB")
    print(f"Vocabulary size: {actual_vocab_size:,}")
    print(f"Special token IDs: {special_token_ids}")
    print(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-docs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizer/bpe_pilot_8192"),
    )

    args = parser.parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )

    train_bpe(
        n_docs=args.n_docs,
        seed=args.seed,
        vocab_size=args.vocab_size,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    setup_logging(
        log_file=LOG_FILE,
        console=False,
    )

    try:
        main()
    except Exception:
        logger.exception("BPE training failed")
        raise
