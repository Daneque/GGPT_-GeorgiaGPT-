"""Benchmark v3 collection against an exported GGPT model (never a train checkpoint)."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from ggpt.inference.generate import GenerationConfig
from ggpt.inference.model_runner import GGPTTextRunner
from ggpt.benchmark.v3_dataset import V3Suite
from ggpt.benchmark.v3_scoring import score_case
from ggpt.inference.artifact import file_sha256, load_model_artifact
from ggpt.utils.logger import get_logger

logger = get_logger("benchmark.v3")

VARIANTS = {
    "greedy": {"do_sample": False},
    "topk_t08": {"do_sample": True, "temperature": 0.8, "top_k": 40},
    "topp_t08": {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
    "conservative": {"do_sample": True, "temperature": 0.65, "top_p": 0.9},
    "diverse": {"do_sample": True, "temperature": 1.0, "top_p": 0.95},
}


def _append(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _existing(path: Path, key: str) -> set:
    found = set()
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                item = json.loads(line)
                identity = tuple(item[key]) if isinstance(item[key], list) else item[key]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid existing result: {path}:{number}") from exc
            if identity in found:
                raise ValueError(f"Duplicate result in {path}: {identity}")
            found.add(identity)
    return found


def run_v3(*, model_dir: str | Path, suite: V3Suite, output: str | Path,
           device: str, variants: tuple[str, ...] = tuple(VARIANTS),
           categories: tuple[str, ...] = ("continuation", "facts", "context"),
           limit: int | None = None, only_score: bool = False,
           resume: bool = False) -> dict:
    if any(variant not in VARIANTS for variant in variants) or len(set(variants)) != len(variants):
        raise ValueError("Unknown or duplicate variants")
    if any(c not in ("continuation", "facts", "context") for c in categories):
        raise ValueError("Unknown category")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    model_dir = Path(model_dir).resolve()
    output = Path(output).resolve()
    manifest_path = model_dir / "manifest.json"
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = []
    counts = dict.fromkeys(categories, 0)
    for case in suite.cases:
        if case.category in counts and (limit is None or counts[case.category] < limit):
            selected.append(case)
            counts[case.category] += 1
    info = {
        "format": "ggpt-benchmark-v3",
        "model_artifact": str(model_dir),
        "model_manifest_sha256": file_sha256(manifest_path),
        "suite_sha256": suite.sha256,
        "suite_name": suite.name,
        "device": device,
        "categories": categories,
        "variants": variants,
        "only_score": only_score,
        "limit_per_category": limit,
        "selected_ids": [c.id for c in selected],
        "scoring": "raw_logits_teacher_forcing",
        "judge": None,
    }
    if resume:
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        # JSON arrays read back as lists, so serialize the expected metadata first.
        if existing != json.loads(json.dumps(info)):
            raise ValueError("Model/dataset/variants differ from existing benchmark")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "manifest.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    scores_path = output / "scores.jsonl"
    generations_path = output / "generations.jsonl"
    scored = _existing(scores_path, "id")
    generated = _existing(generations_path, "key")
    expected_scores = {case.id for case in selected if case.category != "continuation"}
    expected_generated = {(case.id, variant) for case in selected for variant in variants}
    if scored - expected_scores or generated - expected_generated:
        raise ValueError("Existing output contains rows not belonging to the requested run")
    logger.info(
        "%s benchmark=%s model=%s device=%s questions=%d "
        "scoring=%d/%d generations=%d/%d output=%s",
        "resuming" if resume else "starting",
        suite.name,
        model_dir,
        device,
        len(selected),
        len(scored),
        len(expected_scores),
        len(generated),
        0 if only_score else len(expected_generated),
        output,
    )
    logger.info("loading exported model")
    model, tokenizer, _ = load_model_artifact(model_dir, device=device)
    logger.info("model loaded; starting evaluation")
    adapter = GGPTTextRunner(model=model, tokenizer=tokenizer, device=device)
    for case_idx, case in enumerate(selected):
        if case.category != "continuation" and case.id not in scored:
            started = perf_counter()
            measurement = score_case(model, tokenizer, case, device=device)
            _append(scores_path, {
                "id": case.id, "category": case.category, "format": case.format,
                "prompt": case.prompt, "control_prompt": case.control_prompt,
                "metadata": case.metadata, "measurement": measurement,
            })
            scored.add(case.id)
            logger.info(
                "scored %d/%d question=%d/%d id=%s category=%s elapsed=%.2fs",
                len(scored), len(expected_scores), case_idx + 1, len(selected),
                case.id, case.category, perf_counter() - started,
            )
        if only_score:
            continue
        for variant_idx, variant in enumerate(variants):
            key = (case.id, variant)
            if key in generated:
                continue
            config = GenerationConfig(
                max_new_tokens=96 if case.category == "continuation" else 24,
                seed=42 + case_idx * 10 + variant_idx,
                **VARIANTS[variant],
            )
            started = perf_counter()
            text = adapter.generate(case.prompt, config)
            _append(generations_path, {
                "key": list(key), "id": case.id, "category": case.category,
                "format": case.format, "prompt": case.prompt, "response": text,
                "variant": variant, "generation_settings": asdict(config),
                "reference": case.reference,
                "metadata": case.metadata,
            })
            generated.add(key)
            logger.info(
                "generated %d/%d question=%d/%d id=%s category=%s "
                "variant=%s elapsed=%.2fs",
                len(generated), len(expected_generated), case_idx + 1,
                len(selected), case.id, case.category, variant,
                perf_counter() - started,
            )
    summary = {
        "questions": len(selected),
        "scored": len(scored),
        "scored_expected": len(expected_scores),
        "generated": len(generated),
        "generated_expected": 0 if only_score else len(expected_generated),
        "automatic_georgian_semantic_judging": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "finished questions=%d scoring=%d/%d generations=%d/%d output=%s",
        len(selected), len(scored), len(expected_scores),
        len(generated), 0 if only_score else len(expected_generated), output,
    )
    return summary
