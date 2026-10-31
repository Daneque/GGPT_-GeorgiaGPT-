"""Progress logging for Benchmark v3, including resumed runs."""

import io
import json
import logging
from pathlib import Path
import sys

from ggpt.benchmark import v3_runner
from ggpt.benchmark.v3_dataset import V3Case, V3Suite


def _setup_toy_run(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(json.dumps({"format": "test"}))

    class FakeTextRunner:
        def __init__(self, model, tokenizer, device):
            pass

        def generate(self, prompt, config):
            return "generated text"

    monkeypatch.setattr(v3_runner, "load_model_artifact", lambda *args, **kwargs: (None, None, {}))
    monkeypatch.setattr(v3_runner, "GGPTTextRunner", FakeTextRunner)
    monkeypatch.setattr(v3_runner, "score_case", lambda *args, **kwargs: {"ok": True})
    suite = V3Suite(
        "toy",
        (
            V3Case("continue-01", "continuation", "free", "X", None, (), None, {}),
            V3Case("fact-01", "facts", "cloze", "X", " A", (" A", " B"), None, {}),
        ),
        "toy-dataset-sha",
    )
    return model_dir, suite


def test_progress_messages_and_resume(tmp_path, monkeypatch):
    model_dir, suite = _setup_toy_run(tmp_path, monkeypatch)
    output = tmp_path / "results"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = v3_runner.logger
    original_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        params = dict(
            model_dir=model_dir, suite=suite, output=output, device="cpu",
            variants=("greedy", "topp_t08"),
        )
        result = v3_runner.run_v3(**params)
        text = stream.getvalue()
        assert result["generated"] == 4
        assert "generated 1/4 question=1/2 id=continue-01" in text
        assert "variant=greedy" in text
        assert "generated 4/4 question=2/2 id=fact-01" in text
        assert "variant=topp_t08" in text
        assert "scored 1/1 question=2/2 id=fact-01" in text
        assert "finished questions=2 scoring=1/1 generations=4/4" in text

        stream.seek(0)
        stream.truncate(0)
        result_resume = v3_runner.run_v3(**params, resume=True)
        assert result_resume == result
        resumed_text = stream.getvalue()
        assert "resuming benchmark=toy" in resumed_text
        assert "generations=4/4" in resumed_text
        assert "generated 1/4" not in resumed_text
        assert "scored 1/1 question=" not in resumed_text
        assert len((output / "generations.jsonl").read_text().splitlines()) == 4
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        handler.close()


def test_cli_uses_common_logger_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts import benchmark_v3

    model_dir, suite = _setup_toy_run(tmp_path, monkeypatch)
    output = tmp_path / "result"
    monkeypatch.setattr(benchmark_v3, "load_v3_dataset", lambda path: suite)
    monkeypatch.setattr(
        sys, "argv", [
            "benchmark_v3.py", "--model", str(model_dir),
            "--output", str(output), "--device", "cpu", "--variants", "greedy",
        ],
    )
    try:
        benchmark_v3.main()
        log_file = tmp_path / "result.log"
        contents = log_file.read_text(encoding="utf-8")
        assert "ggpt.benchmark.v3" in contents
        assert "generated 1/2 question=1/2 id=continue-01" in contents
        assert "generated 2/2 question=2/2 id=fact-01" in contents
        assert "scored 1/1 question=2/2 id=fact-01" in contents
        assert "finished questions=2" in contents
        assert (output / "generations.jsonl").is_file()
        assert (output / "scores.jsonl").is_file()

        sys.argv.append("--resume")
        benchmark_v3.main()
        contents = log_file.read_text(encoding="utf-8")
        assert "resuming benchmark=toy" in contents
        assert contents.count("generated 1/2 question=") == 1
    finally:
        parent_logger = logging.getLogger("ggpt")
        for handler in parent_logger.handlers[:]:
            parent_logger.removeHandler(handler)
            handler.close()
