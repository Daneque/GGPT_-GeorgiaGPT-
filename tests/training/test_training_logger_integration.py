import logging
from pathlib import Path

from ggpt.utils.logger import setup_logging
from ggpt.training.logging import TrainingLogger


class DummyWriter:
    def add_scalar(self, *args):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def test_existing_project_logger_receives_training_events(tmp_path):
    path = tmp_path / "training.log"
    setup_logging(log_file=path, console=False)
    try:
        with TrainingLogger(tmp_path / "run", writer=DummyWriter()) as logger:
            logger.event("start", 0, device="cpu")
            logger.scalars("train", 1, loss=2.5)
            logger.event("checkpoint", 1, path="latest.pt")
        text = path.read_text(encoding="utf-8")
        assert "ggpt.training" in text
        assert "start step=0" in text
        assert "train step=1" in text
        assert "checkpoint step=1" in text
        assert (tmp_path / "run" / "metrics.jsonl").exists()
    finally:
        parent = logging.getLogger("ggpt")
        for handler in parent.handlers[:]:
            parent.removeHandler(handler)
            handler.close()
