"""Benchmark v3 question contract, distinct from the v2 dataset format."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

CATEGORIES = ("continuation", "facts", "context")
FORMATS = {"continuation": {"free"}, "facts": {"cloze"}, "context": {"cloze", "qa"}}


@dataclass(frozen=True)
class V3Case:
    id: str
    category: str
    format: str
    prompt: str
    reference: str | None
    candidates: tuple[str, ...]
    control_prompt: str | None
    metadata: dict


@dataclass(frozen=True)
class V3Suite:
    name: str
    cases: tuple[V3Case, ...]
    sha256: str


def load_v3_dataset(path: str | Path, *, require_full: bool = True) -> V3Suite:
    data_bytes = Path(path).read_bytes()
    data = json.loads(data_bytes)
    if data.get("version") != 3 or not isinstance(data.get("questions"), list):
        raise ValueError("Expected version=3 and questions array")
    seen: set[str] = set()
    cases: list[V3Case] = []
    for n, row in enumerate(data["questions"]):
        if not isinstance(row, dict):
            raise ValueError(f"Question {n} is not an object")
        ident, category, kind, prompt = (
            row.get("id"), row.get("category"), row.get("format"), row.get("prompt")
        )
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError(f"Invalid or duplicate ID: {ident}")
        seen.add(ident)
        if category not in CATEGORIES or kind not in FORMATS[category]:
            raise ValueError(f"Invalid category or format for {ident}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Empty prompt for {ident}")
        reference = row.get("reference")
        alternatives = row.get("candidates", [])
        control = row.get("control_prompt")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid metadata: {ident}")
        if category == "continuation":
            if reference is not None or alternatives or control is not None:
                raise ValueError(f"Continuation may not have scoring fields: {ident}")
        else:
            if not isinstance(reference, str) or not reference.startswith(" "):
                raise ValueError(f"Reference must be a nonempty leading-space continuation: {ident}")
            if not isinstance(alternatives, list) or len(alternatives) != 4 or any(
                not isinstance(c, str) or not c.startswith(" ") or not c.strip()
                for c in alternatives
            ):
                raise ValueError(f"Exactly four leading-space alternatives required: {ident}")
            if len(set(alternatives)) != 4 or reference not in alternatives:
                raise ValueError(f"Reference must occur exactly once among alternatives: {ident}")
            if category == "context":
                if not isinstance(control, str) or not control.strip() or control == prompt:
                    raise ValueError(f"Missing or identical context control: {ident}")
            elif control is not None:
                raise ValueError(f"Fact case must not include control_prompt: {ident}")
        cases.append(V3Case(ident, category, kind, prompt, reference,
                            tuple(alternatives), control, metadata))
    if require_full:
        counts = {category: sum(case.category == category for case in cases)
                  for category in CATEGORIES}
        if counts != dict.fromkeys(CATEGORIES, 18):
            raise ValueError(f"Expected 18 cases per category, got {counts}")
        if sum(c.format == "cloze" for c in cases if c.category == "context") != 12:
            raise ValueError("Expected 12 cloze and 6 QA context cases")
    return V3Suite(data["name"], tuple(cases), hashlib.sha256(data_bytes).hexdigest())
