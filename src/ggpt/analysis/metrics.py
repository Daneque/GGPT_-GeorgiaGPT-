
"""Per-document text metrics for corpus analysis."""

import hashlib

import pandas as pd
import regex


class DocumentProfiler:
    """Compute descriptive metrics for one text without modifying it."""

    def __init__(self) -> None:
        self._words = regex.compile(r"\p{L}[\p{L}\p{M}]*")
        self._letters = regex.compile(r"\p{L}")

        self._georgian = regex.compile(
            r"(?V1)[\p{Script=Georgian}&&\p{L}]"
        )
        self._latin = regex.compile(
            r"(?V1)[\p{Script=Latin}&&\p{L}]"
        )
        self._cyrillic = regex.compile(
            r"(?V1)[\p{Script=Cyrillic}&&\p{L}]"
        )

        self._urls = regex.compile(
            r"https?://|www\.",
            regex.IGNORECASE,
        )
        self._html = regex.compile(
            r"</?[a-z][a-z0-9-]*(?:\s[^<>]*)?>",
            regex.IGNORECASE,
        )

    def profile(self, text: str | None) -> dict:
        """Return numeric diagnostics of a single document."""
        is_missing = not isinstance(text, str)
        content = "" if is_missing else text
        encoded = content.encode("utf-8")

        n_letters = sum(
            1 for _ in self._letters.finditer(content)
        )
        n_georgian = sum(
            1 for _ in self._georgian.finditer(content)
        )
        n_latin = sum(
            1 for _ in self._latin.finditer(content)
        )
        n_cyrillic = sum(
            1 for _ in self._cyrillic.finditer(content)
        )

        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        n_lines = len(lines)
        n_repeated_lines = n_lines - len(set(lines))

        return {
            "is_missing_text": is_missing,
            "is_empty_text": not content.strip(),
            "n_chars": len(content),
            "n_bytes": len(encoded),
            "n_words": sum(
                1 for _ in self._words.finditer(content)
            ),
            "n_lines": n_lines,
            "n_letters": n_letters,
            "n_georgian_letters": n_georgian,
            "n_latin_letters": n_latin,
            "n_cyrillic_letters": n_cyrillic,
            "georgian_share": (
                n_georgian / n_letters if n_letters else 0.0
            ),
            "latin_share": (
                n_latin / n_letters if n_letters else 0.0
            ),
            "cyrillic_share": (
                n_cyrillic / n_letters if n_letters else 0.0
            ),
            "repeated_line_share": (
                n_repeated_lines / n_lines if n_lines else 0.0
            ),
            "n_url_mentions": sum(
                1 for _ in self._urls.finditer(content)
            ),
            "has_html_like": bool(
                self._html.search(content)
            ),
            "n_replacement_chars": content.count("\ufffd"),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }


def profile_batch(
    texts: pd.Series,
    profiler: DocumentProfiler,
) -> pd.DataFrame:
    """Profile one batch of documents, preserving its index."""
    return pd.DataFrame.from_records(
        (profiler.profile(text) for text in texts),
        index=texts.index,
    )
