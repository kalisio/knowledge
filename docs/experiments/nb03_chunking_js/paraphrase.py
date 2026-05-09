"""Shared tokenization helpers for Vue component-name paraphrasing.

Keeps the camelCase splitter acronym-aware so ``KHTTPClient`` becomes
``"http client"`` rather than ``"k h t t p client"``. The leading ``K``
prefix (Kalisio convention) is stripped when present.
"""

from __future__ import annotations

import re
from pathlib import Path

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def camel_phrase(name: str) -> str:
    """``KZoomControl`` -> ``"zoom control"``, ``KHTTPClient`` -> ``"http client"``.

    Consecutive capitals are kept as one acronym token; 1-char shards
    (usually the leading K) are dropped.
    """
    words = [w.lower() for w in _CAMEL_RE.split(name) if w]
    words = [w for w in words if len(w) > 1]
    return " ".join(words)


def component_phrase(rel_path: str) -> str:
    """Derive a human-readable phrase from a ``.vue`` filename."""
    stem = Path(rel_path).stem
    name = stem[1:] if stem.startswith("K") and len(stem) > 1 else stem
    return camel_phrase(name)
