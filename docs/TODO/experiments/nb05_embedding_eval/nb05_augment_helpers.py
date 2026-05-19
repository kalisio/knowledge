"""LLM-based query translation for §13.2 of nb05.

Wraps a local Ollama instance with on-disk caching so the experiment is
reproducible and offline-friendly. Pairs with the deterministic project
glossary in :mod:`nb05_glossary` to produce the four FR query variants
compared in §13.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence


TRANSLATION_SYSTEM = (
    "You are a French-to-English translator for software engineering questions "
    "about a JavaScript/Vue codebase. Output ONLY the natural English translation "
    "of the question — no commentary, no quotes, no preamble."
)

DEFAULT_HOST = "http://192.168.1.109:11434"
DEFAULT_MODEL = "qwen2.5:7b"


class LLMTranslator:
    """FR→EN translator backed by a local Ollama server with on-disk cache.

    The cache key includes the model name, so swapping models invalidates
    stale translations automatically. Failures degrade gracefully — when
    the host is unreachable the translator returns cached entries where
    available and ``None`` for misses, never raising.
    """

    def __init__(
        self,
        cache_path: Path,
        *,
        host: str | None = None,
        model: str | None = None,
        progress_every: int = 20,
        cold_start_timeout: float = 120.0,
        warm_timeout: float = 30.0,
        reachability_timeout: float = 5.0,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.host = host if host is not None else os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        self.model = model if model is not None else os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self.progress_every = progress_every
        self.cold_start_timeout = cold_start_timeout
        self.warm_timeout = warm_timeout
        self.reachability_timeout = reachability_timeout

    # --- cache ---------------------------------------------------------

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self, cache: dict[str, str]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{text}".encode("utf-8")).hexdigest()[:16]

    # --- network -------------------------------------------------------

    def _is_reachable(self) -> bool:
        try:
            urllib.request.urlopen(
                f"{self.host}/api/tags", timeout=self.reachability_timeout
            ).read(64)
            return True
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def _chat(self, text: str, *, timeout: float) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 200},
            "messages": [
                {"role": "system", "content": TRANSLATION_SYSTEM},
                {"role": "user", "content": text},
            ],
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["message"]["content"].strip()

    # --- main entry ----------------------------------------------------

    def translate_all(self, fr_texts: Sequence[str]) -> list[str | None]:
        """Translate each FR text to EN; cache hits on disk; ``None`` on failure."""
        cache = self._load_cache()
        out: list[str | None] = []

        reachable = self._is_reachable()
        if not reachable:
            print(f"[llm] Ollama at {self.host} unreachable; using cache only")

        n_calls = 0
        n_errors = 0
        for t in fr_texts:
            k = self._cache_key(t)
            if k in cache:
                out.append(cache[k])
                continue
            if not reachable:
                out.append(None)
                continue
            try:
                timeout = self.cold_start_timeout if n_calls == 0 else self.warm_timeout
                translation = self._chat(t, timeout=timeout)
                cache[k] = translation
                out.append(translation)
                n_calls += 1
                if n_calls % self.progress_every == 0:
                    self._save_cache(cache)
                    print(f"[llm] {n_calls} translations done")
            except Exception as exc:
                n_errors += 1
                if n_errors <= 3:
                    print(f"[llm] error on {t[:50]!r}: {type(exc).__name__}: {exc}")
                out.append(None)

        if n_calls or n_errors:
            self._save_cache(cache)
            cached = sum(1 for o in out if o is not None) - n_calls
            print(
                f"[llm] {n_calls} new translations  |  {cached} cache hits  "
                f"|  {n_errors} errors  |  model={self.model}"
            )
        else:
            print(f"[llm] all {len(fr_texts)} cache hits (model={self.model})")
        return out
