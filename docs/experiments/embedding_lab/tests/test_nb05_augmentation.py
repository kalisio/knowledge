from __future__ import annotations

import json
import sys
from pathlib import Path

_HELPER = Path(__file__).resolve().parents[2] / "experiment_helper"
sys.path.insert(0, str(_HELPER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb05_augment_helpers import LLMTranslator  # noqa: E402
from nb05_glossary import (  # noqa: E402
    PROJECT_GLOSSARY,
    augment_queries_with_project_glossary,
    augment_query_with_project_glossary,
    match_project_glossary,
)


def _make_translator(tmp_path: Path, model: str = "qwen2.5:7b") -> LLMTranslator:
    return LLMTranslator(
        cache_path=tmp_path / "cache.json",
        host="http://invalid.local:1",
        model=model,
    )


def test_project_glossary_has_nb05_entries():
    assert len(PROJECT_GLOSSARY) == 39


def test_augment_preserves_original_query_and_appends_project_anchor():
    query = "Où est implémenté le service d'envoi d'emails ?"

    augmented, matches = augment_query_with_project_glossary(query)

    assert augmented == f"{query} (mailer service mailer email)"
    assert [match.expansion for match in matches] == ["mailer service", "mailer email"]


def test_no_glossary_match_returns_query_unchanged():
    query = "Comment ajouter une nouvelle couche à la carte ?"

    augmented, matches = augment_query_with_project_glossary(query)

    assert augmented == query
    assert matches == []


def test_glossary_matching_is_case_insensitive_and_handles_curly_apostrophe():
    matches = match_project_glossary("Journal d’Événement dans Crisis")

    assert [match.expansion for match in matches] == ["EventLog event log"]


def test_batch_glossary_augmentation():
    queries = [
        "L’intégration carte dans Crisis est définie où ?",
        "Question sans terme spécialisé",
    ]

    assert augment_queries_with_project_glossary(queries) == [
        "L’intégration carte dans Crisis est définie où ? (MapActivity map)",
        "Question sans terme spécialisé",
    ]


def test_llm_cache_hits_avoid_network(tmp_path, monkeypatch):
    translator = _make_translator(tmp_path)
    text = "Bonjour le monde"
    key = translator._cache_key(text)
    translator.cache_path.write_text(json.dumps({key: "Hello world"}))
    monkeypatch.setattr(translator, "_is_reachable", lambda: False)

    assert translator.translate_all([text]) == ["Hello world"]


def test_unreachable_llm_host_returns_none_for_cache_miss(tmp_path, monkeypatch):
    translator = _make_translator(tmp_path)
    monkeypatch.setattr(translator, "_is_reachable", lambda: False)

    assert translator.translate_all(["Quelque chose qui n'est pas dans le cache"]) == [None]


def test_llm_cache_key_includes_model_so_swapping_invalidates(tmp_path):
    t1 = _make_translator(tmp_path, model="qwen2.5:7b")
    t2 = _make_translator(tmp_path, model="qwen2.5:14b")

    assert t1._cache_key("hello") != t2._cache_key("hello")


def test_llm_translate_all_preserves_order_from_cache(tmp_path, monkeypatch):
    translator = _make_translator(tmp_path)
    inputs = ["one", "two", "three"]
    cache_data = {translator._cache_key(t): t.upper() for t in inputs}
    translator.cache_path.write_text(json.dumps(cache_data))
    monkeypatch.setattr(translator, "_is_reachable", lambda: False)

    assert translator.translate_all(inputs) == ["ONE", "TWO", "THREE"]


def test_partial_llm_cache_with_unreachable_host_fills_misses_with_none(tmp_path, monkeypatch):
    translator = _make_translator(tmp_path)
    cached_text = "déjà en cache"
    miss_text = "absent du cache"
    translator.cache_path.write_text(
        json.dumps({translator._cache_key(cached_text): "already cached"})
    )
    monkeypatch.setattr(translator, "_is_reachable", lambda: False)

    assert translator.translate_all([cached_text, miss_text]) == ["already cached", None]


def test_successful_llm_chat_calls_are_persisted_to_cache(tmp_path, monkeypatch):
    translator = _make_translator(tmp_path)
    monkeypatch.setattr(translator, "_is_reachable", lambda: True)

    calls: list[str] = []

    def fake_chat(text: str, *, timeout: float) -> str:
        calls.append(text)
        return f"<{text.upper()}>"

    monkeypatch.setattr(translator, "_chat", fake_chat)

    out = translator.translate_all(["alpha", "beta"])
    assert out == ["<ALPHA>", "<BETA>"]
    assert calls == ["alpha", "beta"]

    persisted = json.loads(translator.cache_path.read_text())
    assert persisted[translator._cache_key("alpha")] == "<ALPHA>"
    assert persisted[translator._cache_key("beta")] == "<BETA>"
