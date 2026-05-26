"""JSON chunking with category-aware breadcrumbs.

Module is named ``json_chunking.py`` (not ``json.py``) so it never
shadows the standard library in sub-package imports.

Strategy = category-aware key split + breadcrumb. The splitter
dispatches on the payload's semantic role, determined from the file
path / name:

    schemas_validation  → one chunk per top-level ``properties[name]``
                          plus one ``<schema-meta>`` chunk for ``$id`` /
                          ``title`` / ``required``.
    i18n_translations   → one chunk per top-level section; scalar
                          labels coalesced into a single
                          ``<flat-labels>`` chunk; oversized sections
                          further split with RecursiveJsonSplitter.
    package_tooling     → cherry-pick ``name`` / ``description`` /
                          ``scripts`` / dependency *names* only, as
                          one chunk per file.
    docs_meta /         → whole-file chunk (small, structurally
    test_config           homogeneous).
    other /             → RecursiveJsonSplitter fallback, preserving
    test_fixtures         structural integrity.

Every chunk text starts with a ``// <rel_path> :: <unit>`` header
so JSON chunks use the same retrieval anchor style as JS/Vue chunks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

JSON_CHUNK_SIZE = 500
JSON_CHUNK_OVERLAP = 80

_JSON_FILENAME_SCHEMA = re.compile(
    r"\.(create|update|get)(?:-[a-z]+)?\.json$", re.IGNORECASE
)


# ── category detection ────────────────────────────────────

def json_category(source: str) -> str:
    """Classify a JSON file by payload role from its relative path.

    Classifies files without importing optional analysis helpers at runtime.
    """
    s = source.replace("\\", "/").lower()
    name = Path(source).name.lower()
    parts = s.split("/")
    if name == "package.json":
        return "package_tooling"
    if "/schemas/" in s and _JSON_FILENAME_SCHEMA.search(name):
        return "schemas_validation"
    if "/i18n/" in s:
        return "i18n_translations"
    if ".vitepress" in parts and name == "manifest.json":
        return "docs_meta"
    if ".vitepress" in parts:
        return "docs_data"
    if parts and (parts[0] == "test" or "test" in parts):
        return "test_config" if "config" in parts else "test_fixtures"
    return "other"


# Categories included in the default index. Callers can pass a different
# category set to include more JSON files.
JSON_INDEXED_CATEGORIES = {
    "schemas_validation",
    "i18n_translations",
    "docs_meta",
    "test_config",
}


# ── breadcrumb helpers ────────────────────────────────────

def _breadcrumb_header(source: str, unit: str) -> str:
    if unit and not unit.startswith("<"):
        return f"// {source} :: {unit}"
    return f"// {source}"


def _emit(source: str, unit: str, category: str, text: str, idx: int, strategy: str) -> dict:
    header = _breadcrumb_header(source, unit)
    return {
        "text": header + "\n" + text,
        "metadata": {
            "source": source,
            "strategy": strategy,
            "chunk_index": idx,
            "breadcrumb": {
                "path": source,
                "symbol": unit if unit and not unit.startswith("<") else "",
                "block": category,
            },
            "block_type": category,
        },
    }


# ── category-specific splitters ───────────────────────────

def _split_schema(data: dict, source: str, category: str, strategy: str) -> list[dict]:
    out: list[dict] = []
    meta = {
        k: data[k]
        for k in ("$id", "title", "description", "type", "required", "groups")
        if k in data
    }
    if meta:
        out.append(_emit(
            source, "<schema-meta>", category,
            json.dumps(meta, ensure_ascii=False, indent=2), len(out), strategy,
        ))
    for pname, pspec in (data.get("properties") or {}).items():
        out.append(_emit(
            source, pname, category,
            json.dumps({pname: pspec}, ensure_ascii=False, indent=2),
            len(out), strategy,
        ))
    return out


def _split_i18n(data: dict, source: str, category: str, strategy: str) -> list[dict]:
    out: list[dict] = []
    flat: dict[str, object] = {}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            body = json.dumps({k: v}, ensure_ascii=False, indent=2)
            if len(body) <= JSON_CHUNK_SIZE * 2 or not isinstance(v, dict):
                out.append(_emit(source, k, category, body, len(out), strategy))
            else:
                splitter = RecursiveJsonSplitter(max_chunk_size=JSON_CHUNK_SIZE)
                for sub in splitter.split_json(json_data={k: v}, convert_lists=True):
                    out.append(_emit(
                        source, k, category,
                        json.dumps(sub, ensure_ascii=False),
                        len(out), strategy,
                    ))
        else:
            flat[k] = v
    if flat:
        out.insert(0, _emit(
            source, "<flat-labels>", category,
            json.dumps(flat, ensure_ascii=False, indent=2), 0, strategy,
        ))
        for i, c in enumerate(out):
            c["metadata"]["chunk_index"] = i
    return out


def _split_package(data: dict, source: str, category: str, strategy: str) -> list[dict]:
    keep = {
        "name": data.get("name"),
        "description": data.get("description"),
        "version": data.get("version"),
        "scripts": data.get("scripts"),
        "dependencies": sorted((data.get("dependencies") or {}).keys()) or None,
        "devDependencies": sorted((data.get("devDependencies") or {}).keys()) or None,
    }
    keep = {k: v for k, v in keep.items() if v is not None}
    return [_emit(
        source, "<package>", category,
        json.dumps(keep, ensure_ascii=False, indent=2), 0, strategy,
    )]


def _split_whole_file(text: str, source: str, category: str, strategy: str) -> list[dict]:
    return [_emit(source, "<file>", category, text, 0, strategy)]


def _split_recursive_json(data, source: str, category: str, strategy: str) -> list[dict]:
    splitter = RecursiveJsonSplitter(
        max_chunk_size=JSON_CHUNK_SIZE,
        min_chunk_size=max(50, JSON_CHUNK_SIZE // 4),
    )
    out: list[dict] = []
    for i, sub in enumerate(splitter.split_json(json_data=data, convert_lists=True)):
        out.append(_emit(
            source, "<subtree>", category,
            json.dumps(sub, ensure_ascii=False), i, strategy,
        ))
    return out


# ── public entry ─────────────────────────────────────────

def chunk_json(
    text: str,
    source: str,
    *,
    strategy: str = "D_json_category_breadcrumb",
) -> list[dict]:
    """Chunk one JSON file with category-aware breadcrumbs.

    Falls back to ``RecursiveCharacterTextSplitter`` if the file is
    malformed — robustness matters more than correctness when a stray
    trailing comma sneaks into a schema.
    """
    category = json_category(source)
    try:
        data = json.loads(text)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=JSON_CHUNK_SIZE, chunk_overlap=JSON_CHUNK_OVERLAP,
        )
        pieces = splitter.split_text(text)
        return [
            _emit(source, f"<rct-{i}>", category, p, i, strategy)
            for i, p in enumerate(pieces)
        ]

    if category == "schemas_validation" and isinstance(data, dict):
        return _split_schema(data, source, category, strategy)
    if category == "i18n_translations" and isinstance(data, dict):
        return _split_i18n(data, source, category, strategy)
    if category == "package_tooling" and isinstance(data, dict):
        return _split_package(data, source, category, strategy)
    if category in ("docs_meta", "test_config"):
        return _split_whole_file(text, source, category, strategy)
    return _split_recursive_json(data, source, category, strategy)


JSON_WINNER = "D_json_category_breadcrumb"
