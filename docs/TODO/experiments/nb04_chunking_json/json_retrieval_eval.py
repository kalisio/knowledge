"""Deterministic retrieval eval for the JSON chunking strategies.

Closes the gap the §6 summary flagged ("no retrieval benchmark for JSON").
nb03 set the precedent: mine a gold set directly from the corpus so the
benchmark is reproducible — no LLM judge, no hand labels.

Two gold-query routes, matching the two categories that dominate the
indexed JSON slice (schemas + i18n, per nb01's priority table):

  1. **i18n value-to-key** — developer asks "where is this string set?".
     For each included i18n file, pick a handful of distinctive string
     values (scalar leaves, 8–80 chars, alphabetic). The *query* is the
     value itself ("Site temporarily unavailable"); the *gold source* is
     the set of files where that exact value appears. A value that shows
     up in many files (e.g. a generic "OK") is skipped — it carries no
     localization signal.

  2. **Schema property-name** — developer asks "which schema defines
     field X?". For each included schema, pick each top-level property
     that exposes a ``field.component`` (the Kalisio UI extension). The
     *query* combines the property name and the component in natural
     form ("form field {prop} using {component}"); the *gold source* is
     the set of schemas where that ``(prop, component)`` pair appears.

Strategies evaluated:

  - ``B_recursive_json`` — structural baseline.
  - ``C_category_aware`` — semantic key split without the breadcrumb.
  - ``D_category_plus_breadcrumb`` — the nb04 winner.
  - ``F_json_hybrid`` — D chunks retrieved by dense + BM25 fused with RRF.

A_rct is deliberately excluded: its 0.7 % parse integrity means its
chunks are broken JSON fragments, which the structural experiment has
already ruled out as unfit for the code-gen agent.

Metrics mirror nb03 (``hit@K``, ``mrr``), but ``hit@K`` here accepts a
*set* of gold sources (since a value or a ``(prop, component)`` pair
can legitimately appear in more than one file — a hit against any of
them counts). Per-category breakdowns let us tell i18n from schema.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "nb03_chunking_js"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_filter import scan_corpus  # noqa: E402
from embedding_utils import embed_batch_size, load_embedding_model  # noqa: E402

import json_splitter_experiment as jsonx  # noqa: E402
from json_inventory import JsonFileView, load_views  # noqa: E402
from hybrid import bm25_rank, rrf_fuse  # noqa: E402

DATA = ROOT / "data"
DEFAULT_MODEL = os.getenv(
    "NB04_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


@dataclass
class GoldQuery:
    query: str
    gold_sources: frozenset[str]   # any-of match
    symbol: str                    # raw identifier for diagnostics
    category: str                  # "i18n" | "schema"


# ── i18n gold: value → {files where value appears} ─────────

_WORD_RE = re.compile(r"[A-Za-z]")
_I18N_PER_FILE_CAP = 8
_I18N_MIN_LEN, _I18N_MAX_LEN = 8, 80
# A value appearing in too many files carries no localization signal
# (generic labels like "OK", "Cancel"). 4 is empirical — a translation
# legitimately shared across 2–3 KDK packages still qualifies.
_I18N_MAX_FILE_SHARE = 4


def _iter_i18n_leaves(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _iter_i18n_leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _iter_i18n_leaves(v, f"{path}[{i}]")
    else:
        yield path, d


def _i18n_value_index(views: list[JsonFileView]) -> dict[str, set[str]]:
    """Map each string value to the set of file paths where it appears."""
    idx: dict[str, set[str]] = defaultdict(set)
    for v in views:
        if v.category != "i18n_translations" or not v.parse_ok:
            continue
        if not isinstance(v.data, dict):
            continue
        for _, leaf in _iter_i18n_leaves(v.data):
            if not isinstance(leaf, str):
                continue
            leaf = leaf.strip()
            if not (_I18N_MIN_LEN <= len(leaf) <= _I18N_MAX_LEN):
                continue
            if not _WORD_RE.search(leaf):
                continue
            idx[leaf].add(v.rel_path)
    return idx


def _build_i18n_gold(views: list[JsonFileView]) -> list[GoldQuery]:
    idx = _i18n_value_index(views)
    by_file: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    for value, files in idx.items():
        if len(files) > _I18N_MAX_FILE_SHARE:
            continue
        for f in files:
            by_file[f].append((value, frozenset(files)))

    gold: list[GoldQuery] = []
    for f, entries in by_file.items():
        # Prefer the most distinctive (longest, least-shared) values per file.
        entries.sort(key=lambda e: (len(e[1]), -len(e[0])))
        for value, sources in entries[:_I18N_PER_FILE_CAP]:
            gold.append(GoldQuery(
                query=value,
                gold_sources=sources,
                symbol=value,
                category="i18n",
            ))
    return gold


# ── schema gold: (prop, component) → {schemas with that pair} ──

_SCHEMA_PER_FILE_CAP = 3
# Property names so generic they dominate nearly every schema — these
# give ambiguous queries regardless of strategy, so we skip them.
_SCHEMA_SKIP_PROPS = {"name", "description", "title", "id", "type"}
# UI components stripped of the Kalisio "K" prefix so the query phrase
# reads naturally ("text field" rather than "k text field").
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _component_phrase(component: str) -> str:
    """``form/KTextField`` → ``"text field"``."""
    base = component.rsplit("/", 1)[-1]
    if base.startswith("K") and len(base) > 1:
        base = base[1:]
    words = [w.lower() for w in _CAMEL_RE.split(base) if len(w) > 1]
    return " ".join(words) or base.lower()


def _prop_phrase(name: str) -> str:
    """``streetAddress`` → ``"street address"`` (kept as-is if not camel)."""
    words = [w.lower() for w in _CAMEL_RE.split(name) if w]
    if len(words) == 1 and "_" in name:
        words = [w for w in name.split("_") if w]
    return " ".join(words) if len(words) > 1 else name.lower()


def _schema_pair_index(
    views: list[JsonFileView],
) -> dict[tuple[str, str], set[str]]:
    idx: dict[tuple[str, str], set[str]] = defaultdict(set)
    for v in views:
        if v.category != "schemas_validation" or not v.parse_ok:
            continue
        if not isinstance(v.data, dict):
            continue
        props = v.data.get("properties") or {}
        for pname, pspec in props.items():
            if pname in _SCHEMA_SKIP_PROPS:
                continue
            if not isinstance(pspec, dict):
                continue
            component = (pspec.get("field") or {}).get("component")
            if not component:
                continue
            idx[(pname, component)].add(v.rel_path)
    return idx


def _build_schema_gold(views: list[JsonFileView]) -> list[GoldQuery]:
    idx = _schema_pair_index(views)
    by_file: dict[str, list[tuple[tuple[str, str], frozenset[str]]]] = defaultdict(list)
    for pair, files in idx.items():
        for f in files:
            by_file[f].append((pair, frozenset(files)))

    gold: list[GoldQuery] = []
    for f, entries in by_file.items():
        # Prefer rarer (prop, component) pairs — rarer = more diagnostic.
        entries.sort(key=lambda e: len(e[1]))
        for (pname, component), sources in entries[:_SCHEMA_PER_FILE_CAP]:
            phrase = f"form field {_prop_phrase(pname)} using {_component_phrase(component)}"
            gold.append(GoldQuery(
                query=phrase,
                gold_sources=sources,
                symbol=f"{pname}:{component}",
                category="schema",
            ))
    return gold


def build_gold(views: list[JsonFileView]) -> list[GoldQuery]:
    return _build_i18n_gold(views) + _build_schema_gold(views)


# ── retrieval core (mirrors nb03 retrieval_eval) ──────────

def _dense_rank(chunks, gold: list[GoldQuery], model):
    chunk_texts = [c.text for c in chunks]
    chunk_sources = [c.source for c in chunks]
    bs = embed_batch_size()
    chunk_vecs = model.encode(
        chunk_texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
    )
    query_vecs = model.encode(
        [g.query for g in gold], batch_size=bs,
        normalize_embeddings=True, show_progress_bar=False,
    )
    sims = np.asarray(query_vecs) @ np.asarray(chunk_vecs).T
    ranked = np.argsort(-sims, axis=1)
    return ranked, chunk_texts, chunk_sources


def _compute_metrics(
    chunk_sources: list[str],
    ranked: np.ndarray,
    gold: list[GoldQuery],
    k_values=(5, 10),
) -> dict:
    n = len(gold)
    out: dict = {"chunks": len(chunk_sources), "queries": n}
    mrr_vals = []
    for q_idx, g in enumerate(gold):
        rank = None
        for pos, i in enumerate(ranked[q_idx]):
            if chunk_sources[i] in g.gold_sources:
                rank = pos + 1
                break
        mrr_vals.append(1.0 / rank if rank else 0.0)

    for k in k_values:
        hits = 0
        for q_idx, g in enumerate(gold):
            seen, dedup = set(), []
            for i in ranked[q_idx]:
                s = chunk_sources[i]
                if s not in seen:
                    seen.add(s)
                    dedup.append(s)
                if len(dedup) >= k:
                    break
            if any(src in g.gold_sources for src in dedup):
                hits += 1
        out[f"hit@{k}"] = round(hits / n, 3) if n else 0.0
    out["mrr"] = round(float(np.mean(mrr_vals)) if mrr_vals else 0.0, 3)

    # Per-category breakdown so i18n and schema performance are visible.
    by_cat: dict[str, list[int]] = defaultdict(list)
    for q_idx, g in enumerate(gold):
        by_cat[g.category].append(q_idx)
    out["by_category"] = {}
    for cat, idxs in by_cat.items():
        sub: dict = {"queries": len(idxs)}
        for k in k_values:
            hits = 0
            for q_idx in idxs:
                g = gold[q_idx]
                seen, dedup = set(), []
                for i in ranked[q_idx]:
                    s = chunk_sources[i]
                    if s not in seen:
                        seen.add(s)
                        dedup.append(s)
                    if len(dedup) >= k:
                        break
                if any(src in g.gold_sources for src in dedup):
                    hits += 1
            sub[f"hit@{k}"] = round(hits / len(idxs), 3)
        sub_mrr = [mrr_vals[i] for i in idxs]
        sub["mrr"] = round(float(np.mean(sub_mrr)), 3)
        out["by_category"][cat] = sub
    return out


def _evaluate(views, strategy: str, gold, model, k_values=(5, 10)) -> dict:
    chunks = jsonx.chunk_corpus(views, strategy)
    if not chunks:
        return {"error": "no chunks"}
    ranked, _, chunk_sources = _dense_rank(chunks, gold, model)
    return _compute_metrics(chunk_sources, ranked, gold, k_values)


def _evaluate_hybrid(views, gold, model, k_values=(5, 10)) -> dict:
    """Dense + BM25 fused with RRF on the D-strategy chunks.

    JSON text is mostly identifiers and key names — the exact regime
    where BM25 shines and dense embeddings blur. If the JS result from
    nb03 transfers, hybrid should lift hit@5 a few points over D.
    """
    chunks = jsonx.chunk_corpus(views, "D_category_plus_breadcrumb")
    if not chunks:
        return {"error": "no chunks"}
    dense_ranked, chunk_texts, chunk_sources = _dense_rank(chunks, gold, model)
    bm25_ranked = bm25_rank(chunk_texts, [g.query for g in gold])
    fused = rrf_fuse(dense_ranked, bm25_ranked)
    return _compute_metrics(chunk_sources, fused, gold, k_values)


# ── public entry ────────────────────────────────────────

STRATEGIES = (
    "B_recursive_json",
    "C_category_aware",
    "D_category_plus_breadcrumb",
    "F_json_hybrid",
)


def run(model_name: str = DEFAULT_MODEL, strategies=None) -> dict:
    scan = scan_corpus(DATA, profile="js_vue_rag")
    views = load_views(scan.included_with_extensions({".json"}))
    # Gold mining and indexing both run on the categories that carry
    # signal per nb01 — matches ``JSON_INDEXED_CATEGORIES`` in production.
    views = [v for v in views if v.category in {"schemas_validation", "i18n_translations"}]

    gold = build_gold(views)
    if not gold:
        return {"error": "no gold queries"}

    by_cat_counts = {"i18n": 0, "schema": 0}
    for g in gold:
        by_cat_counts[g.category] += 1

    model = load_embedding_model(model_name)
    strategies = strategies or STRATEGIES
    results: dict = {
        "_meta": {
            "files": len(views),
            "gold_queries": len(gold),
            "gold_by_category": by_cat_counts,
            "model": model_name,
        }
    }
    for name in strategies:
        if name == "F_json_hybrid":
            results[name] = _evaluate_hybrid(views, gold, model)
        else:
            results[name] = _evaluate(views, name, gold, model)
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
