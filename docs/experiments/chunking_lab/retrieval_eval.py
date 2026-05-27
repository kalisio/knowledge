"""Deterministic retrieval eval for the JS chunking strategies.

Problem this closes: ``js_strategies.boundary_quality`` is a
character-level heuristic. It tells us how often a chunk *starts* at a
clean JS boundary, not whether retrieval actually returns useful code.

This script builds a reproducible gold set from the corpus itself:

  1. Walk every JS file. For each top-level ``export function|class|const X``
     symbol, register one gold query:
         query  = ``"Show the implementation of {X}"``
         gold   = that file's rel_path.

  2. For each strategy, chunk the whole JS corpus, embed every chunk once,
     embed every query, retrieve Top-K chunks by cosine similarity,
     collapse to unique source files, and compute recall@K.

No LLM judge, no hand-labeled data — the gold set is derived from the
code itself, so it is fully reproducible.

The embedding model is picked up from ``EMBEDDING_MODEL`` env var and
defaults to ``sentence-transformers/all-MiniLM-L6-v2`` (fast enough to
run in a notebook). The project's production model
(``nomic-ai/nomic-embed-text-v1.5``) gives better absolute numbers but
the *relative ordering* of strategies is what we care about here.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HELPER = Path(__file__).resolve().parents[1] / "experiment_helper"
sys.path.insert(0, str(_HELPER))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = _HELPER.parents[2]  # repo root; same regardless of file depth

from corpus_filter import scan_corpus  # noqa: E402
from embedding_utils import embed_batch_size, load_embedding_model  # noqa: E402
from retrieval_metrics import recall_at_k, symbol_hit  # noqa: E402

import js_strategies as jsx  # noqa: E402
from hybrid import bm25_rank, rrf_fuse  # noqa: E402

DATA = ROOT / "data"
DEFAULT_MODEL = os.getenv(
    "NB03_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

EXPORT_SYMBOL_RE = re.compile(
    r"^\s*export\s+(?:default\s+)?"
    r"(?:async\s+)?"
    r"(?:function|class|const|let)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


@dataclass
class GoldQuery:
    query: str
    gold_source: str   # rel_path
    symbol: str


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_STOP_PREFIXES = ("get", "set", "make", "use", "create", "build", "is", "has", "to")


def _paraphrase(symbol: str) -> str:
    """Turn ``geoTiffGridSource`` into ``"geo tiff grid source"`` and
    ``JSONReader`` into ``"json reader"`` (consecutive capitals are kept
    as one acronym). Also strip one leading verb prefix so the query is a
    phrase, not the raw identifier — retrieval is forced to *understand* it."""
    words = [w.lower() for w in _CAMEL_RE.split(symbol) if w]
    # Drop 1-char shards like the leading "K" in KError, "A" in AModel.
    words = [w for w in words if len(w) > 1]
    if len(words) > 1 and words[0] in _STOP_PREFIXES:
        words = words[1:]
    return " ".join(words)


_SIZE_BUCKETS = [
    ("small", 0, 2_000),
    ("medium", 2_000, 10_000),
    ("large", 10_000, float("inf")),
]


def _extract_queries_from_file(r, per_file_cap: int) -> list[GoldQuery]:
    """Extract up to *per_file_cap* gold queries from a single file."""
    try:
        text = r.path.read_text(errors="ignore")
    except OSError:
        return []
    out: list[GoldQuery] = []
    for m in EXPORT_SYMBOL_RE.finditer(text):
        sym = m.group(1)
        if sym in {"default", "from"} or len(sym) < 4:
            continue
        phrase = _paraphrase(sym)
        if len(phrase.split()) < 2:
            continue
        out.append(GoldQuery(
            query=f"How does {phrase} ({sym}) work?",
            gold_source=r.rel_path,
            symbol=sym,
        ))
        if len(out) >= per_file_cap:
            break
    return out


def build_gold(files, per_file_cap: int = 3, total_cap: int | None = 200) -> list[GoldQuery]:
    """Stratified gold-query extraction.

    Files are bucketed by size (<2 KB / 2-10 KB / >10 KB) and each bucket
    contributes roughly equally to the final set, so large files with many
    exports aren't drowned out by the small-file majority.
    """
    buckets: dict[str, list] = {name: [] for name, _, _ in _SIZE_BUCKETS}
    for r in files:
        try:
            sz = r.path.stat().st_size
        except OSError:
            continue
        for name, lo, hi in _SIZE_BUCKETS:
            if lo <= sz < hi:
                buckets[name].append(r)
                break

    per_bucket_cap = max(1, (total_cap or 200) // len(buckets))
    gold: list[GoldQuery] = []
    for bname, bucket_files in buckets.items():
        bucket_queries: list[GoldQuery] = []
        for r in bucket_files:
            bucket_queries.extend(_extract_queries_from_file(r, per_file_cap))
        if len(bucket_queries) > per_bucket_cap:
            step = max(1, len(bucket_queries) // per_bucket_cap)
            bucket_queries = bucket_queries[::step][:per_bucket_cap]
        gold.extend(bucket_queries)

    if total_cap and len(gold) > total_cap:
        step = max(1, len(gold) // total_cap)
        gold = gold[::step][:total_cap]
    return gold


def _cosine_top_k(query_vecs: np.ndarray, chunk_vecs: np.ndarray, k: int) -> np.ndarray:
    """Return indices of Top-K chunks per query (assumes L2-normalized vecs)."""
    sims = query_vecs @ chunk_vecs.T
    return np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]


def _compute_metrics(
    chunk_sources: list[str],
    chunk_texts: list[str],
    ranked: np.ndarray,
    gold: list[GoldQuery],
    k_values=(3, 5, 10),
) -> dict:
    out: dict = {"chunks": len(chunk_sources), "queries": len(gold)}
    mrr_vals = []
    for q_idx, g in enumerate(gold):
        rank = None
        for pos, i in enumerate(ranked[q_idx]):
            if chunk_sources[i] == g.gold_source:
                rank = pos + 1
                break
        mrr_vals.append(1.0 / rank if rank else 0.0)

    for k in k_values:
        file_hits = 0
        sym_hits = 0
        for q_idx, g in enumerate(gold):
            # hit@K: file-level (deduplicated to K unique files)
            seen, dedup = set(), []
            for i in ranked[q_idx]:
                s = chunk_sources[i]
                if s not in seen:
                    seen.add(s)
                    dedup.append(s)
                if len(dedup) >= k:
                    break
            if g.gold_source in dedup:
                file_hits += 1
            # sym_hit@K: chunk-level (top-K raw chunks, no dedup)
            top_k_texts = [chunk_texts[i] for i in ranked[q_idx][:k]]
            if any(symbol_hit(t, [g.symbol]) for t in top_k_texts):
                sym_hits += 1
        out[f"hit@{k}"] = round(file_hits / len(gold), 3)
        out[f"sym_hit@{k}"] = round(sym_hits / len(gold), 3)
    out["mrr"] = round(float(np.mean(mrr_vals)), 3)
    return out


def _dense_rank(chunks, gold: list[GoldQuery], model) -> tuple[np.ndarray, list[str], list[str]]:
    """Encode chunks + queries, return dense rank matrix + chunk texts/sources."""
    chunk_texts = [c.text for c in chunks]
    chunk_sources = [c.source for c in chunks]
    bs = embed_batch_size()
    chunk_vecs = model.encode(
        chunk_texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False
    )
    query_vecs = model.encode(
        [g.query for g in gold], batch_size=bs,
        normalize_embeddings=True, show_progress_bar=False,
    )
    sims = np.asarray(query_vecs) @ np.asarray(chunk_vecs).T
    ranked = np.argsort(-sims, axis=1)
    return ranked, chunk_texts, chunk_sources


def evaluate(files, strategy: str, gold: list[GoldQuery], model, k_values=(3, 5, 10)) -> dict:
    chunks = jsx.chunk_corpus(files, strategy)
    if not chunks:
        return {"error": "no chunks"}
    ranked, chunk_texts, chunk_sources = _dense_rank(chunks, gold, model)
    return _compute_metrics(chunk_sources, chunk_texts, ranked, gold, k_values)


def evaluate_hybrid(
    files, gold: list[GoldQuery], model, k_values=(3, 5, 10),
) -> dict:
    """Dense + BM25 fused with RRF on D-strategy chunks."""
    chunks = jsx.chunk_corpus(files, "D_js_plus_breadcrumb")
    if not chunks:
        return {"error": "no chunks"}
    dense_ranked, chunk_texts, chunk_sources = _dense_rank(chunks, gold, model)
    bm25_ranked = bm25_rank(chunk_texts, [g.query for g in gold])
    fused_ranked = rrf_fuse(dense_ranked, bm25_ranked)
    return _compute_metrics(chunk_sources, chunk_texts, fused_ranked, gold, k_values)


def run(sample_limit: int | None = None, strategies=None, model_name: str = DEFAULT_MODEL) -> dict:
    files = scan_corpus(DATA, profile="js_vue_rag").included_with_extensions({".js"})
    if sample_limit and sample_limit < len(files):
        step = max(1, len(files) // sample_limit)
        files = files[::step][:sample_limit]

    gold = build_gold(files)
    if not gold:
        return {"error": "no gold queries"}

    file_sizes: dict[str, int] = {}
    for r in files:
        try:
            file_sizes[r.rel_path] = r.path.stat().st_size
        except OSError:
            pass
    bucket_counts = {"small": 0, "medium": 0, "large": 0}
    for g in gold:
        sz = file_sizes.get(g.gold_source, 0)
        for bname, lo, hi in _SIZE_BUCKETS:
            if lo <= sz < hi:
                bucket_counts[bname] += 1
                break

    model = load_embedding_model(model_name)
    available = list(jsx.STRATEGIES.keys()) + ["F_js_hybrid"]
    strategies = strategies or available
    results: dict = {
        "_meta": {
            "sampled_files": len(files),
            "gold_queries": len(gold),
            "gold_bucket_distribution": bucket_counts,
            "model": model_name,
        }
    }
    for name in strategies:
        if name == "F_js_hybrid":
            results[name] = evaluate_hybrid(files, gold, model)
        else:
            results[name] = evaluate(files, name, gold, model)
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
