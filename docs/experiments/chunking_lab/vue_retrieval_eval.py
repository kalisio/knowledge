"""Deterministic retrieval eval for Vue SFC chunking strategies.

Three-route gold-query generation:

  1. **Component filenames** — ``KZoomControl.vue`` → strip ``K`` prefix →
     acronym-aware camelCase split → ``"zoom control"``.  Consecutive
     capitals stay as one acronym token (``KHTTPClient`` → ``"http client"``,
     not ``"k h t t p client"``); 1-char shards are dropped.
     gold_source = the ``.vue`` file.
  2. **Composable definitions** — ``export function useCurrentActivity``
     found in ``.js`` files → strip ``use`` → camelCase split →
     ``"current activity"``.  gold_source = the *definition* file (the
     ``.js`` that exports it), **not** the Vue file that calls it.
  3. **Vue 2 registered names** — ``name: 'k-color-chooser'`` → strip
     ``k-`` prefix → kebab split → ``"color chooser"``.  gold_source =
     the ``.vue`` file containing the registration.

Queries are wrapped as ``"How does {phrase} ({symbol}) work?"`` — the
raw identifier is inlined alongside the paraphrased phrase so retrieval
has both a natural-language anchor and a direct symbol hit available.
Each query carries a ``category`` tag so metrics are reported both
overall and per-category.

Six strategies are evaluated:

  - A/B/C/D — the four structural strategies from ``vue_splitter_experiment``.
  - E       — D + the expanded component phrase injected into every
              chunk header (see ``strategy_e``). Dense retrieval only.
  - F       — E chunks retrieved by dense + BM25, fused with Reciprocal
              Rank Fusion (see ``hybrid``).
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

from corpus_filter import scan_corpus  # noqa: E402
from embedding_utils import embed_batch_size, load_embedding_model  # noqa: E402
from retrieval_metrics import symbol_hit  # noqa: E402

import vue_splitter_experiment as vuex  # noqa: E402
from paraphrase import camel_phrase  # noqa: E402
from strategy_e import strategy_e  # noqa: E402
from hybrid import bm25_rank, rrf_fuse  # noqa: E402

DATA = ROOT / "data"
DEFAULT_MODEL = os.getenv(
    "NB03_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

COMPOSABLE_DEF_RE = re.compile(
    r"export\s+(?:default\s+)?(?:async\s+)?function\s+(use[A-Z]\w+)",
    re.MULTILINE,
)
VUE2_NAME_RE = re.compile(r"""name\s*:\s*['"]([^'"]+)['"]""")


@dataclass
class GoldQuery:
    query: str
    gold_source: str          # rel_path of the gold file
    symbol: str               # original identifier or name
    category: str             # "component_name" | "composable" | "vue2_name"


def _kebab_phrase(name: str) -> str:
    return " ".join(w.lower() for w in name.split("-") if len(w) > 1)


def _build_component_queries(vue_files) -> list[GoldQuery]:
    """Route 1: component filenames → queries."""
    out: list[GoldQuery] = []
    for r in vue_files:
        stem = Path(r.rel_path).stem
        name = stem[1:] if stem.startswith("K") and len(stem) > 1 else stem
        if len(name) < 3:
            continue
        phrase = camel_phrase(name)
        if len(phrase.split()) < 2:
            continue
        out.append(GoldQuery(
            query=f"How does {phrase} ({stem}) work?",
            gold_source=r.rel_path,
            symbol=stem,
            category="component_name",
        ))
    return out


def _build_composable_queries(all_included_files) -> list[GoldQuery]:
    """Route 2: composable definitions → queries.

    gold_source is the *definition* file (where ``export function useXxx``
    appears), not the Vue file that calls it. Single-word composables
    (e.g. ``useStore`` → ``"store"``) use a more specific query template
    that mentions "composable" to reduce ambiguity.
    """
    out: list[GoldQuery] = []
    seen: set[str] = set()
    for r in all_included_files:
        if r.path.suffix not in (".js", ".mjs", ".vue"):
            continue
        try:
            text = r.path.read_text(errors="ignore")
        except OSError:
            continue
        for m in COMPOSABLE_DEF_RE.finditer(text):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            phrase = camel_phrase(name[3:])  # strip "use"
            if not phrase:
                continue
            if len(phrase.split()) < 2:
                query = f"How does the {phrase} composable ({name}) work?"
            else:
                query = f"How does {phrase} ({name}) work?"
            out.append(GoldQuery(
                query=query,
                gold_source=r.rel_path,
                symbol=name,
                category="composable",
            ))
    return out


def _build_vue2_name_queries(vue_files) -> list[GoldQuery]:
    """Route 3: Vue 2 registered ``name:`` property → queries."""
    out: list[GoldQuery] = []
    for r in vue_files:
        try:
            text = r.path.read_text(errors="ignore")
        except OSError:
            continue
        m = VUE2_NAME_RE.search(text)
        if not m:
            continue
        raw_name = m.group(1)
        name = raw_name[2:] if raw_name.startswith("k-") else raw_name
        phrase = _kebab_phrase(name)
        if len(phrase.split()) < 2:
            continue
        out.append(GoldQuery(
            query=f"How does {phrase} ({raw_name}) work?",
            gold_source=r.rel_path,
            symbol=raw_name,
            category="vue2_name",
        ))
    return out


def build_gold(vue_files, all_included_files) -> list[GoldQuery]:
    """Combine all three gold-query routes."""
    gold: list[GoldQuery] = []
    gold.extend(_build_component_queries(vue_files))
    gold.extend(_build_composable_queries(all_included_files))
    gold.extend(_build_vue2_name_queries(vue_files))
    return gold


def _chunk_js_files(js_files) -> list[vuex._Chunk]:
    """Chunk JS files (composable definitions) with the JS-aware splitter
    so they are retrievable alongside Vue chunks."""
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=vuex.CHUNK_SIZE, chunk_overlap=vuex.CHUNK_OVERLAP
    )
    out: list[vuex._Chunk] = []
    for r in js_files:
        text = r.path.read_text(errors="ignore")
        for piece in splitter.split_text(text):
            out.append(vuex._Chunk(text=piece, source=r.rel_path, block="script"))
    return out


def _chunk_corpus(vue_files, splitter_fn, extra_js_files, *, small_sfc_threshold: int = 0) -> list[vuex._Chunk]:
    chunks: list[vuex._Chunk] = []
    for r in vue_files:
        text = r.path.read_text(errors="ignore")
        if small_sfc_threshold and len(text) < small_sfc_threshold:
            chunks.append(vuex._Chunk(text=text, source=r.rel_path, block="whole"))
        else:
            chunks.extend(splitter_fn(text, r.rel_path))
    if extra_js_files:
        chunks.extend(_chunk_js_files(extra_js_files))
    return chunks


def _metrics(
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
            top_k_texts = [chunk_texts[i] for i in ranked[q_idx][:k]]
            if any(symbol_hit(t, [g.symbol]) for t in top_k_texts):
                sym_hits += 1
        out[f"hit@{k}"] = round(file_hits / len(gold), 3)
        out[f"sym_hit@{k}"] = round(sym_hits / len(gold), 3)
    out["mrr"] = round(float(np.mean(mrr_vals)), 3)
    return out


def _by_category(
    chunk_sources, chunk_texts, ranked, gold, k_values=(3, 5, 10),
) -> dict:
    cats = sorted({g.category for g in gold})
    out: dict = {}
    for cat in cats:
        idxs = [i for i, g in enumerate(gold) if g.category == cat]
        if not idxs:
            continue
        cat_gold = [gold[i] for i in idxs]
        cat_ranked = ranked[idxs]
        out[cat] = _metrics(chunk_sources, chunk_texts, cat_ranked, cat_gold, k_values)
    return out


def _dense_rank(chunks: list[vuex._Chunk], gold: list[GoldQuery], model) -> np.ndarray:
    bs = embed_batch_size()
    chunk_vecs = model.encode(
        [c.text for c in chunks], batch_size=bs,
        normalize_embeddings=True, show_progress_bar=False,
    )
    query_vecs = model.encode(
        [g.query for g in gold], batch_size=bs,
        normalize_embeddings=True, show_progress_bar=False,
    )
    sims = np.asarray(query_vecs) @ np.asarray(chunk_vecs).T
    return np.argsort(-sims, axis=1)


_ABCD_STRATEGIES = {
    "A_recursive_generic": vuex.strategy_a,
    "B_recursive_html": vuex.strategy_b,
    "C_sfc_aware": vuex.strategy_c,
    "D_sfc_plus_breadcrumb": vuex.strategy_d,
}


def run(
    model_name: str = DEFAULT_MODEL,
    strategies: list[str] | None = None,
    small_sfc_threshold: int = 0,
) -> dict:
    """Run full retrieval eval.

    ``strategies`` can restrict to a subset of
    ``{"A_recursive_generic", "B_recursive_html", "C_sfc_aware",
    "D_sfc_plus_breadcrumb", "E_expanded", "F_hybrid"}``. When F is
    requested, E is evaluated as well because F reuses E's chunks and
    E's dense ranking.
    """
    scan = scan_corpus(DATA, profile="js_vue_rag")
    vue_files = scan.included_with_extensions({".vue"})
    gold = build_gold(vue_files, scan.included)
    if not gold:
        return {"error": "no gold queries"}

    composable_sources = {g.gold_source for g in gold if g.category == "composable"}
    extra_js = [r for r in scan.included if r.rel_path in composable_sources]

    cat_counts: dict[str, int] = {}
    for g in gold:
        cat_counts[g.category] = cat_counts.get(g.category, 0) + 1

    model = load_embedding_model(model_name)

    requested = strategies or list(_ABCD_STRATEGIES) + ["E_expanded", "F_hybrid"]
    need_e = ("E_expanded" in requested) or ("F_hybrid" in requested)

    results: dict = {
        "_meta": {
            "vue_files": len(vue_files),
            "extra_js_files": len(extra_js),
            "gold_queries": len(gold),
            "gold_by_category": cat_counts,
            "model": model_name,
            "small_sfc_threshold": small_sfc_threshold,
        }
    }

    for name, fn in _ABCD_STRATEGIES.items():
        if name not in requested:
            continue
        chunks = _chunk_corpus(vue_files, fn, extra_js, small_sfc_threshold=small_sfc_threshold)
        sources = [c.source for c in chunks]
        texts = [c.text for c in chunks]
        ranked = _dense_rank(chunks, gold, model)
        results[name] = _metrics(sources, texts, ranked, gold)
        results[f"{name}__by_category"] = _by_category(sources, texts, ranked, gold)

    if need_e:
        e_chunks = _chunk_corpus(vue_files, strategy_e, extra_js, small_sfc_threshold=small_sfc_threshold)
        e_sources = [c.source for c in e_chunks]
        e_texts = [c.text for c in e_chunks]
        e_ranked = _dense_rank(e_chunks, gold, model)
        if "E_expanded" in requested:
            results["E_expanded"] = _metrics(e_sources, e_texts, e_ranked, gold)
            results["E_expanded__by_category"] = _by_category(e_sources, e_texts, e_ranked, gold)
        if "F_hybrid" in requested:
            bm25_ranked = bm25_rank(e_texts, [g.query for g in gold])
            f_ranked = rrf_fuse(e_ranked, bm25_ranked)
            results["F_hybrid"] = _metrics(e_sources, e_texts, f_ranked, gold)
            results["F_hybrid__by_category"] = _by_category(e_sources, e_texts, f_ranked, gold)

    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
