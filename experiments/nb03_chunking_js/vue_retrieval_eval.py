"""Deterministic retrieval eval for Vue SFC chunking strategies.

Three-route gold-query generation:

  1. **Component filenames** — ``KZoomControl.vue`` → strip ``K`` prefix →
     camelCase split → ``"zoom control"``.  gold_source = the ``.vue`` file.
  2. **Composable definitions** — ``export function useCurrentActivity``
     found in ``.js`` files → strip ``use`` → camelCase split →
     ``"current activity"``.  gold_source = the *definition* file (the
     ``.js`` that exports it), **not** the Vue file that calls it.
  3. **Vue 2 registered names** — ``name: 'k-color-chooser'`` → strip
     ``k-`` prefix → kebab split → ``"color chooser"``.  gold_source =
     the ``.vue`` file containing the registration.

Queries are wrapped as ``"How does the code handle <phrase>?"``, same as
the JS eval.  Each query carries a ``category`` tag so metrics can be
reported both overall and per-category.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "nb03_chunking_js"))

from corpus_filter import scan_corpus  # noqa: E402
from embedding_utils import embed_batch_size, load_embedding_model  # noqa: E402
from retrieval_metrics import symbol_hit  # noqa: E402

import vue_splitter_experiment as vuex  # noqa: E402

DATA = ROOT / "data"
DEFAULT_MODEL = os.getenv(
    "NB03_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
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


def _camel_split(name: str) -> str:
    return " ".join(w.lower() for w in _CAMEL_RE.split(name) if w)


def _kebab_split(name: str) -> str:
    return " ".join(w.lower() for w in name.split("-") if w)


def _build_component_queries(vue_files) -> list[GoldQuery]:
    """Route 1: component filenames → queries."""
    out: list[GoldQuery] = []
    for r in vue_files:
        stem = Path(r.rel_path).stem
        name = stem[1:] if stem.startswith("K") and len(stem) > 1 else stem
        if len(name) < 3:
            continue
        phrase = _camel_split(name)
        if len(phrase.split()) < 2:
            continue
        out.append(GoldQuery(
            query=f"How does the code handle {phrase}?",
            gold_source=r.rel_path,
            symbol=stem,
            category="component_name",
        ))
    return out


def _build_composable_queries(all_included_files) -> list[GoldQuery]:
    """Route 2: composable definitions → queries.

    gold_source is the *definition* file (where ``export function useXxx``
    appears), not the Vue file that calls it.

    Single-word composables (e.g. ``useStore`` → ``"store"``) use a more
    specific query template that mentions "composable" to reduce ambiguity.
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
            stripped = name[3:]  # strip "use"
            phrase = _camel_split(stripped)
            if not phrase:
                continue
            if len(phrase.split()) < 2:
                query = f"How does the {phrase} composable work?"
            else:
                query = f"How does the code handle {phrase}?"
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
        name = raw_name
        if name.startswith("k-"):
            name = name[2:]
        phrase = _kebab_split(name)
        if len(phrase.split()) < 2:
            continue
        out.append(GoldQuery(
            query=f"How does the code handle {phrase}?",
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
    """Chunk JS files (composable definitions) with Strategy B (JS-aware)
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


def evaluate(
    vue_files,
    strategy: str,
    gold: list[GoldQuery],
    model,
    k_values=(3, 5, 10),
    *,
    small_sfc_threshold: int = 0,
    extra_js_files=None,
) -> dict:
    """Run retrieval eval for one Vue strategy.

    If *small_sfc_threshold* > 0, Vue files smaller than this (in chars)
    are kept as a single chunk regardless of the strategy.

    *extra_js_files*, when provided, are chunked with the JS splitter and
    added to the corpus so composable-definition queries can match.
    """
    chunks: list[vuex._Chunk] = []
    fn = vuex.STRATEGIES[strategy]
    for r in vue_files:
        text = r.path.read_text(errors="ignore")
        if small_sfc_threshold and len(text) < small_sfc_threshold:
            chunks.append(vuex._Chunk(text=text, source=r.rel_path, block="whole"))
        else:
            chunks.extend(fn(text, r.rel_path))

    if extra_js_files:
        chunks.extend(_chunk_js_files(extra_js_files))

    if not chunks:
        return {"error": "no chunks"}

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

    out: dict = {"chunks": len(chunks), "queries": len(gold)}

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


def _metrics_by_category(
    vue_files,
    strategy: str,
    gold: list[GoldQuery],
    model,
    k_values=(3, 5, 10),
    *,
    small_sfc_threshold: int = 0,
    extra_js_files=None,
) -> dict[str, dict]:
    """Evaluate and report metrics per query category."""
    categories = sorted({g.category for g in gold})
    results: dict[str, dict] = {}
    for cat in categories:
        cat_gold = [g for g in gold if g.category == cat]
        if not cat_gold:
            continue
        results[cat] = evaluate(
            vue_files, strategy, cat_gold, model, k_values,
            small_sfc_threshold=small_sfc_threshold,
            extra_js_files=extra_js_files,
        )
    return results


def run(
    model_name: str = DEFAULT_MODEL,
    strategies: list[str] | None = None,
    small_sfc_threshold: int = 0,
) -> dict:
    scan = scan_corpus(DATA, profile="js_vue_rag")
    vue_files = scan.included_with_extensions({".vue"})
    all_files = scan.included

    gold = build_gold(vue_files, all_files)
    if not gold:
        return {"error": "no gold queries"}

    composable_sources = {g.gold_source for g in gold if g.category == "composable"}
    extra_js = [r for r in all_files if r.rel_path in composable_sources]

    cat_counts = {}
    for g in gold:
        cat_counts[g.category] = cat_counts.get(g.category, 0) + 1

    model = load_embedding_model(model_name)
    strategies = strategies or list(vuex.STRATEGIES.keys())

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
    for name in strategies:
        results[name] = evaluate(
            vue_files, name, gold, model,
            small_sfc_threshold=small_sfc_threshold,
            extra_js_files=extra_js,
        )
        results[f"{name}__by_category"] = _metrics_by_category(
            vue_files, name, gold, model,
            small_sfc_threshold=small_sfc_threshold,
            extra_js_files=extra_js,
        )
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
