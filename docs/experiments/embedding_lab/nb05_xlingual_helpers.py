"""Helpers for the cross-lingual experiment in §12 of nb05.

The experiment evaluates whether French docs (FR→FR) close the gap that
French queries pay against the English-only Kalisio corpus (FR→EN). The
test corpus is `kalisio/dok` — the user docs for Kalisio Crisis — which
ships parallel EN/FR trees.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from nb05_helpers import (
    Recipe,
    dense_rank,
    encode_corpus,
    encode_queries,
    load_recipe_model,
)


GOLD_SCHEMA_VERSION = "nb05.xlingual.v1"

# Hand-labeled query categories used by §12.4.
# anchor: identical proper noun / number / URL across languages — small expected gap.
# translation: EN/FR terms diverge sharply — large expected gap.
ANCHOR_IDS: frozenset[str] = frozenset({
    "Q-005", "Q-006", "Q-007", "Q-014", "Q-027",
    "Q-034", "Q-035", "Q-036", "Q-037", "Q-042", "Q-043",
})
TRANSLATION_IDS: frozenset[str] = frozenset({
    "Q-018", "Q-022", "Q-024", "Q-029", "Q-032", "Q-033", "Q-040", "Q-044",
})

XLINGUAL_SETTINGS: tuple[str, ...] = ("EN→EN", "FR→EN", "FR→FR", "FR→bi")


# --- Corpus loading ---------------------------------------------------------


def load_dok_corpus(data_root: Path) -> dict[str, Any]:
    """Scan the bilingual dok corpus and split chunks into EN / FR / bilingual.

    The dok docs live in `<data_root>/docs/` (EN) and `<data_root>/docs/fr/`
    (FR). We reinstate `tutorials/` (excluded by the JS/Vue profile default
    but a real content directory in dok) and restrict to `.md`.
    """
    from corpus_filter import scan_corpus
    from corpus_filter.models import FilterConfig
    from corpus_filter.profiles import build_js_vue_rag_profile
    from chunking import chunk_files

    base = build_js_vue_rag_profile()
    cfg = FilterConfig(
        excluded_dirs=base.excluded_dirs - {"docs", "tools", "tutorials"},
        excluded_extensions=base.excluded_extensions,
        excluded_filenames=base.excluded_filenames,
        excluded_patterns=base.excluded_patterns,
        max_file_size=base.max_file_size,
        max_line_length=base.max_line_length,
        included_extensions={".md"},
    )
    scan = scan_corpus(root=Path(data_root), config=cfg)
    chunks = chunk_files(scan.included)

    en_chunks = [c for c in chunks if "/fr/" not in c["metadata"]["source"]]
    fr_chunks = [c for c in chunks if "/fr/" in c["metadata"]["source"]]

    en_texts = [c["text"] for c in en_chunks]
    fr_texts = [c["text"] for c in fr_chunks]
    en_sources = [c["metadata"]["source"] for c in en_chunks]
    fr_sources = [c["metadata"]["source"] for c in fr_chunks]

    return {
        "en_chunks": en_chunks,
        "fr_chunks": fr_chunks,
        "en_texts": en_texts,
        "fr_texts": fr_texts,
        "bi_texts": en_texts + fr_texts,
        "en_sources": en_sources,
        "fr_sources": fr_sources,
        "bi_sources": en_sources + fr_sources,  # explicit ordering: EN block first
    }


def load_xlingual_gold(
    gold_path: Path,
    en_sources: Sequence[str],
    fr_sources: Sequence[str],
) -> dict[str, Any]:
    """Load the cross-lingual gold set and verify every gold path is in the corpus."""
    payload = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    got = payload.get("version")
    if got != GOLD_SCHEMA_VERSION:
        raise ValueError(
            f"{gold_path} schema is {got!r}, expected {GOLD_SCHEMA_VERSION!r}"
        )

    doc_root = payload["doc_root"]
    queries = payload["queries"]

    def en_gold(q: dict) -> list[str]:
        return [f"{doc_root}/{rp}" for rp in q["gold_relpath"]]

    def fr_gold(q: dict) -> list[str]:
        return [f"{doc_root}/fr/{rp}" for rp in q["gold_relpath"]]

    en_set, fr_set = set(en_sources), set(fr_sources)
    missing: list[tuple[str, str, str]] = []
    for q in queries:
        for p in en_gold(q):
            if p not in en_set:
                missing.append(("EN", q["id"], p))
        for p in fr_gold(q):
            if p not in fr_set:
                missing.append(("FR", q["id"], p))
    if missing:
        raise AssertionError(
            f"{len(missing)} gold paths not in corpus: {missing[:5]}"
        )

    return {
        "version": got,
        "doc_root": doc_root,
        "queries": queries,
        "en_gold": en_gold,
        "fr_gold": fr_gold,
    }


# --- Dense bake-off ---------------------------------------------------------


def _hits_at_k(
    rank_row: np.ndarray,
    sources: Sequence[str],
    gold_set: set[str],
    k: int,
) -> int:
    """1 if any of the top-k chunk sources is in `gold_set`, else 0."""
    for idx in rank_row[:k]:
        if sources[idx] in gold_set:
            return 1
    return 0


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_xlingual_bakeoff(
    queries: Sequence[dict],
    corpus: dict[str, Any],
    en_gold: Callable[[dict], Sequence[str]],
    fr_gold: Callable[[dict], Sequence[str]],
    available_models: Sequence[str],
    active_recipes: dict[str, Recipe],
    *,
    k: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Encode each model once across EN / FR / bilingual corpora and score 4 settings.

    Mirrors the GPU discipline of §3: load → encode → score → free. Returns
    a long-format DataFrame with one row per (model, query) and a hit@k
    column for each of EN→EN, FR→EN, FR→FR, FR→bi.
    """
    en_texts, fr_texts = corpus["en_texts"], corpus["fr_texts"]
    en_sources = corpus["en_sources"]
    fr_sources = corpus["fr_sources"]
    bi_sources = corpus["bi_sources"]

    rows: list[dict[str, Any]] = []
    for key in available_models:
        recipe = active_recipes[key]
        if verbose:
            print(f"[xl] {key} -> loading")
        model = load_recipe_model(recipe)

        if verbose:
            print(f"[xl] {key} -> encoding EN ({len(en_texts)}) and FR ({len(fr_texts)}) chunks")
        en_vecs = encode_corpus(model, recipe, en_texts)
        fr_vecs = encode_corpus(model, recipe, fr_texts)
        bi_vecs = np.concatenate([en_vecs, fr_vecs], axis=0)

        enq_vecs = encode_queries(model, recipe, [q["en"] for q in queries])
        frq_vecs = encode_queries(model, recipe, [q["fr"] for q in queries])

        en_en = dense_rank(enq_vecs, en_vecs)
        fr_en = dense_rank(frq_vecs, en_vecs)
        fr_fr = dense_rank(frq_vecs, fr_vecs)
        fr_bi = dense_rank(frq_vecs, bi_vecs)

        for qi, q in enumerate(queries):
            gen, gfr = set(en_gold(q)), set(fr_gold(q))
            gbi = gen | gfr
            rows.append({
                "model": key,
                "query_id": q["id"],
                "section": q["section"],
                "richer_in_fr": q.get("richer_in_fr", False),
                "EN→EN": _hits_at_k(en_en[qi], en_sources, gen, k),
                "FR→EN": _hits_at_k(fr_en[qi], en_sources, gen, k),
                "FR→FR": _hits_at_k(fr_fr[qi], fr_sources, gfr, k),
                "FR→bi": _hits_at_k(fr_bi[qi], bi_sources, gbi, k),
            })

        del model, en_vecs, fr_vecs, bi_vecs, enq_vecs, frq_vecs
        del en_en, fr_en, fr_fr, fr_bi
        _free_gpu()

    df = pd.DataFrame(rows)
    df.attrs["k"] = k
    return df


# --- Aggregation ------------------------------------------------------------


def xlingual_leaderboard(
    df_xl: pd.DataFrame,
    available_models: Sequence[str],
) -> pd.DataFrame:
    """Per-model hit@k leaderboard with the three diagnostic delta columns."""
    settings = list(XLINGUAL_SETTINGS)
    out = (
        df_xl.groupby("model")[settings].mean()
        .round(3)
        .loc[list(available_models)]
    )
    out["gap (FR→FR − FR→EN)"] = (out["FR→FR"] - out["FR→EN"]).round(3)
    out["FR→FR − EN→EN"] = (out["FR→FR"] - out["EN→EN"]).round(3)
    out["FR→bi − FR→FR"] = (out["FR→bi"] - out["FR→FR"]).round(3)
    return out


def categorize_query(qid: str) -> str:
    if qid in ANCHOR_IDS:
        return "anchor"
    if qid in TRANSLATION_IDS:
        return "translation"
    return "general"


def xlingual_category_view(
    df_xl: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Per-category × model hit@k table and headline gap-by-category Series.

    Excludes `richer_in_fr` queries (e.g. Q-038 Hub'Eau) so the cross-lingual
    gap doesn't conflate with content-richness gap.
    """
    clean = df_xl[~df_xl["richer_in_fr"]].copy()
    clean["category"] = clean["query_id"].apply(categorize_query)

    settings = list(XLINGUAL_SETTINGS)
    cat_view = clean.groupby(["category", "model"])[settings].mean().round(3)
    cat_gap = (
        clean.assign(gap=lambda d: d["FR→FR"] - d["FR→EN"])
        .groupby("category")["gap"].mean().round(3)
    )
    return cat_view, cat_gap
