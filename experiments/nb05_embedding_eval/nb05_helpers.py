"""Helpers used by nb05_embedding_evaluation.ipynb."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


# --- Gold set ----------------------------------------------------------------

GOLD_SCHEMA_VERSION = "nb05.gold.v2"
SUPPORTED_GOLD_SCHEMA_VERSIONS = {"nb05.gold.v1", GOLD_SCHEMA_VERSION}

LayerType = Literal["A_symbol", "B_docs", "C_code", "negative"]
LAYERS: tuple[LayerType, ...] = ("A_symbol", "B_docs", "C_code", "negative")


@dataclass(frozen=True)
class GoldQuery:
    id: str
    layer: LayerType
    en: str
    fr: str
    gold_sources: tuple[str, ...]
    notes: str = ""

    @property
    def is_negative(self) -> bool:
        """Return True when the query is expected to have no gold source."""
        return self.layer == "negative" or not self.gold_sources


def load_gold(path: str | Path) -> list[GoldQuery]:
    """Load the nb05 gold query set from JSON and validate its schema version."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    got = payload.get("version")
    if got not in SUPPORTED_GOLD_SCHEMA_VERSIONS:
        expected = ", ".join(sorted(SUPPORTED_GOLD_SCHEMA_VERSIONS))
        raise ValueError(f"{path} schema version is {got!r}, expected one of: {expected}")
    return [
        GoldQuery(
            id=q["id"],
            layer=q["layer"],
            en=q["en"],
            fr=q["fr"],
            gold_sources=tuple(q.get("gold_sources", ())),
            notes=q.get("notes", ""),
        )
        for q in payload["queries"]
    ]


def load_gold_validated(path: str | Path, corpus_sources: Iterable[str]) -> list[GoldQuery]:
    """Load gold queries and fail fast when any gold source is missing from the corpus."""
    queries = load_gold(path)
    corpus_set = set(corpus_sources)
    missing = [
        (q.id, s)
        for q in queries if not q.is_negative
        for s in q.gold_sources if s not in corpus_set
    ]
    if missing:
        lines = "\n".join(f"  {qid}: {p}" for qid, p in missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise ValueError(f"{len(missing)} gold path(s) not in corpus:\n{lines}{more}")
    return queries


def gold_summary(queries: list[GoldQuery]) -> dict[str, int]:
    """Count gold queries by evaluation layer."""
    counts: dict[str, int] = {layer: 0 for layer in LAYERS}
    for q in queries:
        counts[q.layer] = counts.get(q.layer, 0) + 1
    counts["total"] = len(queries)
    return counts


# --- Embedding recipes -------------------------------------------------------

@dataclass(frozen=True)
class Recipe:
    model_id: str
    short_label: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_tokens: int = 512
    normalize: bool = True
    trust_remote_code: bool = False
    matryoshka_dim: int | None = None
    gpu_batch_size: int = 32
    family: str = "unknown"


RECIPES: dict[str, Recipe] = {
    "bge-m3": Recipe(
        model_id="BAAI/bge-m3",
        short_label="bge-m3",
        max_tokens=8192,
        gpu_batch_size=16,
        family="multilingual-dense",
    ),
    "e5-large": Recipe(
        model_id="intfloat/multilingual-e5-large",
        short_label="e5-large",
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_tokens=512,
        gpu_batch_size=32,
        family="multilingual-dense",
    ),
    "nomic-v1.5": Recipe(
        model_id="nomic-ai/nomic-embed-text-v1.5",
        short_label="nomic-v1.5",
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
        max_tokens=8192,
        trust_remote_code=True,
        matryoshka_dim=768,
        gpu_batch_size=4,
        family="english-dense",
    ),
    "jina-code": Recipe(
        model_id="jinaai/jina-embeddings-v2-base-code",
        short_label="jina-code",
        max_tokens=8192,
        trust_remote_code=True,
        gpu_batch_size=8,
        family="code-dense",
    ),
    "e5-large-instruct": Recipe(
        model_id="intfloat/multilingual-e5-large-instruct",
        short_label="e5-large-instruct",
        query_prefix=(
            "Instruct: Given a developer question, retrieve the relevant "
            "Kalisio documentation page or source file\nQuery: "
        ),
        passage_prefix="",
        max_tokens=512,
        gpu_batch_size=32,
        family="multilingual-dense",
    ),
    "arctic-l-v2": Recipe(
        model_id="Snowflake/snowflake-arctic-embed-l-v2.0",
        short_label="arctic-l-v2",
        query_prefix="query: ",
        passage_prefix="",
        max_tokens=8192,
        gpu_batch_size=16,
        family="multilingual-dense",
    ),
    "qwen3-0.6b": Recipe(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        short_label="qwen3-0.6b",
        query_prefix=(
            "Instruct: Given a developer question in French or English, retrieve "
            "the relevant Kalisio documentation page or source code file.\nQuery: "
        ),
        passage_prefix="",
        max_tokens=8192,
        gpu_batch_size=4,
        family="multilingual-code-dense",
    ),
}


def load_recipe_model(recipe: Recipe, device: str | None = None):
    """Instantiate a SentenceTransformer model for one embedding recipe."""
    from sentence_transformers import SentenceTransformer

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model = SentenceTransformer(
        recipe.model_id,
        device=device,
        trust_remote_code=recipe.trust_remote_code,
    )
    try:
        model.max_seq_length = min(recipe.max_tokens, model.max_seq_length)
    except AttributeError:
        pass
    return model


def _prefix(texts: list[str], prefix: str) -> list[str]:
    """Apply the model-specific query or passage prefix to each text."""
    return [prefix + t for t in texts] if prefix else texts


def _truncate_dim(vectors: np.ndarray, dim: int | None) -> np.ndarray:
    """Apply optional Matryoshka truncation and renormalize the vectors."""
    if dim is None or dim >= vectors.shape[1]:
        return vectors
    out = vectors[:, :dim]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _is_oom(exc: BaseException) -> bool:
    """Detect CUDA or PyTorch out-of-memory exceptions by type or message."""
    name = type(exc).__name__
    msg = str(exc).lower()
    return name in {"OutOfMemoryError", "CudaOutOfMemoryError"} or "out of memory" in msg


def _encode(model, recipe: Recipe, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts with automatic batch-size reduction on OOM failures."""
    bs = max(1, batch_size)
    last: BaseException | None = None
    while bs >= 1:
        try:
            return model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=recipe.normalize,
            )
        except Exception as exc:
            if not _is_oom(exc):
                raise
            last = exc
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if bs == 1:
                break
            bs = max(1, bs // 2)
            print(f"[encode] {recipe.short_label} OOM -> retry batch={bs}")
    raise RuntimeError(f"Persistent OOM for {recipe.short_label} at batch=1") from last


def encode_corpus(model, recipe: Recipe, texts: list[str]) -> np.ndarray:
    """Encode corpus chunks with passage prefixes and recipe-specific truncation."""
    vecs = _encode(model, recipe, _prefix(texts, recipe.passage_prefix), recipe.gpu_batch_size)
    return _truncate_dim(vecs, recipe.matryoshka_dim)


def encode_queries(model, recipe: Recipe, queries: list[str]) -> np.ndarray:
    """Encode evaluation queries with query prefixes and recipe-specific truncation."""
    bs = max(recipe.gpu_batch_size, 16)
    vecs = _encode(model, recipe, _prefix(queries, recipe.query_prefix), bs)
    return _truncate_dim(vecs, recipe.matryoshka_dim)


def truncation_audit(recipe: Recipe, texts: list[str]) -> dict:
    """Measure how many corpus chunks exceed a recipe's token limit."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        recipe.model_id, trust_remote_code=recipe.trust_remote_code
    )
    truncated = total = lost = 0
    for t in texts:
        prefixed = recipe.passage_prefix + t if recipe.passage_prefix else t
        n = len(tok(prefixed, add_special_tokens=True)["input_ids"])
        total += n
        if n > recipe.max_tokens:
            truncated += 1
            lost += n - recipe.max_tokens
    return {
        "model": recipe.short_label,
        "max_tokens": recipe.max_tokens,
        "n_chunks": len(texts),
        "n_truncated": truncated,
        "truncation_rate": truncated / max(len(texts), 1),
        "lost_fraction": lost / max(total, 1),
    }


def measure_throughput(model, recipe: Recipe, texts: list[str]) -> dict:
    """Measure corpus encoding throughput for one loaded embedding model."""
    if not texts:
        return {"chunks_per_sec": 0.0, "n": 0}
    warmup = texts[: min(recipe.gpu_batch_size, len(texts))]
    encode_corpus(model, recipe, warmup)
    t0 = time.perf_counter()
    encode_corpus(model, recipe, texts)
    dt = time.perf_counter() - t0
    return {
        "model": recipe.short_label,
        "n": len(texts),
        "seconds": dt,
        "chunks_per_sec": len(texts) / max(dt, 1e-9),
    }


# --- Dense / BM25 / RRF ------------------------------------------------------

def dense_rank(query_vecs: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Rank corpus vectors for each query using dense dot-product similarity."""
    sims = query_vecs @ corpus_vecs.T
    return np.argsort(-sims, axis=1)


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 with lowercase accent-insensitive matching."""
    # Accent-strip so "géolocaliser" and "geolocaliser" collide (FR typing).
    nfkd = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return _TOKEN_RE.findall(stripped)


def bm25_rank(
    chunk_texts: list[str],
    queries: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Rank corpus chunks for each query with a local BM25 implementation."""
    docs = [tokenize(t) for t in chunk_texts]
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n_docs, 1)

    df: Counter[str] = Counter()
    tfs: list[Counter[str]] = []
    for d in docs:
        tf = Counter(d)
        tfs.append(tf)
        df.update(tf.keys())

    idf = {term: log(1 + (n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()}

    out = np.empty((len(queries), n_docs), dtype=np.int64)
    for qi, q in enumerate(queries):
        q_terms = tokenize(q)
        scores = np.zeros(n_docs, dtype=np.float64)
        for term in q_terms:
            if term not in idf:
                continue
            w = idf[term]
            for di, tf in enumerate(tfs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = len(docs[di])
                denom = f + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
                scores[di] += w * f * (k1 + 1) / max(denom, 1e-9)
        out[qi] = np.argsort(-scores)
    return out


def rrf_fuse(*rank_matrices: np.ndarray, k_rrf: int = 60) -> np.ndarray:
    """Fuse several rank matrices using Reciprocal Rank Fusion."""
    if not rank_matrices:
        raise ValueError("rrf_fuse needs at least one rank matrix")
    shape = rank_matrices[0].shape
    for m in rank_matrices[1:]:
        if m.shape != shape:
            raise ValueError("all rank matrices must share shape")
    n_q, n_c = shape
    out = np.empty(shape, dtype=np.int64)
    for qi in range(n_q):
        scores = np.zeros(n_c, dtype=np.float64)
        for m in rank_matrices:
            for rank, idx in enumerate(m[qi]):
                scores[idx] += 1.0 / (k_rrf + rank + 1)
        out[qi] = np.argsort(-scores)
    return out


# --- Reranker ----------------------------------------------------------------

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


def load_reranker(model_id: str = DEFAULT_RERANKER, device: str | None = None):
    """Load the cross-encoder reranker used for top-k candidate reranking."""
    from sentence_transformers import CrossEncoder

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    return CrossEncoder(model_id, device=device)


def rerank_topk(reranker, query: str, candidate_texts: Sequence[str]) -> np.ndarray:
    """Return candidate order after scoring query-document pairs with the reranker."""
    pairs = [(query, t) for t in candidate_texts]
    scores = reranker.predict(pairs, show_progress_bar=False)
    return np.argsort(-np.asarray(scores))


# --- Evaluation --------------------------------------------------------------

def _file_rank(rank_row: np.ndarray, chunk_sources: list[str]) -> list[str]:
    """Convert chunk-level ranking into a de-duplicated file-level ranking."""
    seen: set[str] = set()
    out: list[str] = []
    for idx in rank_row:
        s = chunk_sources[idx]
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _hit(retrieved_top_k: list[str], gold: tuple[str, ...]) -> int:
    """Return 1 when at least one gold file appears in the retrieved list."""
    if not gold:
        return 0
    gset = set(gold)
    return int(any(s in gset for s in retrieved_top_k))


def _recall(retrieved_top_k: list[str], gold: tuple[str, ...]) -> float:
    """Compute file-level recall for the retrieved list against gold sources."""
    if not gold:
        return 0.0
    gset = set(gold)
    found = sum(1 for s in retrieved_top_k if s in gset)
    return found / len(gold)


def _reciprocal_rank(retrieved_files: list[str], gold: tuple[str, ...]) -> float:
    """Compute reciprocal rank of the first retrieved gold file."""
    if not gold:
        return 0.0
    gset = set(gold)
    for i, s in enumerate(retrieved_files):
        if s in gset:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranks(
    rank_matrix: np.ndarray,
    queries: list[GoldQuery],
    chunk_sources: list[str],
    *,
    language: str,
    approach: str,
    k: int = 5,
    ks: tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """Evaluate one rank matrix against the gold queries at multiple cutoffs."""
    import pandas as pd

    if rank_matrix.shape[0] != len(queries):
        raise ValueError(
            f"rank_matrix has {rank_matrix.shape[0]} rows but {len(queries)} queries"
        )

    positive_gold: set[str] = set()
    for q in queries:
        if not q.is_negative:
            positive_gold.update(q.gold_sources)

    metric_ks = tuple(sorted(set(ks) | {k}))
    rows = []
    for i, q in enumerate(queries):
        files = _file_rank(rank_matrix[i], chunk_sources)
        top_k = files[:k]
        metrics: dict[str, float | int] = {}
        for metric_k in metric_ks:
            current_top = files[:metric_k]
            if q.is_negative:
                metrics[f"hit@{metric_k}"] = int(
                    not any(s in positive_gold for s in current_top)
                )
                metrics[f"recall@{metric_k}"] = float("nan")
            else:
                metrics[f"hit@{metric_k}"] = _hit(current_top, q.gold_sources)
                metrics[f"recall@{metric_k}"] = _recall(current_top, q.gold_sources)
        if q.is_negative:
            hit = int(not any(s in positive_gold for s in top_k))
            recall = float("nan")
            rr = float("nan")
        else:
            hit = _hit(top_k, q.gold_sources)
            recall = _recall(top_k, q.gold_sources)
            rr = _reciprocal_rank(files, q.gold_sources)
        rows.append({
            "approach": approach,
            "language": language,
            "layer": q.layer,
            "query_id": q.id,
            "is_negative": q.is_negative,
            "hit@k": hit,
            "recall@k": recall,
            "mrr": rr,
            **metrics,
        })
    df = pd.DataFrame(rows)
    df.attrs["k"] = k
    return df


def leaderboard(df_all: pd.DataFrame, *, layer: str | None = None) -> pd.DataFrame:
    """Build a language-level hit@k leaderboard for all approaches."""
    df = df_all if layer is None else df_all[df_all["layer"] == layer]
    pivot = (
        df.pivot_table(values="hit@k", index="approach", columns="language", aggfunc="mean")
        .round(3)
    )
    pivot["mean"] = pivot.mean(axis=1).round(3)
    return pivot.sort_values("mean", ascending=False)


def per_layer_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean hit@k for each approach across evaluation layers."""
    return (
        df_all.groupby(["approach", "layer"])["hit@k"]
        .mean()
        .round(3)
        .unstack("layer")
    )


# --- Cost profile ------------------------------------------------------------

def index_size_bytes(n_chunks: int, dim: int, *, dtype_bytes: int = 4) -> int:
    """Estimate the raw dense-vector index size in bytes."""
    return n_chunks * dim * dtype_bytes


def query_latency_ms(
    encode_query_fn,
    corpus_vecs: np.ndarray,
    sample_queries: list[str],
    *,
    top_k: int = 10,
    warmup: int = 2,
) -> dict:
    """Measure query encoding plus dense-search latency over sample queries."""
    for q in sample_queries[:warmup]:
        qv = encode_query_fn(q)
        _ = np.argsort(-(qv @ corpus_vecs.T), axis=1)[:, :top_k]

    times: list[float] = []
    for q in sample_queries:
        t0 = time.perf_counter()
        qv = encode_query_fn(q)
        _ = np.argsort(-(qv @ corpus_vecs.T), axis=1)[:, :top_k]
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times)
    return {
        "n": len(arr),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }
