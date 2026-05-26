"""BM25 + dense retrieval fused with Reciprocal Rank Fusion (RRF).

Why hybrid. A bi-encoder embeds ``KZoomControl`` as one token and fails to
match the query phrase ``"zoom control"``. BM25 tokenizes on non-alnum and
also camelCase-splits identifiers, so it scores a direct term hit on the
breadcrumb path. The two signals are complementary.

RRF (Cormack et al., 2009) is parameter-light and robust: each retriever
produces a ranking and every chunk's fused score is

    score(c) = sum over retrievers r of 1 / (k_rrf + rank_r(c))

with ``k_rrf = 60`` as the commonly used default. No score normalization
is needed, which is why RRF survives across heterogeneous retrievers.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Lowercase + alnum-split + camelCase-split, drop 1-char shards.

    Matches BM25's term-frequency lens to the query paraphrase: both sides
    see ``zoom`` and ``control`` as separate tokens even when the source
    wrote ``KZoomControl``.
    """
    out: list[str] = []
    for raw in _TOKEN_SPLIT_RE.split(text):
        if not raw:
            continue
        for piece in _CAMEL_SPLIT_RE.split(raw):
            p = piece.lower()
            if len(p) > 1:
                out.append(p)
    return out


def rrf_fuse(
    dense_ranked: np.ndarray,
    bm25_ranked: np.ndarray,
    k_rrf: int = 60,
) -> np.ndarray:
    """Fuse two [n_queries, n_chunks] ranked-index matrices into one."""
    n_q, n_c = dense_ranked.shape
    scores = np.zeros((n_q, n_c), dtype=np.float64)
    for q in range(n_q):
        for pos in range(n_c):
            scores[q, dense_ranked[q, pos]] += 1.0 / (k_rrf + pos + 1)
            scores[q, bm25_ranked[q, pos]] += 1.0 / (k_rrf + pos + 1)
    return np.argsort(-scores, axis=1)


def bm25_rank(chunk_texts: list[str], query_texts: list[str]) -> np.ndarray:
    """Return [n_queries, n_chunks] indices ranked by BM25 score."""
    tokenized = [tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized)
    q_tokens = [tokenize(q) for q in query_texts]
    scores = np.stack([bm25.get_scores(qt) for qt in q_tokens])
    return np.argsort(-scores, axis=1)
