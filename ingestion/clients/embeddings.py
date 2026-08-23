"""Embeds document chunks into the vectors stored in Qdrant."""

import time

import torch
from sentence_transformers import SentenceTransformer

from ingestion.config import get_config
from ingestion.logger import format_duration, get_logger

# Loaded once on first use; the model is large, so we never reload per call.
_model = None

# How many documents are handed to the model at once. This is not the
# model's own batch size (EMBEDDING_BATCH_SIZE, which drives the forward
# pass) but how often the run reports where it is: embedding a corpus takes
# minutes, and a job that says nothing for minutes looks hung.
_PROGRESS_SLICE = 256

log = get_logger("embeddings")


# Documents are embedded as-is while the API prefixes a query with a
# retrieval instruction: Qwen3-Embedding is trained for that asymmetry.
# Both sides must use the model named by EMBEDDING_MODEL, or a chunk
# vector and a query vector are not comparable.
# Embed many documents at once (no query prefix), for throughput. Progress
# is logged as it goes, so a long run can be followed and its rate read.
def encode_batch(texts):
    texts = list(texts)
    if not texts:
        return []
    model = _get_model()
    batch_size = get_config().embedding_batch_size
    vectors = []
    started = time.perf_counter()
    for start in range(0, len(texts), _PROGRESS_SLICE):
        chunk = texts[start:start + _PROGRESS_SLICE]
        encoded = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vectors.extend(vector.tolist() for vector in encoded)
        _log_progress(len(vectors), len(texts), started)
    elapsed = time.perf_counter() - started
    log.info("embedded %d documents in %s (%.0f/s)", len(vectors),
             format_duration(elapsed), len(vectors) / max(elapsed, 1e-6))
    return vectors


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# The vector size the model produces. sentence-transformers renamed the
# accessor, so both names are tried before giving up on a detail that is
# only there to be logged.
def _dimension(model):
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        accessor = getattr(model, name, None)
        if accessor is not None:
            return accessor()
    return 0


# Report how far the embedding has got, and what it is going to cost. The
# estimate comes from the rate measured so far, which is steady enough on a
# homogeneous corpus to be worth printing.
def _log_progress(done, total, started):
    if done >= total:
        return
    elapsed = time.perf_counter() - started
    rate = done / max(elapsed, 1e-6)
    remaining = (total - done) / rate if rate else 0
    log.info("  embedded %d/%d documents (%d%%) -- about %s left",
             done, total, 100 * done // total, format_duration(remaining))


# Load and cache the configured model, on CUDA when available else CPU. The
# first call downloads the weights if they are not in the cache, which can
# take minutes -- hence the log lines around it.
def _get_model():
    global _model
    if _model is None:
        config = get_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading embedding model %s on %s "
                 "(downloaded on first use, then cached)",
                 config.embedding_model, device)
        started = time.perf_counter()
        _model = SentenceTransformer(
            config.embedding_model, trust_remote_code=True, device=device)
        log.info("embedding model ready in %s (%d dimensions)",
                 format_duration(time.perf_counter() - started),
                 _dimension(_model))
    return _model
