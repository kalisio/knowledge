"""Embed text into vectors with the configured SentenceTransformer model.

Loads the model named by EMBEDDING_MODEL once (lazily, on first use) and
exposes encode() for a single text and encode_batch() for many. Vectors are
L2-normalized, so cosine similarity in Qdrant reduces to a dot product.

The ingestion job and the API must embed with the SAME model, so a query
vector and a chunk vector are comparable — that is why the model name comes
from the shared configuration rather than being hard-coded here.
"""

import torch
from sentence_transformers import SentenceTransformer

from config import env_int, require


# Loaded once on first use; the model is large, so we never reload per call.
_model = None


# Embed one text into a normalized vector (list of floats).
def encode(text):
    vector = _get_model().encode(
        text, show_progress_bar=False, normalize_embeddings=True)
    return vector.tolist()


# Embed many texts at once; used by the ingestion job for throughput.
def encode_batch(texts):
    vectors = _get_model().encode(
        list(texts),
        batch_size=env_int("EMBEDDING_BATCH_SIZE", 8),
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [vector.tolist() for vector in vectors]


# Load and cache the configured model, on CUDA when available else CPU.
def _get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(
            require("EMBEDDING_MODEL"), trust_remote_code=True, device=device)
    return _model
