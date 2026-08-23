"""Embeds a question into the query vector the API searches Qdrant with."""

import torch
from sentence_transformers import SentenceTransformer

from api.config import get_config

# Loaded once on first use; the model is large, so we never reload per call.
_model = None

# Retrieval instruction prepended to every query before encoding. Qwen3-
# Embedding is trained to embed queries asymmetrically with such an
# instruction while documents are embedded without it; dropping it hurts
# retrieval, so it lives in code next to the encoder.
QUERY_PREFIX = (
    "Instruct: Given a developer question in French or English, retrieve "
    "the relevant Kalisio documentation page or source code file.\nQuery: "
)


# Embed a query into a normalized vector, prefixed with the retrieval
# instruction.
def encode(text):
    vector = _get_model().encode(
        QUERY_PREFIX + text, show_progress_bar=False,
        normalize_embeddings=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Load and cache the configured model, on CUDA when available else CPU.
def _get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(
            get_config().embedding_model, trust_remote_code=True,
            device=device)
    return _model
