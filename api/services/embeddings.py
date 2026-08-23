"""Embed a question into a query vector.

Loads the model named by EMBEDDING_MODEL once (lazily, on first use). The
query is embedded ASYMMETRICALLY, the way Qwen3-Embedding is trained for
retrieval: encode() prefixes it with a retrieval instruction (QUERY_PREFIX),
while the ingestion job embeds documents without one. Vectors are
L2-normalized, so cosine similarity in Qdrant reduces to a dot product.

The API and the ingestion job must embed with the SAME model for a query
vector and a chunk vector to be comparable -- which is why the model name
comes from the configuration rather than being hard-coded here.
"""

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
# UTILS
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
