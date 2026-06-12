"""Embed text into vectors with the configured SentenceTransformer model.

Loads the model named by EMBEDDING_MODEL once (lazily, on first use). The
query and the documents are embedded ASYMMETRICALLY, the way Qwen3-Embedding
is trained for retrieval: encode() prefixes the query with a retrieval
instruction (QUERY_PREFIX), encode_batch() embeds documents as-is. Vectors
are L2-normalized, so cosine similarity in Qdrant reduces to a dot product.

The ingestion job and the API must embed with the SAME model, so a query
vector and a chunk vector are comparable — that is why the model name comes
from the shared configuration rather than being hard-coded here.
"""

import torch
from sentence_transformers import SentenceTransformer

from config import get_runtime_config


# Loaded once on first use; the model is large, so we never reload per call.
_model = None

# Retrieval instruction prepended to every QUERY before encoding. Qwen3-
# Embedding is trained to embed queries asymmetrically with such an
# instruction while documents are embedded without it; dropping it hurts
# retrieval, so it lives in code next to the encoders.
QUERY_PREFIX = (
    "Instruct: Given a developer question in French or English, retrieve "
    "the relevant Kalisio documentation page or source code file.\nQuery: "
)


# Embed a QUERY into a normalized vector, prefixed with the retrieval
# instruction (asymmetric encoding -- documents go through encode_batch).
def encode(text):
    vector = _get_model().encode(
        QUERY_PREFIX + text, show_progress_bar=False,
        normalize_embeddings=True)
    return vector.tolist()


# Embed many documents/passages at once (no query prefix), for throughput.
def encode_batch(texts):
    vectors = _get_model().encode(
        list(texts),
        batch_size=get_runtime_config().embedding_batch_size,
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
            get_runtime_config().embedding_model,
            trust_remote_code=True, device=device)
    return _model
