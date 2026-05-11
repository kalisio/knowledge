"""Embedding model loading and encoding helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from knowledge.ingestion_job.embedding_utils import load_embedding_model
from knowledge.ingestion_job.rag_system.config import RuntimeConfig


class EmbeddingService:
    """Thin wrapper around SentenceTransformers for corpus and query encoding."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.model = load_embedding_model(config.embedding_model)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, question: str) -> list[float]:
        text = self.config.query_prefix + question if self.config.query_prefix else question
        vector = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.astype(np.float32).tolist()

