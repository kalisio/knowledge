"""Shared helpers for loading the sentence-transformer embedding model."""

import torch
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer on CUDA if available, else CPU.

    Prints the selected device so every script run logs which backend it used,
    which makes GPU/CPU mismatches obvious across a mixed team.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    if device == "cuda":
        print(f"[embed] cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[embed] cpu (CUDA not available)")
    return model


def embed_batch_size(default_gpu: int = 128, default_cpu: int = 32) -> int:
    """Pick a sensible encode() batch size for the active device."""
    return default_gpu if torch.cuda.is_available() else default_cpu
