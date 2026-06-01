import os


# Stub: fixed-dimension zero vector until the real Qwen3 model is wired in
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1024))


# Encode a text into a query vector
def encode(txt):
    return [0.0] * EMBEDDING_DIM
