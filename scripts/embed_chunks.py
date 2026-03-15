"""
Generate embeddings for chunk records written by chunk_md.py.

Usage:
    uv run python scripts/embed_chunks.py
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "outputs" / "md_chunks.jsonl"
OUTPUT_PATH = ROOT_DIR / "outputs" / "md_chunks_with_embeddings.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def generate_embeddings(chunks: list[dict]):
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks")
    return model.encode(texts, show_progress_bar=True, batch_size=32)


def build_record(chunk: dict, vector, record_id: str) -> dict:
    return {
        "id": record_id,
        "text": chunk["text"],
        "metadata": chunk["metadata"],
        "vector": vector.tolist(),
    }


def write_embedding_records(chunks: list[dict], embeddings, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            record_id = index
            record = build_record(chunk, vector, record_id)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    chunks = load_chunks(INPUT_PATH)
    if not chunks:
        raise SystemExit(f"No chunks found in {INPUT_PATH}")

    embeddings = generate_embeddings(chunks)
    write_embedding_records(chunks, embeddings, OUTPUT_PATH)

    print(f"Embedded chunks written: {len(chunks)}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
