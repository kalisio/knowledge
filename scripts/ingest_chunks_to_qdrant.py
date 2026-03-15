"""
Ingest embedded markdown chunks into Qdrant.

Usage:
    uv run python scripts/ingest_chunks_to_qdrant.py
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT_DIR / "outputs" / "md_chunks_with_embeddings.jsonl"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "md_chunks_demo"


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    records = load_records(INPUT_PATH)
    if not records:
        raise SystemExit(f"No records found in {INPUT_PATH}")

    vector_size = len(records[0]["vector"])
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=record["id"],
            vector=record["vector"],
            payload={
                "text": record["text"],
                **record["metadata"],
            },
        )
        for record in records
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    info = client.get_collection(COLLECTION_NAME)
    print(f"collection={COLLECTION_NAME} points={info.points_count}")


if __name__ == "__main__":
    main()
