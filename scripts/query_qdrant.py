"""
Query markdown chunks stored in Qdrant.

Usage:
    conda activate knowledge && python scripts/query_qdrant.py
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "md_chunks_demo"
MODEL_NAME = "all-MiniLM-L6-v2"
LIMIT = 5


def print_results(question: str, client: QdrantClient, model: SentenceTransformer) -> None:
    query_vector = model.encode(question).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=LIMIT,
        with_payload=True,
    )

    print()
    print(f"Question: {question}")
    print(f"Top {LIMIT} chunks:")
    print()

    for index, point in enumerate(results.points, start=1):
        payload = point.payload or {}
        source = payload.get("source", "")
        section_title = payload.get("section_title", "")
        text = payload.get("text", "")
        preview = text[:500].replace("\n", " ")

        print(f"[{index}] score={point.score:.4f}")
        print(f"source: {source}")
        print(f"section: {section_title}")
        print(f"text: {preview}")
        print()


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connected to {COLLECTION_NAME} at {QDRANT_URL}")
    print("Type a question. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        print_results(question, client, model)


if __name__ == "__main__":
    main()
