"""
Ask a question with retrieval from Qdrant and generation from LLM (Ollama or Claude).

Usage:
    conda activate knowledge && python scripts/ask_llm_rag.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_tracker import TrackedClient

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "md_chunks_demo"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
LIMIT = 5


def build_context(results) -> str:
    parts: list[str] = []
    for index, point in enumerate(results.points, start=1):
        payload = point.payload or {}
        source = payload.get("source", "")
        section_title = payload.get("section_title", "")
        text = payload.get("text", "")
        parts.append(
            f"[Chunk {index}]\n"
            f"Source: {source}\n"
            f"Section: {section_title}\n"
            f"Content:\n{text}"
        )
    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant for Kalisio documentation.\n"
    "Answer the user's question using only the provided context.\n"
    "Do not use outside knowledge.\n"
    "If the context does not contain enough information to answer the question, say exactly: "
    "\"I do not know based on the provided Kalisio documentation.\"\n"
    "If the question is unrelated to Kalisio, KDK, or the provided documentation, say exactly: "
    "\"I can only answer questions about the provided Kalisio documentation.\"\n"
    "Do not guess. Do not invent facts. Do not answer beyond the retrieved context."
)


def query_qdrant(question: str, client: QdrantClient, model: SentenceTransformer):
    query_vector = model.encode(question).tolist()
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=LIMIT,
        with_payload=True,
    )


def print_sources(results) -> None:
    print("Retrieved chunks:")
    for index, point in enumerate(results.points, start=1):
        payload = point.payload or {}
        source = payload.get("source", "")
        section_title = payload.get("section_title", "")
        print(f"[{index}] score={point.score:.4f} | {source} | {section_title}")
    print()


def main() -> None:
    qdrant_client = QdrantClient(url=QDRANT_URL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    llm = TrackedClient(source="ask_llm_rag")

    print(f"Qdrant: {QDRANT_URL} / collection={COLLECTION_NAME}")
    print(f"LLM: {llm.provider} / model={llm.model}")
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

        results = query_qdrant(question, qdrant_client, embed_model)
        context = build_context(results)
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

        print()
        print_sources(results)

        try:
            answer = llm.ask(user_prompt, system=SYSTEM_PROMPT)
        except Exception as error:
            print(f"LLM request failed: {error}")
            print()
            continue

        print("Answer:")
        print(answer)
        print()

    llm.summary()
    llm.save()
    llm.ledger_summary()


if __name__ == "__main__":
    main()
