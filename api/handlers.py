"""RAG request handling logic."""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient

from api.schemas import AskResponse, SearchResponse, SearchResultChunk, SourceChunk
from ingestion_job.rag_system.config import ApiConfig
from ingestion_job.rag_system.embedding import EmbeddingService
from ingestion_job.rag_system.llm import LLMClient
from ingestion_job.rag_system.qdrant_store import create_client, query_points


@lru_cache(maxsize=1)
def get_config() -> ApiConfig:
    return ApiConfig()


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingService:
    return EmbeddingService(get_config())


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return create_client(get_config().qdrant_url)


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return LLMClient(get_config())


def build_context(points, *, max_chars: int) -> tuple[str, list[SourceChunk]]:
    parts: list[str] = []
    sources: list[SourceChunk] = []
    used_chars = 0
    for index, point in enumerate(points, start=1):
        payload = point.payload or {}
        text = str(payload.get("text") or "")
        source_path = str(payload.get("source_path") or "")
        breadcrumb = str(payload.get("breadcrumb") or "")
        header = (
            f"[Chunk {index}]\n"
            f"Source: {source_path}\n"
            f"Breadcrumb: {breadcrumb}\n"
            f"Score: {point.score:.4f}\n"
            "Content:\n"
        )
        remaining = max_chars - used_chars - len(header)
        if remaining <= 0:
            break
        clipped = text[:remaining]
        parts.append(header + clipped)
        used_chars += len(header) + len(clipped)
        sources.append(
            SourceChunk(
                source_path=source_path,
                score=float(point.score),
                repository=str(payload.get("repository") or ""),
                chunk_index=payload.get("chunk_index"),
                breadcrumb=breadcrumb,
                preview=text[:240].replace("\n", " "),
            )
        )
    return "\n\n".join(parts), sources


def build_prompt(template: str, question: str, context: str) -> str:
    return template.format(context=context, question=question)


def answer_question(question: str) -> AskResponse:
    config = get_config()
    query_vector = get_embedder().encode_query(question)
    points = query_points(
        get_qdrant_client(),
        config.qdrant_collection,
        query_vector,
        limit=config.top_k,
    )
    context, sources = build_context(points, max_chars=config.max_context_chars)
    response = get_llm_client().ask(build_prompt(config.llm_prompt, question, context))
    return AskResponse(
        answer=response.answer,
        sources=sources,
        provider=response.provider,
        model=response.model,
    )


def _format_lines(start: int, end: int) -> str:
    if start <= 0 and end <= 0:
        return ""
    if start <= 0:
        return str(end)
    if end <= 0 or end == start:
        return str(start)
    return f"{start}-{end}"


def search_chunks(query: str, top_k: int) -> SearchResponse:
    """Retrieve raw chunks for agent consumption — no LLM call."""
    config = get_config()
    query_vector = get_embedder().encode_query(query)
    points = query_points(
        get_qdrant_client(),
        config.qdrant_collection,
        query_vector,
        limit=top_k,
    )
    results: list[SearchResultChunk] = []
    for point in points:
        payload = point.payload or {}
        results.append(
            SearchResultChunk(
                path=str(payload.get("source_path") or ""),
                repo=str(payload.get("repository") or ""),
                lines=_format_lines(
                    int(payload.get("start_line") or 0),
                    int(payload.get("end_line") or 0),
                ),
                score=float(point.score),
                content=str(payload.get("text") or ""),
                commit_history=list(payload.get("commit_history") or []),
            )
        )
    return SearchResponse(results=results)

