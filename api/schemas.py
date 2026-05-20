"""API response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceChunk(BaseModel):
    source_path: str
    score: float
    repository: str = ""
    chunk_index: int | None = None
    breadcrumb: str = ""
    preview: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    provider: str
    model: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResultChunk(BaseModel):
    path: str
    repo: str
    lines: str
    score: float
    content: str
    commit_history: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SearchResultChunk]
