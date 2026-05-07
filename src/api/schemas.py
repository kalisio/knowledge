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
