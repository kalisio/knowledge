from pydantic import BaseModel, Field


class Chunk(BaseModel):
    source_path: str
    repository: str
    breadcrumb: str = ""
    chunk_index: int | None = None
    score: float
    content: str = ""
    commit_history: list[str] = []


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: list[Chunk]
    provider: str
    model: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    results: list[Chunk]
