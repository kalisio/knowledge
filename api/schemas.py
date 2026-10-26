"""The request and response shapes the API endpoints exchange."""

from pydantic import BaseModel, Field


# One retrieved chunk, as /search returns it. The first four fields are the
# contract: where the text is (path + lines), how well it matches, and what
# it says. The rest is context a caller may use or ignore.
class Chunk(BaseModel):
    path: str
    lines: str = ""
    score: float
    content: str = ""
    repo: str = ""
    breadcrumb: str = ""
    chunk_index: int | None = None


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
