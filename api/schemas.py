"""The request and response shapes the API endpoints exchange."""

from pydantic import BaseModel, Field


# One retrieved chunk, as /search returns it. The first five fields are the
# contract: where the code is (path + lines), how well it matches, what it
# says, and why it says it (commit_history). The rest is context a caller
# may use or ignore.
class Chunk(BaseModel):
    path: str
    lines: str = ""
    score: float
    content: str = ""
    commit_history: list[str] = []
    repo: str = ""
    breadcrumb: str = ""
    chunk_index: int | None = None


# What "who imports this file?" answers. `indexed` separates a file nothing
# imports from a file the corpus has never seen: both have no dependents,
# and only one of them means a refactor is safe.
class Dependents(BaseModel):
    repo: str
    path: str
    dependents: list[str] = []
    dependent_count: int = 0
    truncated: bool = False
    dependencies: list[str] = []
    indexed: bool = False


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: list[Chunk]
    provider: str
    model: str


class DependentsRequest(BaseModel):
    repo: str = Field(..., min_length=1, max_length=100)
    path: str = Field(..., min_length=1, max_length=500)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
