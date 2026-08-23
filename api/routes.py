"""The endpoints the knowledge API exposes."""

from fastapi import APIRouter, Depends

import api.services.retrieval as retrieval
from api.schemas import AskRequest, AskResponse, Chunk, SearchRequest
from api.services.security import verify_jwt

router = APIRouter()


@router.get(
    "/health",
    summary="Health Check",
    description="Check the health status of the knowledge API.",
)
def health_check():
    return {"status": "ok"}


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question over the Kalisio corpus",
    description=(
        "Embed the question, retrieve matching code chunks from Qdrant, "
        "call the configured LLM, return a natural-language answer "
        "with its sources."
    ),
    dependencies=[Depends(verify_jwt)],
)
def ask(request: AskRequest):
    return retrieval.answer_question(request.question)


@router.post(
    "/search",
    response_model=list[Chunk],
    summary="Retrieve raw chunks from the Kalisio corpus",
    description=(
        "Embed the query, return the top-k matching code chunks from "
        "Qdrant without calling the LLM. Intended for agents that "
        "compose their own prompt."
    ),
    dependencies=[Depends(verify_jwt)],
)
def search(request: SearchRequest):
    return retrieval.search_chunks(request.query, request.top_k)
