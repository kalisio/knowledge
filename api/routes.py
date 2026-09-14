"""The endpoints the knowledge API exposes."""

from fastapi import APIRouter, Depends

import api.services.file_context as file_context
import api.services.retrieval as retrieval
from api.schemas import (AskRequest, AskResponse, Chunk, FileContext,
                         FileContextRequest, SearchRequest)
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


@router.post(
    "/file-context",
    response_model=FileContext,
    summary="What the index knows about a file before you change it",
    description=(
        "Read what the ingestion job stored about one file: the files that "
        "import it, the files that historically change in the same commits "
        "-- coupling no import declares -- how often it moves, and its "
        "recent commit subjects. Answers the blast radius of a change, "
        "which retrieval cannot."
    ),
    dependencies=[Depends(verify_jwt)],
)
def file_context_of(request: FileContextRequest):
    return file_context.get_file_context(request.repo, request.path)
