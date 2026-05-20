"""FastAPI application for the v1 retriever service."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.handlers import answer_question, search_chunks
from api.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse


app = FastAPI(
    title="knowledge API",
    version=os.getenv("APP_VERSION", "0.1.0"),
    description="RAG retriever API for the Kalisio codebase.",
    contact={
        "name": "Kalisio",
        "url": "https://kalisio.xyz",
        "email": "contact@kalisio.xyz",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, summary="Ask a Kalisio question (LLM-backed)")
def ask(request: AskRequest):
    """Human-facing endpoint: retrieves chunks, calls the LLM, returns an answer.

    Agents should call ``POST /search`` instead to skip the LLM round-trip.
    """
    try:
        return answer_question(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search", response_model=SearchResponse, summary="Retrieve raw chunks (no LLM)")
def search(request: SearchRequest):
    """Agent-facing endpoint: embeds the query, returns top-k chunks from Qdrant.

    No LLM call. Each result carries the full chunk content, ``path``,
    ``repo``, ``lines`` (``start-end`` from the source file), ``score``,
    and ``commit_history`` (last 10 significant commits for the file).
    """
    try:
        return search_chunks(request.query, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
