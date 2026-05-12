"""FastAPI application for the v1 retriever service."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from knowledge.src.handlers import answer_question
from knowledge.src.schemas import AskRequest, AskResponse


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


@app.post("/ask", response_model=AskResponse, summary="Ask a Kalisio question")
def ask(request: AskRequest):
    try:
        return answer_question(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
