import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import api.handlers as handlers
from api.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse


app = FastAPI(
    title="knowledge API",
    version=os.getenv("APP_VERSION", "0.1.0"),
    description="RAG retrieval over the Kalisio code corpus.",
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


@app.get(
    "/health",
    summary="Health Check",
    description="Check the health status of the knowledge API.",
)
def health_check():
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question over the Kalisio corpus",
    description=(
        "Embed the question, retrieve matching code chunks from Qdrant, "
        "call the configured LLM, return a natural-language answer "
        "with its sources."
    ),
)
def ask(request: AskRequest):
    return handlers.answer_question(request.question)


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve raw chunks from the Kalisio corpus",
    description=(
        "Embed the query, return the top-k matching code chunks from "
        "Qdrant without calling the LLM. Intended for agents that "
        "compose their own prompt."
    ),
)
def search(request: SearchRequest):
    return handlers.search_chunks(request.query, request.top_k)
