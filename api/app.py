import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
from api.auth import verify_jwt
from api.config import get_config
from utils import vectordb
from utils.logging import configure_logging

import api.handlers as handlers
from api.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse

# Startup banner: log the effective configuration, auth state, and index
# readiness once at boot. Best-effort -- introspection must never stop the
# server from booting.
@asynccontextmanager
async def lifespan(app):
    config = get_config()
    configure_logging(config.log_level)
    log = logging.getLogger("knowledge.api")

    # Fail fast: auth on without APP_SECRET means no token can be verified,
    # so every authenticated request would 500. Refuse to start rather than
    # serve in that broken state.
    if config.auth_enabled and not config.app_secret:
        raise RuntimeError(
            "APP_SECRET is required when auth is enabled -- set APP_SECRET "
            "or disable auth with KNOWLEDGE_AUTH_ENABLED=false")

    # Best-effort startup banner -- introspection must never stop the server.
    try:
        log.info("starting knowledge API on %s:%s", config.host, config.port)
        log.info("qdrant=%s collection=%s",
                 config.qdrant_url, config.qdrant_collection)
        log.info("embedding model=%s", config.embedding_model)
        log.info("llm model=%s endpoint=%s",
                 config.llm_model, config.llm_endpoint)
        log.info("auth %s | secrets: LLM_API_KEY=%s APP_SECRET=%s",
                 "enabled" if config.auth_enabled else "DISABLED",
                 _present(config.llm_api_key), _present(config.app_secret))
        _log_index_status(log, config.qdrant_collection)
    except Exception as exc:
        log.warning("startup banner skipped: %s", exc)
    yield


app = FastAPI(
    title="knowledge API",
    version=os.getenv("APP_VERSION", "0.1.0"),
    description="RAG retrieval over the Kalisio code corpus.",
    contact={
        "name": "Kalisio",
        "url": "https://kalisio.xyz",
        "email": "contact@kalisio.xyz",
    },
    lifespan=lifespan,
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
    dependencies=[Depends(verify_jwt)]
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
    dependencies=[Depends(verify_jwt)]
)
def search(request: SearchRequest):
    return handlers.search_chunks(request.query, request.top_k)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

# Report a secret as "set" or "missing" without exposing its value.
def _present(value):
    return "set" if value else "missing"


# Log the indexed-chunk count; warn and name the ingestion command if empty.
def _log_index_status(log, collection):
    try:
        chunks = vectordb.count()
    except Exception as exc:
        log.warning("could not reach Qdrant to check the index: %s", exc)
        return
    if chunks == 0:
        log.warning("collection '%s' is EMPTY -- run `python -m "
                    "ingestion.main` to index the corpus before querying",
                    collection)
    else:
        log.info("collection '%s' has %d indexed chunks", collection, chunks)