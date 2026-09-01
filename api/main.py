"""Builds the FastAPI application: middlewares, errors, routes, checks."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_config
from api.logger import configure_logging, get_logger
from api.routes import router
from api.clients.llm import LLMUnreachable
from api.clients.vectordb import QdrantUnreachable
import api.clients.vectordb as vectordb
import api.services.mcp as mcp_server


# Startup checks: fail fast on a broken auth configuration, then log the
# effective configuration, auth state, and index readiness once at boot.
@asynccontextmanager
async def lifespan(app):
    config = get_config()
    configure_logging(config.log_level)
    log = get_logger("api")

    # Fail fast: auth on without APP_SECRET means no token can be verified,
    # so every authenticated request would 500.
    if config.auth_enabled and not config.app_secret:
        raise RuntimeError(
            "APP_SECRET is required when auth is enabled -- set APP_SECRET "
            "or disable auth with KNOWLEDGE_AUTH_ENABLED=false")

    # Best-effort startup banner -- introspection must never stop the server.
    try:
        log.info("starting knowledge API on %s:%s", config.host, config.port)
        log.info("qdrant=%s collection=%s",
                 config.qdrant_url, config.qdrant_collection_code)
        log.info("embedding model=%s", config.embedding_model)
        log.info("llm model=%s endpoint=%s",
                 config.llm_model, config.llm_endpoint)
        log.info("auth %s | secrets: LLM_API_KEY=%s APP_SECRET=%s",
                 "enabled" if config.auth_enabled else "DISABLED",
                 _present(config.llm_api_key), _present(config.app_secret))
        _log_index_status(log, config.qdrant_collection_code)
    except Exception as exc:
        log.warning("startup banner skipped: %s", exc)

    # The MCP sub-app starts its session manager in its own lifespan, which
    # mounting does not run -- delegate to it for the server's lifetime. One
    # sub-app per startup: a session manager cannot be started twice.
    mcp_app = mcp_server.build_http_app()
    app.state.mcp_app = mcp_app
    async with mcp_app.router.lifespan_context(mcp_app):
        yield


# Build the application. A factory rather than a module-level app, so a test
# can build one against its own configuration.
def create_app():
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

    # A dependency that is down is an outage, not a bug in the request:
    # answer 503 with a readable reason instead of a bare 500.
    @app.exception_handler(QdrantUnreachable)
    def qdrant_unreachable_handler(request, exc):
        get_logger("api").error("qdrant unreachable: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMUnreachable)
    def llm_unreachable_handler(request, exc):
        get_logger("api").error("llm unreachable: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.include_router(router)

    # The same retrieval service, spoken as MCP, behind the same token.
    # The lifespan owns the sub-app, so the mount reads it per request.
    async def mcp_app(scope, receive, send):
        await app.state.mcp_app(scope, receive, send)

    app.mount(mcp_server.MOUNT_PATH,
              mcp_server.BearerJWTMiddleware(mcp_app))

    # A mount only matches below itself, so /mcp would be redirected onto
    # /mcp/ -- and that redirect is fatal behind the ingress
    app.add_middleware(mcp_server.TrailingSlashMiddleware,
                       path=mcp_server.MOUNT_PATH)
    return app


app = create_app()


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Report a secret as "set" or "missing" without exposing its value.
def _present(value):
    return "set" if value else "missing"


# Log the indexed-chunk count; warn and name the ingestion command if empty.
def _log_index_status(log, collection):
    try:
        chunks = vectordb.count_chunks()
    except Exception as exc:
        log.warning("could not reach Qdrant to check the index: %s", exc)
        return
    if chunks == 0:
        log.warning("collection '%s' is EMPTY -- run `python -m "
                    "ingestion.bin` to index the corpus before querying",
                    collection)
    else:
        log.info("collection '%s' has %d indexed chunks", collection, chunks)
