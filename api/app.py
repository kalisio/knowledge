"""FastAPI application for the v1 retriever service."""

from __future__ import annotations

import os

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.handlers import answer_question, get_config, search_chunks
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

bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid_token",
        headers={"WWW-Authenticate": 'Bearer realm="kalisio"'},
    )


def verify_jwt(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    config = get_config()
    if not config.app_secret:
        raise RuntimeError(
            "APP_SECRET is not configured — refusing to accept any JWT. "
            "Set APP_SECRET / JWT_AUDIENCE / JWT_ISSUER in knowledge.enc.env."
        )
    try:
        jwt.decode(
            credentials.credentials,
            config.app_secret,
            algorithms=[config.jwt_algorithm],
            audience=config.jwt_audience,
            issuer=config.jwt_issuer,
            options={"require": ["exp"]},
            leeway=10,
        )
    except jwt.PyJWTError:
        raise _unauthorized()


@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a Kalisio question (LLM-backed)",
    dependencies=[Depends(verify_jwt)],
)
def ask(request: AskRequest):
    """Human-facing endpoint: retrieves chunks, calls the LLM, returns an answer.

    Agents should call ``POST /search`` instead to skip the LLM round-trip.
    """
    try:
        return answer_question(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve raw chunks (no LLM)",
    dependencies=[Depends(verify_jwt)],
)
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
