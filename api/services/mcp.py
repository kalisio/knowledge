"""Serves the retrieval service to coding agents over MCP."""

from typing import Annotated

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

import api.services.dependencies as dependencies
import api.services.retrieval as retrieval
from api.schemas import Chunk, Dependents
from api.services.security import verify_jwt


# Where the transport is mounted on the API.
MOUNT_PATH = "/mcp"

SEARCH_CODE_DESCRIPTION = (
    "Search the Kalisio codebase, documentation and DevOps configuration. "
    "Call this BEFORE reading any `.js`, `.vue`, `.json`, `.md`, `.yaml`, "
    "`.yaml.gotmpl` or `.sh` file when you need to understand how a "
    "module, API, function, convention, Helm chart, cluster configuration, "
    "CI workflow or kash script works. Returns the most relevant chunks "
    "with their source path, line numbers, and recent commit history. "
    "Does not cover Python. Do not use GrepTool on documentation, use "
    "this tool instead."
)

GET_DEPENDENTS_DESCRIPTION = (
    "List the files that import a given file of the Kalisio corpus. Call "
    "this BEFORE changing the signature, the exports or the behaviour of an "
    "existing file, to know what would break: it answers what a search "
    "cannot, because it reads the import graph rather than the text. Give "
    "the repository name and the path inside it, as `search_code` returns "
    "them. `dependent_count` is exact even when the list is truncated; "
    "`indexed: false` means the corpus does not hold that file, which is "
    "not the same as nothing depending on it."
)


# Build the MCP server and register the tools it exposes.
def build_server():
    server = MCPServer(
        "kalisio-knowledge",
        instructions="RAG retrieval over the Kalisio code corpus.",
    )

    # The same contract as POST /search: same bounds, same chunk shape.
    @server.tool(description=SEARCH_CODE_DESCRIPTION)
    def search_code(
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        top_k: Annotated[int, Field(ge=1, le=50)] = 5,
    ) -> list[Chunk]:
        return retrieval.search_chunks(query, top_k)

    # Reads the graph the ingestion job builds -- the one question no
    # amount of semantic search can answer.
    @server.tool(description=GET_DEPENDENTS_DESCRIPTION)
    def get_dependents(
        repo: Annotated[str, Field(min_length=1, max_length=100)],
        path: Annotated[str, Field(min_length=1, max_length=500)],
    ) -> Dependents:
        return dependencies.get_dependents(repo, path)

    return server


# Serve the transport at the mount root, so the API mounts it under /mcp.
def build_http_app():
    return build_server().streamable_http_app(
        streamable_http_path="/", stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False))


# Serves `path` as if it were `path/`, so the mount answers both spellings.
class TrailingSlashMiddleware:
    def __init__(self, app, path):
        self.app = app
        self.path = path

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == self.path:
            scope = {**scope, "path": self.path + "/"}
        await self.app(scope, receive, send)


# Checks the Bearer token in front of the mounted MCP app.
class BearerJWTMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            verify_jwt(_scope_credentials(scope))
        except HTTPException as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Read the Bearer token off the raw ASGI headers, in the shape verify_jwt
# expects. None when the header is absent or is not a Bearer token. Header
# bytes are latin-1, the encoding ASGI and Starlette read them with.
def _scope_credentials(scope):
    for name, value in scope["headers"]:
        if name == b"authorization":
            scheme, _, token = value.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and token:
                return HTTPAuthorizationCredentials(
                    scheme=scheme, credentials=token)
            return None
    return None
