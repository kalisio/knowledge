"""Serves the retrieval service to coding agents over MCP."""

from typing import Annotated

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

import api.services.retrieval as retrieval
from api.schemas import Chunk
from api.services.security import verify_jwt


# Where the transport is mounted on the API.
MOUNT_PATH = "/mcp"

SEARCH_CODE_DESCRIPTION = (
    "Search the Kalisio codebase and documentation. Call this BEFORE "
    "reading any `.js`, `.vue`, `.json`, or `.md` file when you need to "
    "understand how a module, API, function, or convention works. Returns "
    "the most relevant code chunks with their source path, line numbers, "
    "and recent commit history. Do not use GrepTool on documentation, use "
    "this tool instead."
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
