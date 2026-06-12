"""JWT bearer authentication for the knowledge API.

The check here mirrors how scripts/make-jwt.py mints tokens: same secret,
same algorithm, same audience/issuer. A token is accepted only if its
signature verifies against APP_SECRET and its aud/iss/exp claims hold.
"""

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.config import get_config


# Take the Bearer token from the Authorization header. auto_error=False:
# hand None to verify_jwt when none is present, so we return our own 401.
bearer_scheme = HTTPBearer(auto_error=False)


# FastAPI dependency: verify the Bearer JWT; skip entirely when auth is off.
def verify_jwt(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    config = get_config()

    # 1. Auth disabled (local dev): let everything through.
    if not config.auth_enabled:
        return

    # 2. A Bearer token must be present.
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise_unauthorized("missing_bearer_token")

    # 3. Fail closed: with auth on but no secret, refuse rather than run open.
    if config.app_secret is None:
        raise HTTPException(
            status_code=500,
            detail="auth misconfigured: APP_SECRET is not set",
        )

    # 4. Verify signature + audience + issuer + expiry. Any failure -> 401.
    #    algorithms is an explicit allow-list so a token cannot pick "none".
    try:
        return jwt.decode(
            credentials.credentials,
            config.app_secret,
            algorithms=[config.jwt_algorithm],
            audience=config.jwt_audience,
            issuer=config.jwt_issuer,
        )
    except jwt.PyJWTError:
        raise_unauthorized("invalid_token")


# Build a 401 carrying the Bearer challenge header.
def raise_unauthorized(detail="invalid_token"):
    raise HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": 'Bearer realm="kalisio"'},
    )
