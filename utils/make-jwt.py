#!/usr/bin/env python3
"""Dev-only utility to mint a JWT for the knowledge API.

Reads ``APP_SECRET`` / ``JWT_AUDIENCE`` / ``JWT_ISSUER`` from the environment
(source ``development/workspaces/services/services.sh knowledge`` first) and
prints a signed HS256 JWT to stdout.

Usage:
    python utils/make-jwt.py
    python utils/make-jwt.py --days 30 --subject mcp-server

Set ``KNOWLEDGE_API_URL`` (default ``http://localhost:8187``) then test with:
    curl -H "Authorization: Bearer $(python utils/make-jwt.py)" \\
         -H "Content-Type: application/json" \\
         -d '{"question":"What is KDK?"}' \\
         $KNOWLEDGE_API_URL/ask
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jwt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--days", type=int, default=7,
        help="Token lifetime in days (default 7)",
    )
    parser.add_argument(
        "--subject", default="dev",
        help="JWT 'sub' claim (default: dev)",
    )
    args = parser.parse_args()

    secret = os.environ.get("APP_SECRET", "")
    audience = os.environ.get("JWT_AUDIENCE", "kalisio")
    issuer = os.environ.get("JWT_ISSUER", "kalisio")
    algorithm = os.environ.get("JWT_ALGORITHM", "HS256")

    if not secret:
        print(
            "error: APP_SECRET is empty. Source services.sh first:\n"
            "  source development/workspaces/services/services.sh knowledge",
            file=sys.stderr,
        )
        return 1

    if args.days > 30:
        print(
            f"warning: minting a {args.days}-day token — dev tokens "
            "should not outlive a single workstation session.",
            file=sys.stderr,
        )

    now = int(time.time())
    payload = {
        "sub": args.subject,
        "aud": audience,
        "iss": issuer,
        "iat": now,
        "exp": now + args.days * 86400,
    }
    token = jwt.encode(payload, secret, algorithm=algorithm)
    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
