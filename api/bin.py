#!/usr/bin/env python
"""Command-line entry point that starts the knowledge API server."""

import sys

import uvicorn

from api.config import get_config
from api.logger import configure_logging, get_logger


def run():
    config = get_config()
    configure_logging(config.log_level)
    # Behind the ingress, uvicorn trusts only 127.0.0.1 by
    uvicorn.run("api.main:app", host=config.host, port=config.port,
                proxy_headers=True, forwarded_allow_ips="*")


def main():
    try:
        run()
    except Exception as exc:
        get_logger("api").error("failed to start the API: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
