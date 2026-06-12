"""Command-line entry point that starts the knowledge API server."""

import uvicorn

from api.config import get_config


def main() -> None:
    config = get_config()
    uvicorn.run("api.app:app", host=config.host, port=config.port)


if __name__ == "__main__":
    main()
