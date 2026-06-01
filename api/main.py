"""Command-line entry point that starts the knowledge API server."""

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8187"))
    workers = int(os.getenv("WORKERS_COUNT", "1"))

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
