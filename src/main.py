"""CLI entry point for the v1 FastAPI service."""

from __future__ import annotations

import uvicorn

from ingestion_job.rag_system.config import ApiConfig


def main() -> None:
    config = ApiConfig()
    uvicorn.run(
        "src.app:app",
        host=config.host,
        port=config.port,
        workers=config.workers_count,
    )


if __name__ == "__main__":
    main()

