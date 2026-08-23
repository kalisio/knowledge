#!/usr/bin/env python
"""Command-line entry point that runs one ingestion."""

import sys

from ingestion.config import get_config
from ingestion.logger import configure_logging, get_logger
from ingestion.main import run


def main():
    config = get_config()
    configure_logging(config.log_level)
    try:
        return run()
    except Exception as exc:
        get_logger("ingestion").exception("ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
