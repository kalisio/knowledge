import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


# Attach the "knowledge" logger to stderr at `log_level`. Idempotent: a
# second call only adjusts the level, it does not stack handlers.
def configure_logging(log_level="INFO"):
    logger = logging.getLogger("knowledge")
    logger.setLevel(log_level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
    # Propagation is disabled so records are not emitted a second time by the
    # root/uvicorn handler.
    logger.propagate = False
    return logger
