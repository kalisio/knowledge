"""Logging for the knowledge ingestion job."""

import logging
import sys
import time
from contextlib import contextmanager

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Everything the service logs hangs under this logger.
ROOT = "knowledge"


# Attach the "knowledge" logger to stderr at `log_level`. Idempotent: a
# second call only adjusts the level, it does not stack handlers.
def configure_logging(log_level="INFO"):
    logger = logging.getLogger(ROOT)
    logger.setLevel(log_level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
    # Propagation is disabled so records are not emitted a second time by the
    # root/uvicorn handler.
    logger.propagate = False
    return logger


# The logger of one part of the service, e.g. get_logger("api").
def get_logger(name):
    return logging.getLogger(f"{ROOT}.{name}")


# Render a duration the way a human reads it: "1m 23s", "450ms".
def format_duration(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


# Time a step and report how long it took, whether it succeeded or not: a
# run that dies after twenty minutes should still say where those minutes
# went.
@contextmanager
def step(log, number, total, what):
    log.info("step %d/%d  %s", number, total, what)
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        log.error("step %d/%d  %s FAILED after %s", number, total, what,
                  format_duration(time.perf_counter() - started))
        raise
    log.info("step %d/%d  %s done in %s", number, total, what,
             format_duration(time.perf_counter() - started))
