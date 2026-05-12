"""Corpus filter package with profile-driven scan presets.

Usage:
    from corpus_filter import scan_corpus, FilterConfig

    # Default profile
    result = scan_corpus()

    # Retrieval-oriented profile
    rag_result = scan_corpus(profile="js_vue_rag")

    # Custom config (takes precedence over profile defaults)
    cfg = FilterConfig(max_file_size=200_000)
    custom = scan_corpus(config=cfg, profile="default")
"""

from .api import scan_corpus
from .models import FileRecord, FilterConfig, ScanResult
from .profiles import PROFILE_DEFAULT, PROFILE_JS_VUE_RAG, available_profiles

__all__ = [
    "FileRecord",
    "FilterConfig",
    "PROFILE_DEFAULT",
    "PROFILE_JS_VUE_RAG",
    "ScanResult",
    "available_profiles",
    "scan_corpus",
]
