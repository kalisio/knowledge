"""Public API for the corpus filter package.

This module provides the primary entry point 'scan_corpus' for users 
to inventory and filter files within the repository.
"""

from __future__ import annotations

from pathlib import Path

from .engine import scan_tree
from .models import FilterConfig, ScanResult
from .profiles import PROFILE_DEFAULT, resolve_profile

# Default root is knowledge/data/ (assuming this package is in knowledge/src/)
DEFAULT_SCAN_ROOT = Path(__file__).resolve().parents[2] / "data"


def scan_corpus(
    root: str | Path | None = None,
    config: FilterConfig | None = None,
    profile: str = PROFILE_DEFAULT,
) -> ScanResult:
    """The main entry point to scan and filter files in a repository.

    Args:
        root: The directory to start scanning from. Defaults to 'data/'.
        config: A custom FilterConfig object. If provided, it overrides 
                the profile defaults.
        profile: The name of a pre-defined configuration profile 
                 (e.g., 'default', 'js_vue_rag').

    Returns:
        A ScanResult object containing lists of included and excluded files.
    """
    root_path = Path(root).resolve() if root is not None else DEFAULT_SCAN_ROOT.resolve()
    
    # Priority: Custom Config > Profile Config
    cfg = config or resolve_profile(profile)
    effective_profile = profile if config is None else f"{profile}+custom_config"
    
    return scan_tree(root=root_path, config=cfg, profile_name=effective_profile)
