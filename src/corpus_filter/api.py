from __future__ import annotations

from pathlib import Path

from .engine import scan_tree
from .models import FilterConfig, ScanResult
from .profiles import PROFILE_DEFAULT, resolve_profile

DEFAULT_SCAN_ROOT = Path(__file__).resolve().parents[2] / "data"


def scan_corpus(
    root: str | Path | None = None,
    config: FilterConfig | None = None,
    profile: str = PROFILE_DEFAULT,
) -> ScanResult:
    """Walk root and classify files as included or excluded.

    - If root is omitted, defaults to repository ``data/``.
    - If config is provided, it takes precedence over profile defaults.
    """
    root_path = Path(root).resolve() if root is not None else DEFAULT_SCAN_ROOT.resolve()
    cfg = config or resolve_profile(profile)
    effective_profile = profile if config is None else f"{profile}+custom_config"
    return scan_tree(root=root_path, config=cfg, profile_name=effective_profile)
