"""Thin wrapper around the ``kli`` Node tool to clone/update repos.

In K8s the ingestion CronJob needs to bootstrap a PV with the Kalisio
repos before scanning. In dev, the same path works against
``$KALISIO_DEVELOPMENT_DIR``. Both modes invoke kli the same way:

    node <kli_dir>/index.js <workspace.js> --no-fail-on-error --clone

This mirrors what ``development/scripts/k-clone`` does for humans.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def clone_with_kli(
    kli_config: Path,
    ingestion_root: Path,
    *,
    kli_dir: Path | None = None,
) -> None:
    """Run kli on ``kli_config`` to clone or update repos under ``ingestion_root``.

    Requires Node on PATH and a kli checkout (``index.js``) at ``kli_dir``
    — defaults to ``<ingestion_root>/kli``, matching the convention used
    by ``k-clone``.
    """
    target_dir = kli_dir if kli_dir is not None else ingestion_root / "kli"
    entry = target_dir / "index.js"
    if not entry.is_file():
        raise RuntimeError(
            f"kli not found at {entry}. Run `k-clone <workspace>` once or set KLI_DIR."
        )
    if not kli_config.is_file():
        raise RuntimeError(f"KLI_CONFIG file does not exist: {kli_config}")

    subprocess.run(
        [
            "node",
            str(entry),
            str(kli_config),
            "--no-fail-on-error",
            "--clone",
        ],
        check=True,
        cwd=ingestion_root,
    )
