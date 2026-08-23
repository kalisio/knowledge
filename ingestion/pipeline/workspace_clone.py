"""Fills the workspace with the repositories to index, by running k-clone."""

import os
import subprocess
from pathlib import Path

from ingestion.logger import get_logger

log = get_logger("ingestion")

# Where the image installs the Kalisio dev tooling. k-clone looks for it
# inside the workspace instead, so the two are linked before it runs.
TOOLING_DIR = os.getenv("KALISIO_TOOLING_DIR", "/opt/kalisio")

# The directories the tooling needs: the scripts themselves, and the kli
# runner k-clone drives.
TOOLED_DIRECTORIES = ("development", "kli")

# How long k-clone may take. The first run of a large workspace clones every
# repository, so this is generous.
CLONE_TIMEOUT = 3600


# Clone `organization`'s `workspace` into `kalisio_development_dir`.
def clone_workspace(kalisio_development_dir, organization, workspace):
    _install_tooling(Path(kalisio_development_dir))
    _warn_on_missing_token(organization)
    subprocess.run(["bash", "k-clone", organization, workspace],
                   check=True, timeout=CLONE_TIMEOUT)


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Point the workspace at the tooling shipped in the image. k-clone resolves
# everything through $KALISIO_DEVELOPMENT_DIR -- the kli file it reads, the
# kli runner it calls -- so a fresh volume has to lead back to /opt/kalisio.
# A workspace that already holds the tooling is a developer's own checkout,
# bind-mounted from their machine: it is left exactly as it is, links
# included, since a link into the image would dangle outside the container.
def _install_tooling(kalisio_development_dir):
    kalisio_development_dir.mkdir(parents=True, exist_ok=True)
    if (kalisio_development_dir / "development").exists():
        log.info("  the workspace already has its tooling, left as is")
        return
    for name in TOOLED_DIRECTORIES:
        _link(Path(TOOLING_DIR) / name, kalisio_development_dir / name)


# Link `target` to `source`, unless something is already there.
def _link(source, target):
    if target.exists() or target.is_symlink():
        log.info("  %s already in the workspace, left as is", target.name)
        return
    if not source.exists():
        log.warning("  %s is missing from the image (%s): k-clone may fail",
                    source.name, source)
        return
    target.symlink_to(source, target_is_directory=True)
    log.info("  %s linked to the copy shipped in the image (%s)",
             target.name, source)


# The tooling clones over https with a token; without one it falls back to
# ssh, which needs a key the container does not have.
def _warn_on_missing_token(organization):
    variable = "GITLAB_IRSN_TOKEN" if organization == "irsn" \
        else "KALISIO_GITHUB_TOKEN"
    if not os.getenv(variable):
        log.warning("  %s is not set: k-clone will fall back to ssh, which "
                    "needs a key inside the container", variable)
