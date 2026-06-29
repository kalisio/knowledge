"""Prepare and select the repositories consumed by the ingestion job."""

import subprocess


# Directories under the workspace root that are not part of the indexed
# corpus. Adjust this set to change the production corpus.
SKIP_REPOSITORIES = {"development", "kli", "knowledge", "qdrant_data"}


# Ensure the workspace has development tooling and the selected repositories.
def ensure_repositories(root, config, names=None):
    root.mkdir(parents=True, exist_ok=True)
    _ensure_development(root, config.development_repo_url)
    _run_k_clone(root, config.kli_organization, config.kli_workspace)
    repository_dirs = _discover_repositories(root, names)
    if not repository_dirs:
        selected = ", ".join(names) if names else "any indexable repository"
        raise RuntimeError(
            f"k-clone completed but did not provide {selected} under {root}")
    return repository_dirs


# Ensure root/development exists; it provides k-clone and workspace manifests.
def _ensure_development(root, repo_url):
    development_dir = root / "development"
    k_clone = development_dir / "scripts" / "k-clone"
    if k_clone.exists():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(development_dir)],
        check=True)


# Run k-clone from its scripts directory because it uses relative paths.
def _run_k_clone(root, organization, workspace):
    scripts_dir = root / "development" / "scripts"
    subprocess.run(
        ["bash", "k-clone", organization, workspace],
        cwd=scripts_dir,
        check=True)


# Return first-level git repositories, excluding non-corpus directories.
def _discover_repositories(root, names=None):
    repository_dirs = [
        path for path in sorted(root.iterdir())
        if path.name not in SKIP_REPOSITORIES and (path / ".git").exists()
    ]
    if not names:
        return repository_dirs
    selected = set(names)
    return [path for path in repository_dirs if path.name in selected]
