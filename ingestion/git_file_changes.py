import subprocess
from pathlib import Path


# Get the files changed in each repo under `root` since `last_ingestion`.
def find_file_changes(root, last_ingestion):
    changed_files = set()
    for git_dir in Path(root).glob("*/.git"):
        repo_dir = git_dir.parent
        output = subprocess.run(
            ["git", "-C", str(repo_dir), "log", f"--since={last_ingestion}",
             "--name-only", "--pretty=format:"],
            capture_output=True, text=True).stdout
        for line in output.splitlines():
            if line:
                changed_files.add(repo_dir / line)
    return changed_files
