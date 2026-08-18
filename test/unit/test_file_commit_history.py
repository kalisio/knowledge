import ingestion.file_commit_history as file_commit_history
from ingestion.file_commit_history import (
    MAX_COMMITS, enrich_chunks_with_commit_history, get_recent_commits)

FIRST = "2026-01-01T00:00:00+00:00"
SECOND = "2026-02-01T00:00:00+00:00"
THIRD = "2026-03-01T00:00:00+00:00"


# --- the subjects kept, newest first --------------------------------------

def test_get_recent_commits_returns_the_subjects_newest_first(repo):
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")
    repo.commit("map/x.js", "const a = 2", SECOND, "center on the bbox")

    assert get_recent_commits(repo.path, "map/x.js") == [
        "center on the bbox", "add the base map"]


def test_get_recent_commits_only_reports_the_requested_file(repo):
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")
    repo.commit("map/y.js", "const b = 1", SECOND, "add the layer store")

    assert get_recent_commits(repo.path, "map/x.js") == ["add the base map"]


def test_get_recent_commits_is_empty_for_an_unknown_file(repo):
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")

    assert get_recent_commits(repo.path, "map/never.js") == []


def test_get_recent_commits_keeps_mechanical_subjects(repo):
    # Judging which subjects carry meaning is not this module's business:
    # every non-merge subject is reported as-is.
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")
    repo.commit("map/x.js", "const a = 2", SECOND, "chore: reformat")

    assert get_recent_commits(repo.path, "map/x.js") == [
        "chore: reformat", "add the base map"]


def test_get_recent_commits_drops_merge_commits(repo):
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")
    repo.git("checkout", "-q", "-b", "feature")
    repo.commit("map/x.js", "const a = 2", SECOND, "center on the bbox")
    repo.git("checkout", "-q", "main")
    repo.git("merge", "--no-ff", "-q", "-m", "merge feature", "feature",
             date=THIRD)

    assert get_recent_commits(repo.path, "map/x.js") == [
        "center on the bbox", "add the base map"]


# --- the limit ------------------------------------------------------------

def test_get_recent_commits_keeps_at_most_the_limit(repo):
    for index in range(4):
        repo.commit("map/x.js", f"const a = {index}", FIRST,
                    f"describe behaviour {index}")

    assert len(get_recent_commits(repo.path, "map/x.js", limit=2)) == 2


def test_get_recent_commits_returns_nothing_for_a_non_positive_limit(repo):
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")

    assert get_recent_commits(repo.path, "map/x.js", limit=0) == []


def test_max_commits_is_the_default_limit(repo):
    for index in range(MAX_COMMITS + 2):
        repo.commit("map/x.js", f"const a = {index}", FIRST,
                    f"describe behaviour {index}")

    assert len(get_recent_commits(repo.path, "map/x.js")) == MAX_COMMITS


# --- git being unavailable ------------------------------------------------

def test_get_recent_commits_is_empty_outside_a_repository(tmp_path):
    # Enriching a chunk is best-effort: a repository the job cannot read must
    # cost the history, never the ingestion.
    assert get_recent_commits(tmp_path, "map/x.js") == []


# --- enriching chunks -----------------------------------------------------

def _chunk(repository, source_path, index=0):
    return {"text": "const a = 1", "metadata": {
        "repository": repository, "source_path": source_path,
        "chunk_index": index}}


def test_enrich_chunks_stamps_each_chunk_with_its_file_history(repo):
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")
    chunks = [_chunk("kdk", "map/x.js", 0), _chunk("kdk", "map/x.js", 1)]

    enrich_chunks_with_commit_history(chunks, repo.workspace)

    assert [c["metadata"]["commit_history"] for c in chunks] == [
        ["add the base map"], ["add the base map"]]


def test_enrich_chunks_fetches_the_history_once_per_file(repo, monkeypatch):
    fetched = []
    monkeypatch.setattr(
        file_commit_history, "get_recent_commits",
        lambda repo_dir, source_path: fetched.append(source_path) or [])
    chunks = [_chunk("kdk", "map/x.js", 0), _chunk("kdk", "map/x.js", 1),
              _chunk("kdk", "map/y.js", 0)]

    enrich_chunks_with_commit_history(chunks, repo.workspace)

    assert fetched == ["map/x.js", "map/y.js"]


def test_enrich_chunks_gives_an_empty_history_outside_a_repository(tmp_path):
    chunks = [_chunk("unknown", "map/x.js")]

    enrich_chunks_with_commit_history(chunks, tmp_path)

    assert chunks[0]["metadata"]["commit_history"] == []
