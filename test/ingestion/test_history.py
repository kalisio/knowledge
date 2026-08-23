"""The commit history window, read from real git repositories.

The commits are made with known subjects and known dates on a tmp_path, then
read back. Nothing about git is mocked -- what is asserted is what `git log`
really returns for that file, which is the only way to catch what actually
goes wrong here: a rename that loses the history, a window that empties a
stable file, a subject that does not survive the trip.
"""

import time

from ingestion.config import get_config
from ingestion.services.history import collect_commit_history, read_history

FIRST = "2026-01-01T00:00:00+00:00"
SECOND = "2026-02-01T00:00:00+00:00"
THIRD = "2026-03-01T00:00:00+00:00"


# An ISO date `days` ago, for commits placed relative to the window.
def days_ago(days):
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                         time.gmtime(time.time() - days * 86400))


# --- what comes back -------------------------------------------------------

def test_the_subjects_come_back_newest_first(repo):
    repo.commit("map/x.js", "const a = 1", days_ago(20), "add the base map")
    repo.commit("map/x.js", "const a = 2", days_ago(10), "center on the bbox")

    assert read_history(repo.path, "map/x.js") == [
        "center on the bbox", "add the base map"]


def test_only_the_commits_touching_the_file_are_reported(repo):
    repo.commit("map/x.js", "const a = 1", days_ago(10), "add the base map")
    repo.commit("map/y.js", "const b = 1", days_ago(5), "add the layer store")

    assert read_history(repo.path, "map/x.js") == ["add the base map"]


def test_an_unknown_file_has_no_history(repo):
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")

    assert read_history(repo.path, "map/never.js") == []


def test_the_history_follows_a_renamed_file(repo):
    # The point of the history is to explain the code, and code moves.
    repo.commit("map/x.js", "const a = 1", days_ago(20), "add the base map")
    repo.git("mv", "map/x.js", "map/base-map.js")
    repo.git("commit", "-q", "-m", "rename to base-map", date=days_ago(10))

    assert read_history(repo.path, "map/base-map.js") == [
        "rename to base-map", "add the base map"]


def test_a_merge_commit_is_not_reported(repo):
    # A merge records a branch topology, not a change to the file.
    repo.commit("map/x.js", "const a = 1", FIRST, "add the base map")
    repo.git("checkout", "-q", "-b", "feature")
    repo.commit("map/x.js", "const a = 2", SECOND, "center on the bbox")
    repo.git("checkout", "-q", "main")
    repo.git("merge", "--no-ff", "-q", "-m", "merge feature", "feature",
             date=THIRD)

    assert read_history(repo.path, "map/x.js") == [
        "center on the bbox", "add the base map"]


# --- the sliding window ----------------------------------------------------

def test_a_commit_inside_the_window_is_kept(repo, ingestion_env):
    ingestion_env(COMMIT_HISTORY_MAX_AGE_DAYS=180, COMMIT_HISTORY_MIN_COMMITS=0)
    repo.commit("map/x.js", "const a = 1", days_ago(30), "recent enough")

    assert read_history(repo.path, "map/x.js") == ["recent enough"]


def test_a_commit_older_than_the_window_drops_off(repo, ingestion_env):
    ingestion_env(COMMIT_HISTORY_MAX_AGE_DAYS=180, COMMIT_HISTORY_MIN_COMMITS=0)
    repo.commit("map/x.js", "const a = 1", days_ago(400), "long forgotten")
    repo.commit("map/x.js", "const a = 2", days_ago(30), "still relevant")

    assert read_history(repo.path, "map/x.js") == ["still relevant"]


def test_new_commits_join_the_window(repo, ingestion_env):
    # Reading the window back from git is what adds them: there is no stored
    # history to append to, so nothing can go out of sync.
    ingestion_env(COMMIT_HISTORY_MAX_AGE_DAYS=180, COMMIT_HISTORY_MIN_COMMITS=0)
    repo.commit("map/x.js", "const a = 1", days_ago(30), "first change")
    assert read_history(repo.path, "map/x.js") == ["first change"]

    repo.commit("map/x.js", "const a = 2", days_ago(1), "second change")

    assert read_history(repo.path, "map/x.js") == [
        "second change", "first change"]


def test_a_wider_window_reaches_further_back(repo, ingestion_env):
    repo.commit("map/x.js", "const a = 1", days_ago(400), "long forgotten")
    repo.commit("map/x.js", "const a = 2", days_ago(30), "still relevant")

    ingestion_env(COMMIT_HISTORY_MAX_AGE_DAYS=730, COMMIT_HISTORY_MIN_COMMITS=0)
    assert read_history(repo.path, "map/x.js") == [
        "still relevant", "long forgotten"]


def test_a_window_of_zero_keeps_everything(repo, ingestion_env):
    ingestion_env(COMMIT_HISTORY_MAX_AGE_DAYS=0, COMMIT_HISTORY_MIN_COMMITS=0)
    repo.commit("map/x.js", "const a = 1", days_ago(3000), "ancient history")

    assert read_history(repo.path, "map/x.js") == ["ancient history"]


# --- the floor under the window --------------------------------------------

def test_a_stable_file_keeps_its_history_despite_the_window(
        repo, ingestion_env):
    # 84% of the kdk files have no commit in the last six months, and those
    # are exactly the files whose "why" is hardest to guess from the code.
    ingestion_env(COMMIT_HISTORY_MAX_AGE_DAYS=180, COMMIT_HISTORY_MIN_COMMITS=2)
    repo.commit("map/x.js", "const a = 1", days_ago(900), "why it exists")
    repo.commit("map/x.js", "const a = 2", days_ago(800), "why it is odd")

    assert read_history(repo.path, "map/x.js") == [
        "why it is odd", "why it exists"]


def test_the_floor_keeps_the_most_recent_commits(repo, ingestion_env):
    ingestion_env(COMMIT_HISTORY_MAX_AGE_DAYS=180, COMMIT_HISTORY_MIN_COMMITS=2)
    for index in range(4):
        repo.commit("map/x.js", f"const a = {index}", days_ago(900 - index),
                    f"old change {index}")

    assert read_history(repo.path, "map/x.js") == ["old change 3",
                                                   "old change 2"]


def test_an_active_file_is_not_trimmed_down_to_the_floor(repo, ingestion_env):
    ingestion_env(COMMIT_HISTORY_MAX_AGE_DAYS=180, COMMIT_HISTORY_MIN_COMMITS=2)
    for index in range(5):
        repo.commit("map/x.js", f"const a = {index}", days_ago(30 - index),
                    f"recent change {index}")

    assert len(read_history(repo.path, "map/x.js")) == 5


def test_the_floor_cannot_invent_commits(repo, ingestion_env):
    ingestion_env(COMMIT_HISTORY_MIN_COMMITS=10)
    repo.commit("map/x.js", "const a = 1", days_ago(900), "the only one")

    assert read_history(repo.path, "map/x.js") == ["the only one"]


# --- the optional cap ------------------------------------------------------

def test_the_history_is_uncapped_by_default():
    assert get_config().commit_history_depth == 0


def test_the_cap_trims_a_very_active_file(repo, ingestion_env):
    ingestion_env(COMMIT_HISTORY_DEPTH=2, COMMIT_HISTORY_MAX_AGE_DAYS=180)
    for index in range(5):
        repo.commit("map/x.js", f"const a = {index}", days_ago(30 - index),
                    f"recent change {index}")

    assert read_history(repo.path, "map/x.js") == ["recent change 4",
                                                   "recent change 3"]


# --- how a subject travels -------------------------------------------------

# Nothing is filtered on how a subject is written, so there is no branch to
# exercise per prefix -- `wip:`, `chore:` and a bare sentence all travel the
# same path. What can still go wrong is the transport: subjects are read one
# per line, alongside their timestamp.

def test_a_subject_survives_accents_and_quotes(repo):
    subject = "fix(map): corrige l'affichage — \"tuiles\" & <balises> #1526"
    repo.commit("map/x.js", "const a = 1", days_ago(10), subject)

    assert read_history(repo.path, "map/x.js") == [subject]


def test_only_the_subject_of_a_multi_line_message_is_kept(repo):
    # A commit body would otherwise land in the history as extra entries.
    path = repo.file("map/x.js")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("const a = 1")
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "center on the bbox",
             "-m", "The bbox is\nspread over\nseveral lines.",
             date=days_ago(10))

    assert read_history(repo.path, "map/x.js") == ["center on the bbox"]


def test_a_commit_without_a_message_is_not_reported(repo):
    repo.commit("map/x.js", "const a = 1", days_ago(20), "add the base map")
    repo.file("map/x.js").write_text("const a = 2")
    repo.git("add", "-A")
    repo.git("commit", "-q", "--allow-empty-message", "-m", "",
             date=days_ago(10))

    assert read_history(repo.path, "map/x.js") == ["add the base map"]


def test_a_very_long_subject_is_kept_whole(repo):
    # kdk really has subjects of 200 characters; truncating one would cut it
    # exactly where the explanation is.
    subject = ("fix: When layer service data is initialized from a file data "
               "is not updated when the layer already exists " + "x" * 120)
    repo.commit("map/x.js", "const a = 1", days_ago(10), subject)

    assert read_history(repo.path, "map/x.js") == [subject]


def test_surrounding_whitespace_is_trimmed(repo):
    repo.commit("map/x.js", "const a = 1", days_ago(10), "  add the base map  ")

    assert read_history(repo.path, "map/x.js") == ["add the base map"]


# --- outside a repository --------------------------------------------------

def test_a_directory_that_is_not_a_repository_has_no_history(tmp_path):
    assert read_history(tmp_path, "map/x.js") == []


def test_a_missing_git_binary_is_not_an_error(repo, monkeypatch):
    import ingestion.services.history as module

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(module.subprocess, "run", no_git)

    assert read_history(repo.path, "map/x.js") == []


# --- collecting for a whole workspace --------------------------------------

def test_the_history_is_collected_per_file(repo):
    repo.commit("map/x.js", "const a = 1", days_ago(10), "add the base map")
    repo.commit("map/y.js", "const b = 1", days_ago(5), "add the layer store")

    histories = collect_commit_history(
        [("kdk", "map/x.js"), ("kdk", "map/y.js")], {"kdk": repo.path})

    assert histories == {
        ("kdk", "map/x.js"): ["add the base map"],
        ("kdk", "map/y.js"): ["add the layer store"],
    }


def test_a_file_outside_a_repository_collects_an_empty_history(tmp_path):
    histories = collect_commit_history([("kdk", "map/x.js")], {"kdk": tmp_path})

    assert histories == {("kdk", "map/x.js"): []}


def test_a_repository_that_is_not_on_disk_collects_an_empty_history():
    # A file keyed on a repository the scan did not report is not a crash.
    assert collect_commit_history([("gone", "map/x.js")], {}) == {
        ("gone", "map/x.js"): []}
