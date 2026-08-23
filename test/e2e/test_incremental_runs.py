"""Running the job again: what changed, and only what changed.

A re-run is where a pipeline quietly goes wrong -- stale chunks left behind,
a whole corpus re-embedded for nothing, a deleted file still answering
queries. Each test here edits the workspace, runs the job again, and checks
the index against what actually moved.
"""

import pytest

import ingestion.chunkers as chunkers
import ingestion.services.vectordb as vectordb

from conftest import base_env
from helpers import (CODE_COLLECTION, FILES_COLLECTION, METADATA_COLLECTION,
                     SAMPLES,
                     VECTOR_SIZE, chunks_of, code_points, file_key,
                     days_ago, history_of, points_by_file, read_sample,
                     requires_qdrant, sha1)

GUIDE = "kdk/docs/catalog/guide.md"
GEOLOCATION = "kdk/core/client/geolocation.js"
COMPONENT = "kano/client/components/KLayerList.vue"
I18N = "kano/client/i18n/en.json"

NEW_JS = """// Rewritten module.
export function distanceInMeters (from, to) {
  return Math.hypot(to[0] - from[0], to[1] - from[1]) * 111320
}
"""

NEW_MD = """# Layer catalog

## Retention

Descriptors are kept until the project that owns them is deleted.
"""


# The corpus ingested once; every test here starts from that state.
@pytest.fixture
def ingested(pipeline):
    pipeline.workspace.install_samples()
    assert pipeline.run() == 0
    pipeline.logs.clear()
    return pipeline


# {(repository, path): {point ids}} -- the fingerprint of the whole index.
def ids_by_file():
    return {key: {point.id for point in points}
            for key, points in points_by_file().items()}


@pytest.mark.Incremental
class TestNothingChanged:
    # A second run over an untouched workspace must be a no-op: no
    # re-embedding, and the exact same points.
    @requires_qdrant
    def test_a_rerun_changes_nothing_and_embeds_nothing(self, ingested):
        before = ids_by_file()
        batches = list(ingested.embed_batches)

        assert ingested.run() == 0

        assert ids_by_file() == before
        assert ingested.embed_batches == batches
        assert "files to index: 0" in ingested.logs.text

    # The run still records that it happened, so the next one knows.
    @requires_qdrant
    def test_a_rerun_moves_the_last_ingestion_forward(self, ingested):
        first = vectordb.get_last_ingestion()

        assert ingested.run() == 0

        assert vectordb.get_last_ingestion() > first

    # The previous timestamp is reported at startup of the next run.
    @requires_qdrant
    def test_a_rerun_reports_the_previous_ingestion(self, ingested):
        assert ingested.run() == 0

        assert "last_ingestion=none" not in ingested.logs.text
        assert "(4 indexed points)" not in ingested.logs.text  # 13 chunks
        assert "indexed points" in ingested.logs.text


@pytest.mark.Incremental
class TestOneFileChanged:
    # Editing one file re-indexes that file and nothing else.
    @requires_qdrant
    def test_only_the_edited_file_is_reindexed(self, ingested):
        before = ids_by_file()

        ingested.workspace.commit(GEOLOCATION, NEW_JS, "refactor: rewrite")
        assert ingested.run() == 0

        after = ids_by_file()
        edited = file_key(GEOLOCATION)
        assert after[edited].isdisjoint(before[edited])
        for key in set(before) - {edited}:
            assert after[key] == before[key]
        assert "files to index: 1" in ingested.logs.text

    # The new digest is the digest of the new content, on every chunk.
    @requires_qdrant
    def test_the_file_digest_follows_the_new_content(self, ingested):
        old_digest = sha1(read_sample("geolocation.js"))

        ingested.workspace.commit(GEOLOCATION, NEW_JS, "refactor: rewrite")
        assert ingested.run() == 0

        digests = {chunk["file_sha1"] for chunk in chunks_of(GEOLOCATION)}
        assert digests == {sha1(NEW_JS)}
        assert old_digest not in digests

    # No stale chunk survives: the old symbols are gone from the index.
    @requires_qdrant
    def test_the_previous_content_is_gone_from_the_index(self, ingested):
        ingested.workspace.commit(GEOLOCATION, NEW_JS, "refactor: rewrite")
        assert ingested.run() == 0

        indexed = "\n".join(point.payload["content"] for point in code_points())
        assert "GeolocationTracker" not in indexed
        assert "distanceInMeters" in indexed

    # Only the edited file's chunks are re-embedded -- an unchanged corpus
    # must not be paid for again.
    @requires_qdrant
    def test_only_the_edited_chunks_are_reembedded(self, ingested):
        ingested.workspace.commit(GEOLOCATION, NEW_JS, "refactor: rewrite")
        assert ingested.run() == 0

        assert ingested.embed_batches[-1] == len(chunks_of(GEOLOCATION))

    # The API serves the new content, and no longer the old one.
    @requires_qdrant
    def test_the_api_serves_the_new_content(self, ingested):
        ingested.workspace.commit(GEOLOCATION, NEW_JS, "refactor: rewrite")
        assert ingested.run() == 0

        response = ingested.client.post(
            "/search", json={"query": "distanceInMeters hypot", "top_k": 10})

        results = response.json()
        assert results[0]["path"] == "core/client/geolocation.js"
        assert "distanceInMeters" in results[0]["content"]
        assert all("GeolocationTracker" not in result["content"]
                   for result in results)

    # The commit that changed the file is what the chunk now reports,
    # newest first.
    @requires_qdrant
    def test_the_commit_history_follows_the_file(self, ingested):
        ingested.workspace.commit(
            GEOLOCATION, NEW_JS, "refactor(geo): rewrite the helpers")
        assert ingested.run() == 0

        assert history_of(GEOLOCATION) == [
            "refactor(geo): rewrite the helpers", "feat: import the corpus"]

    # Only the most recent commits are kept, newest first. The depth is
    # capped because this history is copied onto every chunk of the file.
    @requires_qdrant
    def test_the_commit_history_is_capped_at_the_configured_depth(
            self, ingested, configure):
        configure(**{**base_env(ingested.development_dir, ingested.qdrant_url),
                     "COMMIT_HISTORY_DEPTH": 4})
        for index in range(6):
            ingested.workspace.commit(
                GEOLOCATION, NEW_JS + f"// edit {index}\n",
                f"chore: edit {index}")
        assert ingested.run() == 0

        history = history_of(GEOLOCATION)
        assert len(history) == 4
        assert history[0] == "chore: edit 5"

    # However many chunks a file has, its history is stored once and served
    # with every one of them.
    @requires_qdrant
    def test_a_file_has_one_history_whatever_its_chunk_count(self, ingested):
        assert len(chunks_of(GUIDE)) > 1

        response = ingested.client.post(
            "/search", json={"query": "layer catalog", "top_k": 50})
        served = [result["commit_history"] for result in response.json()
                  if result["path"] == "docs/catalog/guide.md"]

        assert len(served) == len(chunks_of(GUIDE))
        assert all(history == ["feat: import the corpus"]
                   for history in served)

    # Commit subjects are stored verbatim, whatever they look like: prefixed
    # or not, misspelled, chores, version bumps, accents and quotes.
    @requires_qdrant
    @pytest.mark.parametrize("subject", [
        "fix(map): corrige l'affichage des tuiles",
        'feat: add "quoted" support & <angle> brackets',
        "wip:RAG pipeline:general structure #6 - test(api): cover the budget",
        "refactore: rename the helper — accents éàü included",
        "updated the loading indicator without any prefix",
        "chore: bumped kano to 2.8.0",
        "2.8.0",
    ])
    def test_a_commit_subject_is_stored_verbatim(self, ingested, subject):
        ingested.workspace.commit(GEOLOCATION, NEW_JS, subject)
        assert ingested.run() == 0

        assert history_of(GEOLOCATION)[0] == subject


@pytest.mark.Incremental
class TestTwoFilesChanged:
    # Editing two files re-indexes exactly those two, across repositories.
    @requires_qdrant
    def test_exactly_the_two_edited_files_are_reindexed(self, ingested):
        before = ids_by_file()

        ingested.workspace.commit(GEOLOCATION, NEW_JS, "refactor: rewrite")
        ingested.workspace.commit(GUIDE, NEW_MD, "docs: rewrite the guide")
        assert ingested.run() == 0

        after = ids_by_file()
        edited = {file_key(GEOLOCATION), file_key(GUIDE)}
        for key in edited:
            assert after[key].isdisjoint(before[key])
        for key in set(before) - edited:
            assert after[key] == before[key]
        assert "files to index: 2" in ingested.logs.text

    # Both new digests are stored, and both new contents are retrievable.
    @requires_qdrant
    def test_both_edits_are_visible_through_the_api(self, ingested):
        ingested.workspace.commit(GEOLOCATION, NEW_JS, "refactor: rewrite")
        ingested.workspace.commit(GUIDE, NEW_MD, "docs: rewrite the guide")
        assert ingested.run() == 0

        assert ({chunk["file_sha1"] for chunk in chunks_of(GEOLOCATION)}
                == {sha1(NEW_JS)})
        assert ({chunk["file_sha1"] for chunk in chunks_of(GUIDE)}
                == {sha1(NEW_MD)})

        response = ingested.client.post(
            "/search", json={"query": "descriptors are kept until the project",
                             "top_k": 5})
        assert response.json()[0]["path"] == "docs/catalog/guide.md"
        assert "Retention" in response.json()[0]["content"]

    # A rewritten markdown file loses the headings it no longer has.
    @requires_qdrant
    def test_a_rewritten_file_loses_its_old_structure(self, ingested):
        ingested.workspace.commit(GUIDE, NEW_MD, "docs: rewrite the guide")
        assert ingested.run() == 0

        breadcrumbs = {chunk["breadcrumb"] for chunk in chunks_of(GUIDE)}
        assert breadcrumbs == {"Layer catalog", "Layer catalog > Retention"}

    # Editing two files re-embeds the chunks of those two files, no more.
    @requires_qdrant
    def test_only_the_two_files_are_reembedded(self, ingested):
        ingested.workspace.commit(GEOLOCATION, NEW_JS, "refactor: rewrite")
        ingested.workspace.commit(GUIDE, NEW_MD, "docs: rewrite the guide")
        assert ingested.run() == 0

        assert ingested.embed_batches[-1] == (
            len(chunks_of(GEOLOCATION)) + len(chunks_of(GUIDE)))


@pytest.mark.Incremental
class TestFilesRemoved:
    # A deleted file loses every chunk it had; the rest of the index stands.
    @requires_qdrant
    def test_a_deleted_file_is_dropped_from_the_index(self, ingested):
        before = ids_by_file()

        ingested.workspace.remove(COMPONENT, "chore: drop the component")
        assert ingested.run() == 0

        after = ids_by_file()
        assert file_key(COMPONENT) not in after
        for key in set(before) - {file_key(COMPONENT)}:
            assert after[key] == before[key]
        assert "files deleted: 1" in ingested.logs.text

    # A deleted file stops answering queries -- an assistant citing a file
    # that no longer exists is worse than one that finds nothing.
    @requires_qdrant
    def test_a_deleted_file_stops_being_served(self, ingested):
        ingested.workspace.remove(COMPONENT, "chore: drop the component")
        assert ingested.run() == 0

        response = ingested.client.post(
            "/search", json={"query": "q-item-section avatar", "top_k": 50})

        served = {result["path"]
                  for result in response.json()}
        assert "client/components/KLayerList.vue" not in served

    # Deleting a file costs no embedding at all.
    @requires_qdrant
    def test_a_deletion_embeds_nothing(self, ingested):
        batches = list(ingested.embed_batches)

        ingested.workspace.remove(COMPONENT, "chore: drop the component")
        assert ingested.run() == 0

        assert ingested.embed_batches == batches

    # A renamed file moves in the index: the old path is gone, the new one
    # is indexed, and the history follows the rename.
    @requires_qdrant
    def test_a_renamed_file_moves_in_the_index(self, ingested):
        repo = ingested.root / "kano"
        ingested.workspace.git(
            repo, "mv", "client/components/KLayerList.vue",
            "client/components/KLayersList.vue")
        ingested.workspace.git(repo, "commit", "-q", "-m", "refactor: rename")

        assert ingested.run() == 0

        indexed = set(points_by_file())
        assert ("kano", "client/components/KLayerList.vue") not in indexed
        assert ("kano", "client/components/KLayersList.vue") in indexed
        assert history_of("kano/client/components/KLayersList.vue") == [
            "refactor: rename", "feat: import the corpus"]

    # Emptying a file is not the same as deleting it: it yields no chunk,
    # and its old chunks must still go.
    @requires_qdrant
    def test_an_emptied_file_loses_its_chunks(self, ingested):
        ingested.workspace.commit(GEOLOCATION, "", "chore: empty the module")
        assert ingested.run() == 0

        assert file_key(GEOLOCATION) not in points_by_file()
        assert len(points_by_file()) == len(SAMPLES) - 1


@pytest.mark.Incremental
class TestTheCommitHistoryWindow:
    # The history lives in its own collection, one entry per file.
    @requires_qdrant
    def test_the_history_is_stored_in_its_own_collection(self, ingested):
        assert vectordb.check_collection_exists(FILES_COLLECTION)

        client = vectordb._get_qdrant_client()
        entries, _ = client.scroll(collection_name=FILES_COLLECTION,
                                   limit=100, with_payload=True)
        assert {(entry.payload["repo"], entry.payload["path"])
                for entry in entries} == {file_key(path) for path in SAMPLES}

    # A commit that falls out of the window is dropped on the next run, even
    # though the file itself did not change.
    @requires_qdrant
    def test_a_commit_leaving_the_window_is_dropped(self, pipeline, configure):
        pipeline.workspace.install_samples(
            "feat: import the corpus", date=days_ago(400))
        pipeline.workspace.commit(
            GEOLOCATION, NEW_JS, "fix(geo): the one that matters",
            date=days_ago(2))
        assert pipeline.run() == 0
        # The old commit is held above the window by the floor.
        assert history_of(GEOLOCATION) == [
            "fix(geo): the one that matters", "feat: import the corpus"]

        # Nothing changes on disk; only the window and its floor narrow.
        configure(**{**base_env(pipeline.development_dir, pipeline.qdrant_url),
                     "COMMIT_HISTORY_MAX_AGE_DAYS": 30,
                     "COMMIT_HISTORY_MIN_COMMITS": 1})
        assert pipeline.run() == 0

        assert history_of(GEOLOCATION) == ["fix(geo): the one that matters"]

    # A stable file keeps a history: the floor holds it above the window.
    @requires_qdrant
    def test_a_file_nobody_touches_keeps_its_history(self, ingested,
                                                     configure):
        configure(**{**base_env(ingested.development_dir, ingested.qdrant_url),
                     "COMMIT_HISTORY_MAX_AGE_DAYS": 1,
                     "COMMIT_HISTORY_MIN_COMMITS": 1})

        assert ingested.run() == 0

        assert history_of(GUIDE) == ["feat: import the corpus"]

    # A new commit joins the window without the file being re-embedded for
    # it: the history is refreshed on every run.
    @requires_qdrant
    def test_a_new_commit_joins_the_history(self, ingested):
        ingested.workspace.commit(
            GEOLOCATION, NEW_JS, "fix(geo): a later fix")

        assert ingested.run() == 0

        assert history_of(GEOLOCATION) == [
            "fix(geo): a later fix", "feat: import the corpus"]

    # A deleted file takes its history entry with it.
    @requires_qdrant
    def test_a_deleted_file_loses_its_history_entry(self, ingested):
        assert history_of(COMPONENT)

        ingested.workspace.remove(COMPONENT, "chore: drop the component")
        assert ingested.run() == 0

        assert history_of(COMPONENT) == []


@pytest.mark.Incremental
class TestIndexingConfigChanged:
    # Changing the embedding model invalidates every stored vector: the
    # whole corpus is re-embedded, and the new model is recorded.
    @requires_qdrant
    def test_a_new_embedding_model_forces_a_full_reindex(
            self, ingested, configure):
        indexed_chunks = len(code_points())

        configure(**{**base_env(ingested.development_dir, ingested.qdrant_url),
                     "EMBEDDING_MODEL": "another-model"})
        assert ingested.run() == 0

        assert ingested.embed_batches[-1] == indexed_chunks
        assert vectordb.get_indexed_config() == {
            "embedding_model": "another-model",
            "chunking_version": chunkers.CHUNKING_VERSION}
        assert "indexing config changed" in ingested.logs.text

    # Bumping the chunking version does the same: the stored chunks were cut
    # by rules that no longer apply.
    @requires_qdrant
    def test_a_new_chunking_version_forces_a_full_reindex(
            self, ingested, monkeypatch):
        indexed_chunks = len(code_points())
        monkeypatch.setattr(chunkers, "CHUNKING_VERSION",
                            chunkers.CHUNKING_VERSION + 1)

        assert ingested.run() == 0

        assert ingested.embed_batches[-1] == indexed_chunks
        assert (vectordb.get_indexed_config()["chunking_version"]
                == chunkers.CHUNKING_VERSION)

    # Switching to a model of another dimension cannot reuse the collection:
    # it is recreated at the new size and refilled from scratch.
    @requires_qdrant
    def test_a_new_vector_size_recreates_the_collection(
            self, ingested, configure, monkeypatch):
        import ingestion.main as ingestion_main
        wider = VECTOR_SIZE * 2
        configure(**{
            **base_env(ingested.development_dir, ingested.qdrant_url),
            "QDRANT_VECTOR_SIZE_COLLECTION_CODE": wider,
            "EMBEDDING_MODEL": "wider-model"})
        monkeypatch.setattr(
            ingestion_main.embeddings, "encode_batch",
            lambda texts: [[1.0] + [0.0] * (wider - 1) for _ in texts])

        assert ingested.run() == 0

        assert vectordb.get_collection_vector_size(CODE_COLLECTION) == wider
        assert len(points_by_file()) == len(SAMPLES)


@pytest.mark.Incremental
class TestRecoveringFromAPartialRun:
    # A run killed before its last step leaves no bookkeeping behind; the
    # next run must reuse the index it finds rather than rebuild it.
    @requires_qdrant
    def test_lost_bookkeeping_does_not_wipe_the_index(self, ingested):
        before = ids_by_file()
        batches = list(ingested.embed_batches)
        vectordb.remove_collection(METADATA_COLLECTION)

        assert ingested.run() == 0

        assert ids_by_file() == before
        assert ingested.embed_batches == batches

    # If the code collection itself is gone, the digests are gone with it,
    # so the whole corpus is re-indexed -- and served again.
    @requires_qdrant
    def test_a_lost_code_collection_is_rebuilt(self, ingested):
        vectordb.remove_collection(CODE_COLLECTION)

        assert ingested.run() == 0

        assert len(points_by_file()) == len(SAMPLES)
        response = ingested.client.post(
            "/search", json={"query": "layer catalog", "top_k": 50})
        assert len(response.json()) == len(code_points())

    # A failing clone aborts before anything is touched, exits non-zero, and
    # says which command failed with which code.
    @requires_qdrant
    def test_a_failed_clone_leaves_the_index_untouched(
            self, ingested, monkeypatch):
        import subprocess

        import ingestion.main as ingestion_main
        before = ids_by_file()
        last_ingestion = vectordb.get_last_ingestion()

        def failing_clone(command, *args, **kwargs):
            raise subprocess.CalledProcessError(returncode=3, cmd=command)

        monkeypatch.setattr(ingestion_main.subprocess, "run", failing_clone)

        assert ingested.run() == 1

        assert ids_by_file() == before
        assert vectordb.get_last_ingestion() == last_ingestion
        assert "k-clone kalisio apps failed" in ingested.logs.text
        assert "exit code 3" in ingested.logs.text
