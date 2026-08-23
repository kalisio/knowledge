"""The first ingestion: scan, chunk, embed, index, then serve.

One run of the real job over four files -- one per supported type, spread
over two repositories -- checked step by step: what git hands over, what the
chunkers produce, what lands in Qdrant, and what the API gives back.
"""

import pytest

import ingestion.chunkers as chunkers
import ingestion.services.vectordb as vectordb

from helpers import (CODE_COLLECTION, METADATA_COLLECTION, SAMPLES,
                     VECTOR_SIZE, chunks_of, code_points, file_key,
                     history_of, points_by_file, read_sample, requires_qdrant,
                     sha1)

# Everything a stored chunk carries. The first six are the contract of
# issue #6; the rest is what the pipeline needs to maintain the index.
PAYLOAD_KEYS = {"path", "repo", "start_line", "end_line", "content",
                "file_type", "chunk_index", "breadcrumb", "content_sha1",
                "file_sha1"}


# The corpus in place and ingested once.
@pytest.fixture
def ingested(pipeline):
    pipeline.workspace.install_samples()
    assert pipeline.run() == 0
    return pipeline


@pytest.mark.Ingestion
class TestFirstRun:
    # The run walks every step and reports each one.
    @requires_qdrant
    def test_the_run_reports_every_step(self, ingested):
        logs = ingested.logs.text

        assert "collection '%s' created" % METADATA_COLLECTION in logs
        assert "collection '%s' created" % CODE_COLLECTION in logs
        assert "files to process: 4" in logs
        assert "files deleted: 0" in logs
        assert "files to index: 4" in logs
        assert "last_ingestion=none" in logs

    # k-clone is called with the configured organisation and workspace, and
    # only after the collections are ready.
    @requires_qdrant
    def test_the_clone_step_uses_the_configured_workspace(self, ingested):
        assert ingested.clone_calls == [["bash", "k-clone", "kalisio", "apps"]]

    # Both collections exist with the vector size their configuration names.
    @requires_qdrant
    def test_the_collections_are_created_with_the_configured_sizes(
            self, ingested):
        assert (vectordb.get_collection_vector_size(CODE_COLLECTION)
                == VECTOR_SIZE)
        assert vectordb.get_collection_vector_size(METADATA_COLLECTION) == 1

    # All four types are indexed, each under its own repository.
    @requires_qdrant
    def test_every_supported_file_type_is_indexed(self, ingested):
        indexed = set(points_by_file())

        assert indexed == {file_key(path) for path in SAMPLES}
        assert {repository for repository, _ in indexed} == {"kdk", "kano"}

    # Nothing is indexed twice, and nothing is indexed empty.
    @requires_qdrant
    def test_no_chunk_is_empty_or_duplicated(self, ingested):
        points = code_points()

        assert len({point.id for point in points}) == len(points)
        assert all(point.payload["content"].strip() for point in points)

    # The state persisted for the next run: when it ran, and with what.
    @requires_qdrant
    def test_the_indexing_state_is_persisted(self, ingested):
        assert vectordb.get_last_ingestion() is not None
        assert vectordb.get_indexed_config() == {
            "embedding_model": "test-model",
            "chunking_version": chunkers.CHUNKING_VERSION,
        }

    # Documents are embedded in one batch, with the configured batch size.
    @requires_qdrant
    def test_the_documents_are_embedded_as_documents(self, ingested):
        assert ingested.embed_batches == [len(code_points())]
        call = ingested.model.calls[-1]
        assert call.batch_size == 4                  # EMBEDDING_BATCH_SIZE
        assert call.normalize_embeddings is True
        # Documents carry no retrieval instruction -- only queries do.
        assert all("Instruct:" not in text for text in call.sentences)


@pytest.mark.Ingestion
class TestPayloadContract:
    # Every stored chunk carries the full payload the API reads back.
    @requires_qdrant
    def test_every_chunk_carries_the_whole_payload(self, ingested):
        for point in code_points():
            assert set(point.payload) == PAYLOAD_KEYS

    # The file digest is the SHA-1 of the file, shared by all its chunks --
    # that is what lets the next run tell whether the file changed.
    @requires_qdrant
    def test_the_file_digest_is_the_sha1_of_the_file(self, ingested):
        for source_path, sample in SAMPLES.items():
            digests = {chunk["file_sha1"] for chunk in chunks_of(source_path)}
            assert digests == {sha1(read_sample(sample))}

    # The chunk digest is the SHA-1 of the chunk's own text.
    @requires_qdrant
    def test_the_chunk_digest_is_the_sha1_of_the_chunk(self, ingested):
        for point in code_points():
            assert point.payload["content_sha1"] == sha1(point.payload["content"])

    # Chunk indexes are contiguous from zero, per file.
    @requires_qdrant
    def test_chunk_indexes_are_contiguous_per_file(self, ingested):
        for source_path in SAMPLES:
            chunks = chunks_of(source_path)
            assert ([chunk["chunk_index"] for chunk in chunks]
                    == list(range(len(chunks))))

    # file_type is the extension, and source_path is repo-relative -- the
    # repository name lives in its own field, never in the path.
    @requires_qdrant
    def test_the_file_identity_is_split_into_repository_and_path(
            self, ingested):
        for source_path in SAMPLES:
            repository, relative = file_key(source_path)
            for chunk in chunks_of(source_path):
                assert chunk["repo"] == repository
                assert chunk["path"] == relative
                assert chunk["file_type"] == source_path.rsplit(".", 1)[-1]
                assert not chunk["path"].startswith(repository + "/")

    # The commit history is stored once per file, not on each of its chunks.
    @requires_qdrant
    def test_the_commit_history_is_stored_once_per_file(self, ingested):
        for source_path in SAMPLES:
            assert history_of(source_path) == ["feat: import the corpus"]
            for chunk in chunks_of(source_path):
                assert "commit_history" not in chunk

    # The stored vector has the configured dimension.
    @requires_qdrant
    def test_the_stored_vectors_have_the_configured_dimension(self, ingested):
        client = vectordb._get_qdrant_client()
        records, _ = client.scroll(
            collection_name=CODE_COLLECTION, limit=5, with_vectors=True)
        assert all(len(record.vector) == VECTOR_SIZE for record in records)


@pytest.mark.Ingestion
class TestChunkingPerFileType:
    # Markdown is cut on its heading tree, and each chunk states where it
    # comes from, so a retrieved fragment is never context-free.
    @requires_qdrant
    def test_markdown_chunks_carry_their_heading_breadcrumb(self, ingested):
        chunks = chunks_of("kdk/docs/catalog/guide.md")

        breadcrumbs = [chunk["breadcrumb"] for chunk in chunks]
        assert "Layer catalog > Installation" in breadcrumbs
        assert "Layer catalog > Configuration > Zoom levels" in breadcrumbs
        for chunk in chunks:
            assert chunk["content"].startswith("Context: ")
            assert "Source: docs/catalog/guide.md" in chunk["content"]

    # A fenced code block is never split in half: the snippet a developer
    # gets back has to be runnable.
    @requires_qdrant
    def test_a_markdown_code_block_stays_whole(self, ingested):
        chunks = chunks_of("kdk/docs/catalog/guide.md")

        holding = [chunk for chunk in chunks
                   if "import { catalog }" in chunk["content"]]
        assert len(holding) == 1
        assert "app.configure(catalog)" in holding[0]["content"]
        assert holding[0]["content"].count("```") == 2

    # JavaScript chunks name the file and the symbol they belong to.
    @requires_qdrant
    def test_javascript_chunks_are_headed_by_their_symbol(self, ingested):
        chunks = chunks_of("kdk/core/client/geolocation.js")

        for chunk in chunks:
            header = chunk["content"].splitlines()[0]
            assert header.startswith("// core/client/geolocation.js")
            if chunk["breadcrumb"]:
                assert header.endswith(f":: {chunk['breadcrumb']}")
        assert "GeolocationTracker" in {
            chunk["breadcrumb"] for chunk in chunks}

    # A Vue SFC is cut per block, and each block is headed in the syntax of
    # the language it holds.
    @requires_qdrant
    def test_a_vue_component_is_split_per_block(self, ingested):
        chunks = chunks_of("kano/client/components/KLayerList.vue")
        headers = [chunk["content"].splitlines()[0] for chunk in chunks]

        assert any("[template]" in header for header in headers)
        assert any("[script]" in header for header in headers)
        assert any("[style]" in header for header in headers)
        for header in headers:
            if "[script]" in header:
                assert header.startswith("// ")
            else:
                assert header.startswith("<!-- ")
        assert {chunk["breadcrumb"] for chunk in chunks} == {"layer list"}

    # Every part of the component is indexed: the whole template, the whole
    # script, the whole style. Nothing silently falls off the end.
    @requires_qdrant
    def test_a_vue_component_loses_nothing(self, ingested):
        indexed = "\n".join(
            chunk["content"]
            for chunk in chunks_of("kano/client/components/KLayerList.vue"))

        assert "q-item-section" in indexed          # template
        assert "useCatalog" in indexed              # script
        assert "overflow-y" in indexed              # style

    # An i18n catalog is cut per section, so a component's labels stay
    # together and are retrievable by the component name.
    @requires_qdrant
    def test_an_i18n_catalog_is_split_per_section(self, ingested):
        chunks = chunks_of("kano/client/i18n/en.json")

        by_breadcrumb = {chunk["breadcrumb"]: chunk["content"]
                         for chunk in chunks}
        assert {"KLayerList", "KCatalogPanel"} <= set(by_breadcrumb)
        assert "No layer available" in by_breadcrumb["KLayerList"]
        assert "Add a layer" in by_breadcrumb["KCatalogPanel"]
        # Flat top-level labels are grouped in their own chunk.
        assert any("APPLICATION_NAME" in text for text in by_breadcrumb.values())


@pytest.mark.Ingestion
class TestServingTheIngestedCorpus:
    # What was ingested is what the API returns, with its provenance.
    @requires_qdrant
    def test_search_returns_the_indexed_corpus(self, ingested):
        response = ingested.client.post(
            "/search", json={"query": "layer catalog", "top_k": 50})

        assert response.status_code == 200
        results = response.json()
        assert {result["path"] for result in results} == {
            relative for _, relative in map(file_key, SAMPLES)}
        for result in results:
            assert result["repo"] in {"kdk", "kano"}
            assert 0.0 <= result["score"] <= 1.0
            assert result["commit_history"] == ["feat: import the corpus"]

    # Retrieval actually ranks: a question about a subject brings back the
    # file that covers it, not just any indexed chunk.
    @requires_qdrant
    @pytest.mark.parametrize("query,expected", [
        ("how do I restrict the zoom levels of a layer",
         "docs/catalog/guide.md"),
        ("watchPosition clearWatch navigator geolocation tracker",
         "core/client/geolocation.js"),
        ("q-item-section avatar icon list of layers component",
         "client/components/KLayerList.vue"),
        ("translation label toggle the layer visibility",
         "client/i18n/en.json"),
    ])
    def test_a_question_retrieves_the_file_that_answers_it(
            self, ingested, query, expected):
        response = ingested.client.post(
            "/search", json={"query": query, "top_k": 3})

        assert response.status_code == 200
        assert response.json()[0]["path"] == expected

    # The query is embedded asymmetrically: the retrieval instruction is
    # prepended for the query and only for the query.
    @requires_qdrant
    def test_the_query_is_embedded_with_the_retrieval_instruction(
            self, ingested):
        ingested.client.post("/search", json={"query": "zoom", "top_k": 1})

        last = ingested.model.calls[-1]
        assert isinstance(last.sentences, str)
        assert last.sentences.startswith("Instruct: ")
        assert last.sentences.endswith("Query: zoom")

    # The two services are deployed apart and each owns its embedding code,
    # so the asymmetry they must agree on is only observable here: the same
    # text embeds one way as a question and another as a document.
    @requires_qdrant
    def test_a_question_and_a_document_are_embedded_differently(
            self, ingested):
        text = "the catalog stores layer descriptors"
        indexing_call = ingested.model.calls[0]

        ingested.client.post("/search", json={"query": text, "top_k": 1})
        query_call = ingested.model.calls[-1]

        assert all("Instruct:" not in document
                   for document in indexing_call.sentences)
        assert query_call.sentences.startswith("Instruct: ")
        assert query_call.sentences.endswith(text)

    # Both sides must load the model named by the same setting, or a query
    # vector and a chunk vector are not comparable at all.
    @requires_qdrant
    def test_both_services_embed_with_the_same_configured_model(
            self, ingested):
        import api.config as api_config
        import ingestion.config as ingestion_config

        assert (api_config.get_config().embedding_model
                == ingestion_config.get_config().embedding_model)

    # /ask hands the retrieved chunks to the LLM and returns the answer with
    # its sources and the provenance of the model that produced it.
    @requires_qdrant
    def test_ask_answers_from_the_retrieved_context(self, ingested):
        response = ingested.client.post(
            "/ask", json={"question": "how do I install the catalog?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "stub answer"
        assert body["provider"] == "stub"
        assert body["model"] == "stub-model"
        assert body["sources"]

        prompt = ingested.prompts[-1]
        assert "how do I install the catalog?" in prompt
        assert "app.configure(catalog)" in prompt
        # Each block states where it came from, so the model can cite it.
        assert "Source: docs/catalog/guide.md" in prompt
        assert "Breadcrumb: " in prompt
