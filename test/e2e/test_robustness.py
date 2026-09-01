"""Files and configurations that stray from the nominal case.

Every chunker works on the file it was written against. What matters is what
happens on the next file: a component using a nested template, a JSON that
is not a translation catalog, a repository with no commit, a submodule that
was never initialised, a setting typed by hand. Several of these describe
defects the pipeline used to have; they stay as the fence that keeps them
from coming back.
"""

import pytest

import ingestion.clients.vectordb as vectordb

from conftest import base_env
from helpers import (SAMPLES, chunks_of, code_points, file_key,
                     points_by_file, requires_qdrant, sha1)

GUIDE = "kdk/docs/catalog/guide.md"
GEOLOCATION = "kdk/core/client/geolocation.js"

# A component built the way KDK components are: a nested <template> for a
# v-for row, and markup after it.
NESTED_TEMPLATE_VUE = """<template>
  <q-list>
    <template v-for="item in items" :key="item.name">
      <q-item>{{ item.label }}</q-item>
    </template>
    <KLayerFooter class="footer-marker" />
  </q-list>
</template>

<script setup>
const items = []
</script>
"""

# The shape of a KDK client module: an exported object holding long methods,
# followed by short exported helpers. This is where the symbol tagging drifts.
MIXIN_JS = """export const layerMixin = {
  methods: {
    %s
    async refreshLayer (name) {
      const layer = this.getLayerByName(name)
      if (!layer) return
      await this.updateLayer(name, layer)
    }
  }
}

export function toGeoJson (features) {
  return { type: 'FeatureCollection', features }
}

export function toBoundingBox (geojson) {
  return geojson.features.map((feature) => feature.bbox)
}
""" % "\n    ".join(
    f"async step{index} (options) {{\n"
    f"      const result = await this.call{index}(options)\n"
    f"      return result\n    }},"
    for index in range(6))

# One exported function, three variables local to its body, and a body long
# enough to be cut in two: the second half belongs to publicHelper.
LOCAL_VARIABLES_JS = """export function publicHelper (options) {
  const internalCache = new Map()
  let temporaryValue = 0
  var legacyFlag = false
%s
  return { internalCache, temporaryValue, legacyFlag }
}
""" % "\n".join(
    f"  internalCache.set('key{index}', options.value{index} || temporaryValue)"
    for index in range(20))


@pytest.mark.Robustness
class TestVueComponents:
    # A nested <template> is ordinary Vue (v-for rows, named slots). Every
    # part of the component must still be indexed.
    @requires_qdrant
    def test_a_component_with_a_nested_template_is_indexed_whole(
            self, pipeline):
        pipeline.workspace.commit(
            "kano/client/components/KNested.vue", NESTED_TEMPLATE_VUE)
        assert pipeline.run() == 0

        indexed = "\n".join(
            chunk["content"]
            for chunk in chunks_of("kano/client/components/KNested.vue"))
        assert "q-item" in indexed              # before the nested template
        assert "footer-marker" in indexed       # after the nested template

    # The script of such a component is still reachable, whatever happens to
    # the template -- this is what the pipeline does index today.
    @requires_qdrant
    def test_the_script_of_a_nested_template_component_survives(
            self, pipeline):
        pipeline.workspace.commit(
            "kano/client/components/KNested.vue", NESTED_TEMPLATE_VUE)
        assert pipeline.run() == 0

        indexed = "\n".join(
            chunk["content"]
            for chunk in chunks_of("kano/client/components/KNested.vue"))
        assert "const items" in indexed

    # A typed setup block is split with the script splitter all the same.
    @requires_qdrant
    def test_a_typescript_setup_block_is_indexed(self, pipeline):
        pipeline.workspace.commit(
            "kano/client/components/KTyped.vue",
            '<template><div /></template>\n\n'
            '<script setup lang="ts">\n'
            'const count: number = 0\nexport { count }\n</script>\n')
        assert pipeline.run() == 0

        indexed = "\n".join(
            chunk["content"]
            for chunk in chunks_of("kano/client/components/KTyped.vue"))
        assert "count: number" in indexed

    # A component with nothing but blank blocks yields nothing, and that is
    # not an error.
    @requires_qdrant
    def test_a_component_with_empty_blocks_yields_no_chunk(self, pipeline):
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        pipeline.workspace.commit(
            "kano/client/components/KBlank.vue",
            "<template>\n\n</template>\n\n<style>\n\n</style>\n")

        assert pipeline.run() == 0

        assert ("kano", "client/components/KBlank.vue") not in points_by_file()


@pytest.mark.Robustness
class TestJavaScriptSymbols:
    # _TOP_LEVEL_SYMBOL is meant to match top-level declarations. Its `^\\s*`
    # prefix matches any indentation, so a variable declared inside a
    # function body is captured too -- and, being the nearest match, it
    # becomes the symbol the chunk is filed under.
    @requires_qdrant
    def test_a_local_variable_is_not_taken_for_a_top_level_symbol(
            self, pipeline):
        pipeline.workspace.commit(
            "kdk/core/client/helper.js", LOCAL_VARIABLES_JS)
        assert pipeline.run() == 0

        breadcrumbs = {chunk["breadcrumb"]
                       for chunk in chunks_of("kdk/core/client/helper.js")}
        assert breadcrumbs == {"publicHelper"}

    # The symbol in the header and in the breadcrumb is what tells a reader
    # (and the LLM) which function a fragment belongs to. It must match the
    # fragment it is stamped on.
    @requires_qdrant
    def test_a_chunk_is_stamped_with_the_symbol_it_starts_on(self, pipeline):
        pipeline.workspace.commit("kdk/core/client/mixin.js", MIXIN_JS)
        assert pipeline.run() == 0

        for chunk in chunks_of("kdk/core/client/mixin.js"):
            body = chunk["content"]
            if "export function toGeoJson" in body:
                assert chunk["breadcrumb"] == "toGeoJson"

    # Whatever the symbol, the code itself is never lost.
    @requires_qdrant
    def test_no_javascript_content_is_lost(self, pipeline):
        pipeline.workspace.commit("kdk/core/client/mixin.js", MIXIN_JS)
        assert pipeline.run() == 0

        indexed = "\n".join(
            chunk["content"] for chunk in chunks_of("kdk/core/client/mixin.js"))
        assert "toGeoJson" in indexed
        assert "toBoundingBox" in indexed
        assert "refreshLayer" in indexed

    # Windows line endings are content, not a parse error.
    @requires_qdrant
    def test_crlf_line_endings_are_indexed(self, pipeline):
        pipeline.workspace.commit(
            "kdk/core/client/crlf.js",
            "export const first = 1\r\nexport const second = 2\r\n")
        assert pipeline.run() == 0

        indexed = "\n".join(
            chunk["content"] for chunk in chunks_of("kdk/core/client/crlf.js"))
        assert "second" in indexed

    # A file whose bytes are not valid UTF-8 must not stop the run.
    @requires_qdrant
    def test_a_non_utf8_file_does_not_break_the_run(self, pipeline):
        pipeline.workspace.commit_bytes(
            "kdk/core/client/latin1.js",
            b"// caf\xe9 au lait\nexport const cafe = 1\n")

        assert pipeline.run() == 0

        chunks = chunks_of("kdk/core/client/latin1.js")
        assert chunks
        # The digest is taken on the same lossy read the chunker used, so
        # the file is not seen as changed on every later run.
        assert chunks[0]["file_sha1"] == sha1(
            (pipeline.root / "kdk/core/client/latin1.js").read_text(
                encoding="utf-8", errors="ignore"))

    # A rerun over that same file must still be a no-op.
    @requires_qdrant
    def test_a_non_utf8_file_is_stable_across_runs(self, pipeline):
        pipeline.workspace.commit_bytes(
            "kdk/core/client/latin1.js",
            b"// caf\xe9 au lait\nexport const cafe = 1\n")
        assert pipeline.run() == 0
        batches = list(pipeline.embed_batches)

        assert pipeline.run() == 0

        assert pipeline.embed_batches == batches


@pytest.mark.Robustness
class TestMarkdownAndJson:
    # A page with no H1 still gets a breadcrumb: its file name.
    @requires_qdrant
    def test_a_markdown_page_without_a_title_falls_back_to_its_name(
            self, pipeline):
        pipeline.workspace.commit(
            "kdk/docs/notes.md", "Some loose prose with no heading at all.\n")
        assert pipeline.run() == 0

        assert chunks_of("kdk/docs/notes.md")[0]["breadcrumb"] == "notes"

    # A long section with no subheading must be cut: a single oversized
    # chunk would be silently truncated by the embedding model.
    @requires_qdrant
    def test_a_long_unstructured_page_is_split(self, pipeline):
        pipeline.workspace.commit(
            "kdk/docs/long.md", "# Long\n\n" + ("sentence about layers. " * 200))
        assert pipeline.run() == 0

        chunks = chunks_of("kdk/docs/long.md")
        assert len(chunks) > 1
        assert all(len(chunk["content"]) < 1200 for chunk in chunks)

    # An empty page yields nothing and is not an error.
    @requires_qdrant
    def test_an_empty_markdown_page_yields_no_chunk(self, pipeline):
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        pipeline.workspace.commit("kdk/docs/blank.md", "   \n\n  \n")

        assert pipeline.run() == 0

        assert ("kdk", "docs/blank.md") not in points_by_file()

    # A translation catalog that does not parse still gets indexed, on
    # character boundaries -- a broken file is better retrieved than lost.
    @requires_qdrant
    def test_a_malformed_i18n_catalog_falls_back_to_a_plain_split(
            self, pipeline):
        pipeline.workspace.commit(
            "kano/client/i18n/broken.json", '{ "TITLE": "Map",, }')
        assert pipeline.run() == 0

        chunks = chunks_of("kano/client/i18n/broken.json")
        assert chunks
        assert "TITLE" in chunks[0]["content"]

    # Only the JSON roles worth indexing are indexed: a test fixture is not
    # documentation, and must not dilute retrieval.
    @requires_qdrant
    @pytest.mark.parametrize("source_path,indexed", [
        ("kano/client/i18n/en.json", True),
        ("kdk/core/common/schemas/users.create.json", True),
        ("kdk/docs/.vitepress/config.json", True),
        ("kdk/test/data/fixture.json", False),
        ("kdk/core/common/schemas/users.json", False),
    ])
    def test_only_the_indexable_json_roles_are_indexed(
            self, pipeline, source_path, indexed):
        pipeline.workspace.commit(source_path, '{"TITLE": "Map"}')

        assert pipeline.run() == 0

        assert (file_key(source_path) in points_by_file()) is indexed

    # A JSON file that yields no chunk appears in no stored chunk, so its
    # file entry is the only record that it was ever looked at. With it, a
    # later run leaves the file alone; without it, the file came back as new
    # on every run, to be re-read and re-chunked forever.
    @requires_qdrant
    def test_a_never_indexed_json_is_not_reconsidered(self, pipeline):
        pipeline.workspace.commit("kdk/test/data/fixture.json", '{"a": 1}')
        assert pipeline.run() == 0
        pipeline.logs.clear()

        assert pipeline.run() == 0

        assert "files to index: 0" in pipeline.logs.text
        assert pipeline.embed_batches == []

    # Recorded, not forgotten: editing it selects it again, so a file that
    # starts yielding chunks is not skipped forever.
    @requires_qdrant
    def test_an_edited_json_is_reconsidered_even_with_nothing_indexed(
            self, pipeline):
        fixture = "kdk/test/data/fixture.json"
        pipeline.workspace.commit(fixture, '{"a": 1}')
        assert pipeline.run() == 0
        pipeline.workspace.commit(fixture, '{"a": 2}')
        pipeline.logs.clear()

        assert pipeline.run() == 0

        assert "files to index: 1" in pipeline.logs.text


@pytest.mark.Robustness
class TestScanningTheWorkspace:
    # A workspace holds submodules and vendored checkouts whose ".git" is a
    # file, not a directory. Scanning must step over them.
    @requires_qdrant
    def test_an_uninitialised_submodule_does_not_kill_the_run(self, pipeline):
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        vendored = pipeline.root / "vendored"
        vendored.mkdir()
        (vendored / ".git").write_text("gitdir: /nowhere/modules/vendored\n")

        assert pipeline.run() == 0

        assert ("kdk", "docs/catalog/guide.md") in points_by_file()

    # Only files git tracks are indexed: a scratch file left in a working
    # copy is not part of the corpus.
    @requires_qdrant
    def test_an_untracked_file_is_not_indexed(self, pipeline):
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        (pipeline.root / "kdk/docs/scratch.md").write_text("# Scratch\n")

        assert pipeline.run() == 0

        assert ("kdk", "docs/scratch.md") not in points_by_file()

    # A directory that is not a git repository is not a repository.
    @requires_qdrant
    def test_a_directory_that_is_not_a_repository_is_skipped(self, pipeline):
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        loose = pipeline.root / "loose"
        loose.mkdir()
        (loose / "notes.md").write_text("# Loose\n")

        assert pipeline.run() == 0

        assert {repository for repository, _ in points_by_file()} == {"kdk"}

    # The scan filters: build output, dependencies, manifests and generated
    # bundles are all kept out of the corpus.
    @requires_qdrant
    @pytest.mark.parametrize("source_path", [
        "kdk/node_modules/left-pad/index.js",
        "kdk/dist/bundle.js",
        "kdk/coverage/report.md",
        "kdk/.github/workflows/notes.md",
        "kdk/package.json",
        "kdk/CHANGELOG.md",
        "kdk/core/client/app.min.js",
        "kdk/core/client/vendor.bundle.js",
        "kdk/yarn-lock.json",
    ])
    def test_the_scan_filters_keep_a_file_out(self, pipeline, source_path):
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        pipeline.workspace.commit(source_path, "// noise\nconst a = 1\n")

        assert pipeline.run() == 0

        assert file_key(source_path) not in points_by_file()

    # An unsupported extension is left alone.
    @requires_qdrant
    def test_an_unsupported_extension_is_not_indexed(self, pipeline):
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        pipeline.workspace.commit("kdk/core/client/styles.css", ".a { top: 0 }")

        assert pipeline.run() == 0

        assert ("kdk", "core/client/styles.css") not in points_by_file()

    # A file bigger than the limit is skipped, and a file that grows past
    # the limit is dropped from the index rather than left stale.
    @requires_qdrant
    def test_a_file_growing_past_the_size_limit_leaves_the_index(
            self, pipeline, configure):
        configure(**{**base_env(pipeline.development_dir, pipeline.qdrant_url),
                     "MAX_FILE_SIZE": 400})
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        pipeline.workspace.commit(
            "kdk/docs/small.md", "# Small\n\nStill under the limit.\n")
        assert pipeline.run() == 0
        assert ("kdk", "docs/small.md") in points_by_file()

        pipeline.workspace.commit(
            "kdk/docs/small.md", "# Small\n\n" + ("padding " * 200))
        assert pipeline.run() == 0

        assert ("kdk", "docs/small.md") not in points_by_file()

    # A path with spaces and accents is a path like any other.
    @requires_qdrant
    def test_a_path_with_spaces_and_accents_is_indexed(self, pipeline):
        source_path = "kdk/docs/guide des données.md"
        pipeline.workspace.commit(
            source_path, "# Données\n\nLes données géospatiales.\n")

        assert pipeline.run() == 0

        assert file_key(source_path) in points_by_file()

    # A repository with no commit yet has no tracked file and no history.
    @requires_qdrant
    def test_a_repository_without_any_commit_is_harmless(self, pipeline):
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")
        empty = pipeline.root / "fresh"
        empty.mkdir()
        pipeline.workspace.git(empty, "init", "-q", "-b", "main")

        assert pipeline.run() == 0

        assert {repository for repository, _ in points_by_file()} == {"kdk"}

    # The same relative path in two repositories is two different files.
    @requires_qdrant
    def test_the_same_path_in_two_repositories_stays_distinct(self, pipeline):
        pipeline.workspace.commit("kdk/docs/index.md", "# KDK\n\nCore.\n")
        pipeline.workspace.commit("kano/docs/index.md", "# Kano\n\nMaps.\n")

        assert pipeline.run() == 0

        indexed = points_by_file()
        assert ("kdk", "docs/index.md") in indexed
        assert ("kano", "docs/index.md") in indexed
        assert len({point.id for points in indexed.values()
                    for point in points}) == len(code_points())

    # Two identical files in two repositories share their content but not
    # their identity: both must be indexed, and deleting one leaves the
    # other alone.
    @requires_qdrant
    def test_identical_files_in_two_repositories_are_indexed_separately(
            self, pipeline):
        same = "# Shared\n\nExactly the same prose.\n"
        pipeline.workspace.commit("kdk/docs/shared.md", same)
        pipeline.workspace.commit("kano/docs/shared.md", same)
        assert pipeline.run() == 0

        pipeline.workspace.remove("kano/docs/shared.md")
        assert pipeline.run() == 0

        indexed = points_by_file()
        assert ("kdk", "docs/shared.md") in indexed
        assert ("kano", "docs/shared.md") not in indexed


@pytest.mark.Robustness
class TestConfiguration:
    # Restricting the supported extensions restricts the corpus.
    @requires_qdrant
    def test_the_supported_extensions_are_honoured(self, pipeline, configure):
        configure(**{**base_env(pipeline.development_dir, pipeline.qdrant_url),
                     "SUPPORTED_FILE_EXTENSIONS": "md"})
        pipeline.workspace.install_samples()

        assert pipeline.run() == 0

        assert set(points_by_file()) == {("kdk", "docs/catalog/guide.md")}

    # A file name added to the ignore list stops being indexed, and the
    # chunks it already had are removed.
    @requires_qdrant
    def test_a_newly_ignored_file_is_removed_from_the_index(
            self, pipeline, configure):
        pipeline.workspace.install_samples()
        assert pipeline.run() == 0
        assert ("kdk", "docs/catalog/guide.md") in points_by_file()

        configure(**{**base_env(pipeline.development_dir, pipeline.qdrant_url),
                     "IGNORED_FILENAMES": "package.json,guide.md"})
        assert pipeline.run() == 0

        assert ("kdk", "docs/catalog/guide.md") not in points_by_file()
        assert len(points_by_file()) == len(SAMPLES) - 1

    # LOG_LEVEL raises the floor: at WARNING the step-by-step INFO lines are
    # gone. The entry point is what applies it, so this goes through it.
    @requires_qdrant
    def test_the_log_level_is_honoured(self, pipeline, configure):
        configure(**{**base_env(pipeline.development_dir, pipeline.qdrant_url),
                     "LOG_LEVEL": "WARNING"})
        pipeline.workspace.commit(GUIDE, "# Guide\n\nProse.\n")

        assert pipeline.run_from_cli() == 0

        assert "files to process" not in pipeline.logs.text

    # A vector size that does not match the model destroys the collection
    # before the mismatch is discovered: the index is emptied and the run
    # fails. A hand-typed setting is enough to lose the corpus.
    @requires_qdrant
    def test_a_vector_size_that_does_not_match_the_model_empties_the_index(
            self, pipeline, configure):
        pipeline.workspace.install_samples()
        assert pipeline.run() == 0
        assert code_points()

        configure(**{**base_env(pipeline.development_dir, pipeline.qdrant_url),
                     "QDRANT_VECTOR_SIZE_COLLECTION_CODE": 999})
        with pytest.raises(Exception) as failure:
            pipeline.run()

        assert "dimension" in str(failure.value).lower()
        assert code_points() == []               # the corpus is gone


@pytest.mark.Robustness
class TestServingEdgeCases:
    # The corpus in place, ingested once.
    @pytest.fixture
    def ingested(self, pipeline):
        pipeline.workspace.install_samples()
        assert pipeline.run() == 0
        return pipeline

    # Asking for more than the corpus holds returns the corpus, not an error.
    @requires_qdrant
    def test_a_top_k_larger_than_the_corpus_is_fine(self, ingested):
        response = ingested.client.post(
            "/search", json={"query": "layer", "top_k": 50})

        assert response.status_code == 200
        assert len(response.json()) == len(code_points())

    # A query that matches nothing in particular still returns the closest
    # chunks: retrieval has no relevance floor, so the LLM is the one that
    # has to say "the context does not cover this".
    @requires_qdrant
    def test_an_off_topic_query_still_returns_the_closest_chunks(
            self, ingested):
        response = ingested.client.post(
            "/search", json={"query": "recette de la tarte tatin", "top_k": 3})

        assert response.status_code == 200
        assert len(response.json()) == 3

    # A question in French, with accents and punctuation, is embedded and
    # answered like any other.
    @requires_qdrant
    def test_a_question_with_accents_is_served(self, ingested):
        response = ingested.client.post(
            "/ask", json={"question": "Comment régler les niveaux de zoom ?"})

        assert response.status_code == 200
        assert "Comment régler les niveaux de zoom ?" in ingested.prompts[-1]

    # The context budget is a hard limit: whole blocks are dropped, never
    # half a chunk, and the question survives.
    @requires_qdrant
    def test_the_context_budget_truncates_the_prompt(
            self, ingested, configure):
        configure(**{**base_env(ingested.development_dir, ingested.qdrant_url),
                     "MAX_CONTEXT_CHARS": 400, "TOP_K": 6})

        response = ingested.client.post(
            "/ask", json={"question": "how do I install the catalog?"})

        assert response.status_code == 200
        prompt = ingested.prompts[-1]
        assert prompt.count("[Chunk ") <= 2
        assert "how do I install the catalog?" in prompt

    # A budget smaller than a single chunk yields an empty context rather
    # than a half-truncated one, and the call still goes through.
    @requires_qdrant
    def test_a_budget_below_one_chunk_yields_an_empty_context(
            self, ingested, configure):
        configure(**{**base_env(ingested.development_dir, ingested.qdrant_url),
                     "MAX_CONTEXT_CHARS": 10})

        response = ingested.client.post(
            "/ask", json={"question": "how do I install the catalog?"})

        assert response.status_code == 200
        assert "[Chunk " not in ingested.prompts[-1]
        assert response.json()["sources"]        # the sources are still cited

    # An index emptied under a running API is reported as "not ingested",
    # not as "no match".
    @requires_qdrant
    def test_an_emptied_index_returns_the_503_hint(self, ingested):
        for repository, source_path in list(points_by_file()):
            vectordb.delete_file(repository, source_path)

        response = ingested.client.post(
            "/search", json={"query": "layer", "top_k": 5})

        assert response.status_code == 503
        assert "python -m ingestion.bin" in response.json()["detail"]
