import pytest

from ingestion.chunkers.javascript import chunk_javascript
from ingestion.chunkers.json import chunk_json
from ingestion.chunkers.markdown import CHUNK_SIZE, chunk_markdown
from ingestion.chunkers.vue import chunk_vue

MARKDOWN = ("# Base map\n\nThe mixin centers the view.\n\n"
            "## Center\n\nPass a bbox.\n")
JAVASCRIPT = "export function center () {\n  return [0, 0]\n}\n"
VUE = ("<template>\n  <div class=\"map\" />\n</template>\n\n"
       "<script>\nexport default { name: 'KMapPanel' }\n</script>\n")
JSON = '{"KMapPanel": {"TITLE": "Map"}, "LABEL": "Layers"}'

# Each chunker with a file it is meant to handle. The JSON chunker only
# indexes a few roles, so its sample sits at a path it recognises.
CHUNKERS = [
    pytest.param(chunk_markdown, "docs/base-map.md", MARKDOWN, id="markdown"),
    pytest.param(chunk_javascript, "map/base.js", JAVASCRIPT, id="javascript"),
    pytest.param(chunk_vue, "map/KMapPanel.vue", VUE, id="vue"),
    pytest.param(chunk_json, "i18n/en.json", JSON, id="json"),
]


# --- the contract every chunker owes chunk_files --------------------------

@pytest.mark.parametrize("chunker, source_path, text", CHUNKERS)
def test_chunker_produces_chunks(chunker, source_path, text):
    assert chunker(text, source_path)


@pytest.mark.parametrize("chunker, source_path, text", CHUNKERS)
def test_chunker_emits_the_expected_shape(chunker, source_path, text):
    for chunk in chunker(text, source_path):
        assert set(chunk) == {"text", "metadata"}
        assert set(chunk["metadata"]) == {
            "source_path", "chunk_index", "breadcrumb"}


@pytest.mark.parametrize("chunker, source_path, text", CHUNKERS)
def test_chunker_echoes_the_source_path(chunker, source_path, text):
    for chunk in chunker(text, source_path):
        assert chunk["metadata"]["source_path"] == source_path


@pytest.mark.parametrize("chunker, source_path, text", CHUNKERS)
def test_chunker_numbers_chunks_from_zero_without_a_gap(
        chunker, source_path, text):
    # payload_id() mixes chunk_index in, so a repeated index would collapse
    # two chunks of the same file onto one entry.
    chunks = chunker(text, source_path)
    assert [chunk["metadata"]["chunk_index"]
            for chunk in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("chunker, source_path, text", CHUNKERS)
def test_chunker_never_emits_blank_text(chunker, source_path, text):
    # A blank chunk still costs an embedding call and can be retrieved.
    for chunk in chunker(text, source_path):
        assert chunk["text"].strip()


@pytest.mark.parametrize("chunker, source_path, text", CHUNKERS)
def test_chunker_leaves_the_file_wide_metadata_to_chunk_files(
        chunker, source_path, text):
    # repository and file_sha1 are stamped by chunk_files(); a chunker filling
    # them in would give the same key two sources of truth.
    for chunk in chunker(text, source_path):
        assert "repository" not in chunk["metadata"]
        assert "file_sha1" not in chunk["metadata"]


@pytest.mark.parametrize("chunker, source_path, text", CHUNKERS)
def test_chunker_returns_nothing_for_an_empty_file(
        chunker, source_path, text):
    assert chunker("", source_path) == []


# --- markdown: headings drive the split and the breadcrumb ----------------

def test_markdown_breadcrumb_follows_the_heading_path():
    chunks = chunk_markdown(MARKDOWN, "docs/base-map.md")
    breadcrumbs = [chunk["metadata"]["breadcrumb"] for chunk in chunks]

    assert "Base map" in breadcrumbs
    assert "Base map > Center" in breadcrumbs


def test_markdown_falls_back_to_the_file_name_without_a_title():
    chunks = chunk_markdown("Just prose, no heading.\n", "docs/base-map.md")

    assert chunks[0]["metadata"]["breadcrumb"] == "base-map"


def test_markdown_splits_prose_under_a_single_heading():
    # Headings alone used to drive the split, so a document with no
    # subheading became one chunk however long it was, and the embedding
    # model silently truncated everything past its input limit.
    text = "# Title\n\n" + "word " * 20000    # 200x CHUNK_SIZE

    chunks = chunk_markdown(text, "docs/base-map.md")

    assert len(chunks) > 1
    assert max(len(chunk["text"]) for chunk in chunks) < CHUNK_SIZE * 2


def test_markdown_keeps_a_fenced_code_block_whole():
    # The size exemption exists for code: a block cut in half indexes
    # fragments that no longer parse, and loses its closing fence.
    text = "# Title\n\n## Snippet\n\n```js\n" + "const a = 1\n" * 200 + "```\n"

    fenced = [chunk for chunk in chunk_markdown(text, "docs/base-map.md")
              if "```" in chunk["text"]]

    assert len(fenced) == 1
    assert len(fenced[0]["text"]) > CHUNK_SIZE * 4
    assert fenced[0]["text"].count("```") == 2


def test_markdown_prefixes_the_breadcrumb_and_the_path():
    # The prefix is what the embedding sees, so a chunk retrieved on its own
    # still says where it came from.
    chunk = chunk_markdown(MARKDOWN, "docs/base-map.md")[0]

    assert chunk["text"].startswith(
        "Context: Base map\nSource: docs/base-map.md")


# --- javascript: chunks are tagged with the enclosing symbol --------------

def test_javascript_breadcrumb_is_the_nearest_top_level_symbol():
    chunk = chunk_javascript(JAVASCRIPT, "map/base.js")[0]

    assert chunk["metadata"]["breadcrumb"] == "center"


def test_javascript_tracks_the_symbol_across_chunks():
    # A chunk is tagged with the symbol in effect where it starts, so a long
    # file hands each part of itself to the right declaration.
    text = ("export function alpha () {\n  return 1\n}\n\n"
            + "// filler\n" * 120
            + "export const beta = () => 2\n"
            + "// filler\n" * 120)
    breadcrumbs = {chunk["metadata"]["breadcrumb"]
                   for chunk in chunk_javascript(text, "map/base.js")}

    assert breadcrumbs == {"alpha", "beta"}


def test_javascript_header_names_the_file_and_the_symbol():
    chunk = chunk_javascript(JAVASCRIPT, "map/base.js")[0]

    assert chunk["text"].startswith("// map/base.js :: center")


# --- vue: one pass per single-file-component block ------------------------

def test_vue_chunks_each_block_of_the_component():
    texts = [chunk["text"] for chunk in chunk_vue(VUE, "map/KMapPanel.vue")]

    assert any("[template]" in text for text in texts)
    assert any("[script]" in text for text in texts)


def test_vue_breadcrumb_is_the_readable_component_name():
    chunk = chunk_vue(VUE, "map/KMapPanel.vue")[0]

    assert chunk["metadata"]["breadcrumb"] == "map panel"


def test_vue_skips_empty_blocks():
    text = VUE + "<style scoped>\n</style>\n"

    assert not any("[style]" in chunk["text"]
                   for chunk in chunk_vue(text, "map/KMapPanel.vue"))


# --- json: only the roles worth indexing ----------------------------------

def test_json_indexes_translations_per_section():
    breadcrumbs = [chunk["metadata"]["breadcrumb"]
                   for chunk in chunk_json(JSON, "i18n/en.json")]

    assert "KMapPanel" in breadcrumbs


def test_json_indexes_a_schema_per_property():
    text = '{"title": "User", "properties": {"email": {"type": "string"}}}'
    breadcrumbs = [chunk["metadata"]["breadcrumb"]
                   for chunk in chunk_json(text, "schemas/user.create.json")]

    assert "email" in breadcrumbs


def test_json_skips_manifests_and_unknown_roles():
    # package.json and fixtures are structure, not knowledge: indexing them
    # would answer questions about dependency pins rather than about code.
    assert chunk_json(JSON, "package.json") == []
    assert chunk_json(JSON, "test/fixtures/sample.json") == []


def test_json_falls_back_to_a_size_split_on_malformed_content():
    # A file that fails to parse is still text somebody wrote; losing it
    # silently would be worse than indexing it plainly.
    assert chunk_json("{not json", "i18n/en.json")
