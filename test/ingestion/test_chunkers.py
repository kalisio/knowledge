"""Each chunker on its own, one behaviour per test.

Three layers here: the contract every chunker owes chunk_files, the line
range every chunk must carry, and then what is specific to each file type.
The end-to-end suite checks the same chunkers through the whole pipeline;
this is where a failure says which chunker, and which rule, broke.
"""

import pytest

from ingestion.chunkers.javascript import chunk_javascript
from ingestion.chunkers.json import chunk_json
from ingestion.chunkers.markdown import chunk_markdown
from ingestion.chunkers.vue import chunk_vue
from ingestion.config import get_config

MARKDOWN = ("# Base map\n\nThe mixin centers the view.\n\n"
            "## Center\n\nPass a bbox.\n")
JAVASCRIPT = "export function center () {\n  return [0, 0]\n}\n"
VUE = ("<template>\n  <div class=\"map\" />\n</template>\n\n"
       "<script>\nexport default { name: 'KMapPanel' }\n</script>\n")
JSON = '{\n  "KMapPanel": {"TITLE": "Map"},\n  "LABEL": "Layers"\n}'

# Each chunker with a file it is meant to handle. The JSON chunker only
# indexes a few roles, so its sample sits at a path it recognises.
CHUNKERS = [
    pytest.param(chunk_markdown, "docs/base-map.md", MARKDOWN, id="markdown"),
    pytest.param(chunk_javascript, "map/base.js", JAVASCRIPT, id="javascript"),
    pytest.param(chunk_vue, "map/KMapPanel.vue", VUE, id="vue"),
    pytest.param(chunk_json, "i18n/en.json", JSON, id="json"),
]


# --- the contract every chunker owes chunk_files ---------------------------

@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_chunker_produces_chunks(chunker, path, text):
    assert chunker(text, path)


@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_chunker_emits_the_expected_shape(chunker, path, text):
    for chunk in chunker(text, path):
        assert set(chunk) == {"text", "metadata"}
        assert set(chunk["metadata"]) == {
            "path", "chunk_index", "breadcrumb", "start_line", "end_line"}


@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_chunker_echoes_the_path(chunker, path, text):
    for chunk in chunker(text, path):
        assert chunk["metadata"]["path"] == path


@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_chunker_numbers_chunks_from_zero_without_a_gap(chunker, path, text):
    chunks = chunker(text, path)

    assert ([chunk["metadata"]["chunk_index"] for chunk in chunks]
            == list(range(len(chunks))))


@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_chunker_never_emits_blank_text(chunker, path, text):
    for chunk in chunker(text, path):
        assert chunk["text"].strip()


@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_chunker_leaves_the_file_wide_metadata_to_chunk_files(
        chunker, path, text):
    # repo and file_sha1 are stamped by chunk_files, which is the only caller
    # that knows the workspace the file came from.
    for chunk in chunker(text, path):
        assert "repo" not in chunk["metadata"]
        assert "file_sha1" not in chunk["metadata"]


@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_chunker_returns_nothing_for_an_empty_file(chunker, path, text):
    assert chunker("", path) == []


# --- the line range every chunk carries ------------------------------------

@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_the_line_range_is_within_the_file(chunker, path, text):
    last_line = len(text.splitlines())

    for chunk in chunker(text, path):
        start = chunk["metadata"]["start_line"]
        end = chunk["metadata"]["end_line"]
        assert 1 <= start <= end <= last_line


@pytest.mark.parametrize("chunker, path, text", CHUNKERS)
def test_the_line_range_starts_where_the_previous_chunk_started_or_later(
        chunker, path, text):
    # Chunks come out in file order. They may overlap, but a chunk never
    # starts above the one before it -- that is the drift that used to put
    # chunks under the wrong symbol.
    starts = [chunk["metadata"]["start_line"]
              for chunk in chunker(text, path)]

    assert starts == sorted(starts)


# --- markdown --------------------------------------------------------------

def test_markdown_breadcrumb_follows_the_heading_path():
    text = "# Map\n\nIntro.\n\n## Layers\n\nAbout.\n\n### Base\n\nDetail.\n"

    breadcrumbs = [chunk["metadata"]["breadcrumb"]
                   for chunk in chunk_markdown(text, "docs/map.md")]

    assert breadcrumbs == ["Map", "Map > Layers", "Map > Layers > Base"]


def test_markdown_falls_back_to_the_file_name_without_a_title():
    chunks = chunk_markdown("Loose prose.\n", "docs/notes.md")

    assert chunks[0]["metadata"]["breadcrumb"] == "notes"


def test_markdown_splits_prose_under_a_single_heading():
    size = get_config().chunk_size
    text = "# Long\n\n" + ("sentence about layers. " * 200)

    chunks = chunk_markdown(text, "docs/long.md")

    assert len(chunks) > 1
    assert all(len(chunk["text"]) < size * 2 for chunk in chunks)


def test_markdown_keeps_a_fenced_code_block_whole():
    size = get_config().chunk_size
    code = "\n".join(f"const value{index} = {index}" for index in range(60))
    text = f"# Snippet\n\n```js\n{code}\n```\n"

    chunks = chunk_markdown(text, "docs/snippet.md")

    holding = [chunk for chunk in chunks if "const value0" in chunk["text"]]
    assert len(holding) == 1
    assert "const value59" in holding[0]["text"]
    assert len(holding[0]["text"]) > size


def test_markdown_prefixes_the_breadcrumb_and_the_path():
    chunk = chunk_markdown(MARKDOWN, "docs/base-map.md")[0]

    assert chunk["text"].startswith(
        "Context: Base map\nSource: docs/base-map.md\n\n")


def test_markdown_lines_point_at_the_heading_of_the_section():
    text = "# Map\n\nIntro.\n\n## Layers\n\nAbout.\n"
    lines = text.splitlines()

    chunks = chunk_markdown(text, "docs/map.md")

    second = chunks[1]["metadata"]
    assert lines[second["start_line"] - 1] == "## Layers"


# --- javascript ------------------------------------------------------------

def test_javascript_breadcrumb_is_the_enclosing_top_level_symbol():
    chunk = chunk_javascript(JAVASCRIPT, "map/base.js")[0]

    assert chunk["metadata"]["breadcrumb"] == "center"


def test_javascript_header_names_the_file_and_the_symbol():
    chunk = chunk_javascript(JAVASCRIPT, "map/base.js")[0]

    assert chunk["text"].startswith("// map/base.js :: center\n")


def test_javascript_ignores_variables_declared_inside_a_function():
    # `^\s*` used to match any indentation, so the last local variable of a
    # long function became the symbol its chunks were filed under.
    body = "\n".join(f"  cache.set('key{index}', index)" for index in range(40))
    text = f"export function helper () {{\n  const cache = new Map()\n{body}\n}}\n"

    chunks = chunk_javascript(text, "map/helper.js")

    assert len(chunks) > 1                      # the function is split
    assert {chunk["metadata"]["breadcrumb"] for chunk in chunks} == {"helper"}


def test_javascript_files_a_chunk_under_the_symbol_it_starts_on():
    padding = "\n".join(f"  const step{index} = {index}" for index in range(60))
    text = (f"export function first () {{\n{padding}\n}}\n\n"
            "export function second () {\n  return 2\n}\n")

    chunks = chunk_javascript(text, "map/two.js")

    opening = [chunk for chunk in chunks
               if "export function second" in chunk["text"]]
    assert opening
    assert all(chunk["metadata"]["breadcrumb"] == "second"
               for chunk in opening)


def test_javascript_leaves_a_chunk_above_the_first_symbol_unnamed():
    text = "import _ from 'lodash'\n\n" + JAVASCRIPT

    assert chunk_javascript(text, "map/base.js")[0]["metadata"][
        "breadcrumb"] == ""


@pytest.mark.parametrize("declaration, symbol", [
    ("export function named () {}", "named"),
    ("export default function fallback () {}", "fallback"),
    ("export async function loader () {}", "loader"),
    ("export class Tracker {}", "Tracker"),
    ("const helper = () => {}", "helper"),
    ("let mutable = 1", "mutable"),
    ("var legacy = 1", "legacy"),
    ("function* generator () {}", "generator"),
])
def test_javascript_recognises_every_declaration_form(declaration, symbol):
    chunks = chunk_javascript(declaration + "\n", "map/base.js")

    assert chunks[0]["metadata"]["breadcrumb"] == symbol


def test_javascript_lines_point_at_the_declaration():
    text = "import _ from 'lodash'\n\nexport function center () {\n  return 0\n}\n"

    chunk = chunk_javascript(text, "map/base.js")[0]

    assert chunk["metadata"]["start_line"] == 1
    assert chunk["metadata"]["end_line"] == 5


# --- vue -------------------------------------------------------------------

def test_vue_chunks_each_block_of_the_component():
    headers = [chunk["text"].splitlines()[0]
               for chunk in chunk_vue(VUE, "map/KMapPanel.vue")]

    assert any("[template]" in header for header in headers)
    assert any("[script]" in header for header in headers)


def test_vue_breadcrumb_is_the_readable_component_name():
    for chunk in chunk_vue(VUE, "map/KMapPanel.vue"):
        assert chunk["metadata"]["breadcrumb"] == "map panel"


def test_vue_skips_empty_blocks():
    text = "<template>\n\n</template>\n\n<script>\nconst a = 1\n</script>\n"

    chunks = chunk_vue(text, "map/KEmpty.vue")

    assert len(chunks) == 1
    assert "[script]" in chunks[0]["text"]


def test_vue_keeps_a_block_holding_a_nested_template():
    # A non-greedy match would close the root block on the inner </template>
    # and drop everything after it -- v-for rows and named slots are common.
    text = ("<template>\n  <q-list>\n"
            "    <template v-for=\"item in items\" :key=\"item.id\">\n"
            "      <q-item />\n"
            "    </template>\n"
            "    <KFooter class=\"marker\" />\n  </q-list>\n</template>\n")

    indexed = "\n".join(chunk["text"]
                        for chunk in chunk_vue(text, "map/KList.vue"))

    assert "q-item" in indexed
    assert "marker" in indexed


def test_vue_indexes_an_unclosed_block_rather_than_dropping_it():
    text = "<template>\n  <div class=\"orphan\" />\n"

    indexed = "\n".join(chunk["text"]
                        for chunk in chunk_vue(text, "map/KBroken.vue"))

    assert "orphan" in indexed


def test_vue_lines_cover_each_block():
    lines = VUE.splitlines()

    chunks = chunk_vue(VUE, "map/KMapPanel.vue")

    template, script = chunks[0]["metadata"], chunks[1]["metadata"]
    assert lines[template["start_line"] - 1].strip() == '<div class="map" />'
    assert lines[script["start_line"] - 1].startswith("export default")


# --- json ------------------------------------------------------------------

def test_json_indexes_translations_per_section():
    chunks = chunk_json(JSON, "i18n/en.json")

    breadcrumbs = [chunk["metadata"]["breadcrumb"] for chunk in chunks]
    assert "KMapPanel" in breadcrumbs


def test_json_groups_the_flat_labels_together():
    chunks = chunk_json(JSON, "i18n/en.json")

    grouped = [chunk for chunk in chunks if chunk["metadata"]["breadcrumb"] == ""]
    assert len(grouped) == 1
    assert "LABEL" in grouped[0]["text"]


def test_json_indexes_a_schema_per_property():
    text = ('{"$id": "users", "properties": {"name": {"type": "string"}, '
            '"email": {"type": "string"}}}')

    chunks = chunk_json(text, "common/schemas/users.create.json")

    breadcrumbs = [chunk["metadata"]["breadcrumb"] for chunk in chunks]
    assert "name" in breadcrumbs
    assert "email" in breadcrumbs


@pytest.mark.parametrize("path", [
    "package.json",
    "test/data/fixture.json",
    "common/schemas/users.json",
])
def test_json_skips_the_roles_that_are_not_worth_indexing(path):
    assert chunk_json('{"TITLE": "Map"}', path) == []


def test_json_falls_back_to_a_size_split_on_malformed_content():
    chunks = chunk_json('{"TITLE": "Map",, }', "i18n/en.json")

    assert chunks
    assert "TITLE" in chunks[0]["text"]


def test_json_lines_point_at_the_key_of_the_section():
    lines = JSON.splitlines()

    chunks = chunk_json(JSON, "i18n/en.json")

    section = next(chunk["metadata"] for chunk in chunks
                   if chunk["metadata"]["breadcrumb"] == "KMapPanel")
    assert '"KMapPanel"' in lines[section["start_line"] - 1]
