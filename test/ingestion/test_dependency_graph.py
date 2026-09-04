"""The import graph: what counts as an edge, and what must never become one.

The graph answers "what breaks if I change this file", so a wrong edge is
worse than a missing one: it sends an agent to reassure itself about a file
that does not exist. Most of these tests pin what is *not* an edge.

The sources of test/data/graph are laid out as a small workspace and checked
against test/data/expected/graph.json, the way the chunkers are checked
against their own references. Regenerate it after an intentional change:

    UPDATE_GRAPH_REFERENCE=1 pytest test/ingestion/test_dependency_graph.py
"""

import json
import os
import shutil
from pathlib import Path

from ingestion.pipeline.dependency_graph import build_dependency_graph

DATA = Path(__file__).resolve().parent.parent / "data"

# The sources of test/data/graph, and where each one sits in the workspace.
# The paths matter: they are what the relative imports resolve against.
GRAPH_SAMPLES = {
    "kdk/core/client/store.js": "store.js",
    "kdk/core/client/api.js": "api.js",
    "kdk/core.client.js": "core.client.js",
    "kdk/package.json": "package.json",
    "kano/client/components/KLayerList.vue": "KLayerList.vue",
}


# A workspace on disk: {"repo/path.js": "source"}. Repositories are the first
# path segment, the way a real workspace nests them per organisation.
def workspace(tmp_path, files):
    paths = []
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        paths.append(path)
    return paths


def build(tmp_path, files, repositories=None):
    paths = workspace(tmp_path, files)
    repository_dirs = {name: tmp_path / name
                       for name in (repositories or
                                    {name.split("/")[0] for name in files})}
    return build_dependency_graph(paths, tmp_path, repository_dirs)


# --- the whole graph, against its reference --------------------------------

def test_the_graph_of_the_sample_workspace_matches_its_reference(tmp_path):
    # One assertion covering the four shapes an edge takes, on files that
    # look like the real thing rather than one-line stubs: a relative
    # import, a re-export, an import by package name across repositories,
    # and a file nothing imports.
    files = []
    for target, sample in GRAPH_SAMPLES.items():
        path = tmp_path / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA / "graph" / sample, path)
        if not target.endswith("package.json"):   # not an indexable file
            files.append(path)

    graph = build_dependency_graph(
        files, tmp_path,
        {"kdk": tmp_path / "kdk", "kano": tmp_path / "kano"})

    built = {f"{repo}/{path}": entry
             for (repo, path), entry in sorted(graph.items())}
    reference_path = DATA / "expected" / "graph.json"
    if os.environ.get("UPDATE_GRAPH_REFERENCE"):
        reference_path.write_text(
            json.dumps(built, indent=2, ensure_ascii=False) + "\n")
    assert built == json.loads(reference_path.read_text())


# --- the edge itself --------------------------------------------------------

def test_a_relative_import_is_an_edge_in_both_directions(tmp_path):
    graph = build(tmp_path, {
        "kdk/client/store.js": "export const store = {}",
        "kdk/client/api.js": "import { store } from './store.js'",
    })

    assert graph[("kdk", "client/api.js")]["dependencies"] == [
        "kdk/client/store.js"]
    assert graph[("kdk", "client/store.js")]["dependents"] == [
        "kdk/client/api.js"]


def test_a_require_call_is_an_edge_too(tmp_path):
    graph = build(tmp_path, {
        "kdk/utils.js": "module.exports = {}",
        "kdk/index.cjs": "const utils = require('./utils')",
    })

    assert graph[("kdk", "utils.js")]["dependents"] == ["kdk/index.cjs"]


def test_a_re_export_is_an_edge(tmp_path):
    # An index.js that re-exports its directory depends on every file it
    # names. 346 of these in the corpus, 15% of all the edges, and they are
    # what makes a barrel a barrel.
    graph = build(tmp_path, {
        "kdk/store.js": "export const store = {}",
        "kdk/index.js": "export * from './store.js'",
    })

    assert graph[("kdk", "store.js")]["dependents"] == ["kdk/index.js"]


def test_an_exported_constant_that_looks_like_a_path_is_not_an_edge(tmp_path):
    # `from` is required after `export` for exactly this reason.
    graph = build(tmp_path, {
        "kdk/store.js": "export const store = {}",
        "kdk/config.js": "export const location = './store.js'",
    })

    assert graph[("kdk", "store.js")]["dependents"] == []


def test_the_extension_may_be_left_out(tmp_path):
    graph = build(tmp_path, {
        "kdk/store.js": "export const store = {}",
        "kdk/api.js": "import { store } from './store'",
    })

    assert graph[("kdk", "store.js")]["dependents"] == ["kdk/api.js"]


def test_a_directory_resolves_to_its_index(tmp_path):
    graph = build(tmp_path, {
        "kdk/mixins/index.js": "export * from './base.js'",
        "kdk/app.js": "import mixins from './mixins'",
    })

    assert graph[("kdk", "mixins/index.js")]["dependents"] == ["kdk/app.js"]


def test_a_file_is_listed_once_however_often_it_is_imported(tmp_path):
    graph = build(tmp_path, {
        "kdk/store.js": "export const store = {}",
        "kdk/api.js": ("import { a } from './store.js'\n"
                       "import { b } from './store'\n"),
    })

    assert graph[("kdk", "store.js")]["dependents"] == ["kdk/api.js"]


# --- what is deliberately not an edge --------------------------------------

def test_a_third_party_import_is_not_an_edge(tmp_path):
    graph = build(tmp_path, {
        "kdk/app.js": ("import { ref } from 'vue'\n"
                       "import _ from 'lodash'\n"),
    })

    assert graph[("kdk", "app.js")]["dependencies"] == []


def test_an_import_of_a_file_nobody_scanned_invents_nothing(tmp_path):
    # A package with no "main" resolves to an index.js that may not exist.
    # Creating the node anyway is how a graph grows a file that is not in
    # the corpus -- and it was the top of the PageRank ranking when the
    # prototype did exactly that.
    graph = build(tmp_path, {
        "kdk/app.js": "import config from './config.json'",
    })

    assert graph[("kdk", "app.js")]["dependencies"] == []
    assert list(graph) == [("kdk", "app.js")]


def test_a_file_importing_itself_is_not_an_edge(tmp_path):
    graph = build(tmp_path, {
        "kdk/app.js": "import { self } from './app.js'",
    })

    assert graph[("kdk", "app.js")]["dependencies"] == []
    assert graph[("kdk", "app.js")]["dependents"] == []


def test_a_markdown_file_is_never_parsed(tmp_path):
    # Markdown is indexed for retrieval, and its code blocks are full of
    # import lines that describe nothing about the corpus.
    graph = build(tmp_path, {
        "kdk/store.js": "export const store = {}",
        "kdk/README.md": "```js\nimport { store } from './store.js'\n```",
    })

    assert graph[("kdk", "store.js")]["dependents"] == []


def test_only_the_script_block_of_a_vue_file_is_read(tmp_path):
    # The template can hold anything that looks like an import -- a code
    # sample in documentation, a string in an attribute.
    graph = build(tmp_path, {
        "kdk/store.js": "export const store = {}",
        "kdk/other.js": "export const other = {}",
        "kdk/KMap.vue": (
            "<template>\n"
            "  <pre>import { other } from './other.js'</pre>\n"
            "</template>\n"
            "<script>\n"
            "import { store } from './store.js'\n"
            "export default {}\n"
            "</script>\n"),
    })

    assert graph[("kdk", "store.js")]["dependents"] == ["kdk/KMap.vue"]
    assert graph[("kdk", "other.js")]["dependents"] == []


# --- across repositories, the part no local grep can produce ---------------

def test_a_package_subpath_crosses_repositories(tmp_path):
    # Every Kalisio import takes this form: @kalisio/kdk/core.client, not
    # @kalisio/kdk. Resolving only the package name loses all of them.
    files = {
        "kdk/package.json": json.dumps({"name": "@kalisio/kdk"}),
        "kdk/core.client.js": "export const core = {}",
        "kano/src/main.js": "import { core } from '@kalisio/kdk/core.client'",
    }
    graph = build(tmp_path, files, repositories={"kdk", "kano"})

    assert graph[("kdk", "core.client.js")]["dependents"] == ["kano/src/main.js"]


def test_a_bare_package_import_falls_back_on_main(tmp_path):
    files = {
        "kdk/package.json": json.dumps({"name": "@kalisio/kdk",
                                        "main": "core.client.js"}),
        "kdk/core.client.js": "export const core = {}",
        "kano/src/main.js": "import { core } from '@kalisio/kdk'",
    }
    graph = build(tmp_path, files, repositories={"kdk", "kano"})

    assert graph[("kdk", "core.client.js")]["dependents"] == ["kano/src/main.js"]


def test_a_monorepo_package_is_resolved_too(tmp_path):
    files = {
        "eko/packages/core-ui/package.json":
            json.dumps({"name": "@kalisio/kdk-core-ui", "main": "src/i18n.js"}),
        "eko/packages/core-ui/src/i18n.js": "export const i18n = {}",
        "kano/src/main.js": "import i18n from '@kalisio/kdk-core-ui'",
    }
    graph = build(tmp_path, files, repositories={"eko", "kano"})

    assert graph[("eko", "packages/core-ui/src/i18n.js")]["dependents"] == [
        "kano/src/main.js"]


def test_an_uncloned_package_yields_no_edge(tmp_path):
    graph = build(tmp_path, {
        "kano/src/main.js": "import x from '@kalisio/leaflet-pmtiles'",
    })

    assert graph[("kano", "src/main.js")]["dependencies"] == []


# --- the shape the API reads back ------------------------------------------

def test_every_scanned_file_gets_an_entry(tmp_path):
    # "nothing depends on this file" is an answer. The API tells it apart
    # from "the corpus has never seen this file", and it can only do that
    # if the entry exists.
    graph = build(tmp_path, {
        "kdk/lonely.js": "export const lonely = {}",
        "kdk/README.md": "# Nothing",
    })

    assert graph[("kdk", "lonely.js")] == {"dependents": [], "dependencies": []}
    assert ("kdk", "README.md") in graph


def test_the_lists_are_sorted(tmp_path):
    graph = build(tmp_path, {
        "kdk/store.js": "export const store = {}",
        "kdk/zeta.js": "import { store } from './store.js'",
        "kdk/alpha.js": "import { store } from './store.js'",
    })

    assert graph[("kdk", "store.js")]["dependents"] == [
        "kdk/alpha.js", "kdk/zeta.js"]
