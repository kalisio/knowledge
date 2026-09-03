"""Builds the import graph of the workspace: which file imports which."""

import json
import re
from collections import defaultdict
from pathlib import Path

from ingestion.logger import get_logger
from ingestion.pipeline.change_detection import get_file_key

log = get_logger("graph")

# Only these carry imports. Markdown and JSON are indexed for retrieval but
# have no place in a dependency graph.
PARSED_EXTENSIONS = (".js", ".mjs", ".cjs", ".vue")

# Where a workspace keeps its manifests: at the root of a repository, and one
# level down for a monorepo (kdk-ekosystem/packages/kdk-core-ui).
MANIFEST_LOCATIONS = ("package.json", "packages/*/package.json")

# The module specifier of an `import ... from 'x'`, a bare `import 'x'`, an
# `export ... from 'x'` or a `require('x')`.
#
# The re-export form is not an afterthought: 346 of them in the corpus, 15%
# of all the edges, and they are the ones that build the barrels -- an
# index.js that re-exports its directory depends on every file it names.
# `from` is optional after `import` (a bare import has none) and required
# after `export`, so that `export const path = './x'` stays a constant and
# not an edge.
#
# One import in six hundred is missed: the ones whose `from` sits on another
# line, 32 out of 5623. A missed import is an edge that is absent, never an
# edge that is wrong.
_SPECIFIER = re.compile(
    r"""(?:^\s*import\s+(?:[^'"]*?from\s+)?"""
    r"""|^\s*export\s+[^'"]*?from\s+"""
    r"""|require\(\s*)"""
    r"""['"]([^'"]+)['"]""",
    re.M)

# A .vue file is not JavaScript: only its <script> block is.
_VUE_SCRIPT = re.compile(r"<script[^>]*>(.*?)</script>", re.S | re.I)

# What a relative specifier may leave out. Node resolves "../store/layers" to
# any of these, in this order.
_IMPLICIT_SUFFIXES = (".js", ".mjs", ".cjs", ".vue")
_IMPLICIT_INDEXES = ("index.js", "index.mjs", "index.vue")

# The scopes whose packages are workspace repositories rather than third
# parties. An import of @kalisio/kdk is an edge; an import of vue is not.
INTERNAL_SCOPES = ("@kalisio/", "@weacast/")


# The dependency graph of the scanned files, as
# {(repo, path): {"dependents": [...], "dependencies": [...]}} where each
# list holds "repo/path" strings -- the way a reader names a file, and what
# the API hands an agent back.
#
# Every scanned file gets an entry, including one nothing imports: "no file
# depends on this" is an answer, and telling it apart from "not indexed"
# is what keeps the API from guessing.
def build_dependency_graph(files, workspace_root, repository_dirs):
    workspace_root = Path(workspace_root)
    keys = _index_by_resolved_path(files, workspace_root)
    packages = _package_roots(repository_dirs)

    dependencies = defaultdict(list)
    dependents = defaultdict(list)
    attempted = resolved = across_repositories = 0

    for path in files:
        if path.suffix.lower() not in PARSED_EXTENSIONS:
            continue
        source_key = get_file_key(path, workspace_root)
        for specifier in _specifiers(path):
            if not _is_internal(specifier):
                continue                      # a third party, out of scope
            attempted += 1
            target = _resolve(specifier, path, keys, packages)
            if target is None or target == source_key:
                continue
            resolved += 1
            if target[0] != source_key[0]:
                across_repositories += 1
            dependencies[source_key].append(_name(target))
            dependents[target].append(_name(source_key))

    graph = {key: {"dependents": sorted(set(dependents[key])),
                   "dependencies": sorted(set(dependencies[key]))}
             for key in (get_file_key(path, workspace_root) for path in files)}

    edges = sum(len(entry["dependencies"]) for entry in graph.values())
    log.info("  %d files parsed, %d edges (%d across repositories)",
             sum(1 for path in files
                 if path.suffix.lower() in PARSED_EXTENSIONS),
             edges, across_repositories)
    log.info("  imports resolved: %d/%d (%d%%)", resolved, attempted,
             round(100 * resolved / attempted) if attempted else 0)
    return graph


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# "kdk/map/client/index.js" -- one string rather than a pair, because this is
# what the agent reads and what it would paste back as a path.
def _name(file_key):
    repository, path = file_key
    return f"{repository}/{path}"


# {resolved absolute path: (repo, path)} for the scanned files. Resolving
# once here is what makes a "../../.." specifier a dictionary lookup rather
# than a walk of the workspace.
def _index_by_resolved_path(files, workspace_root):
    return {path.resolve(): get_file_key(path, workspace_root)
            for path in files}


# {package name: (its directory, what "main" names)} for every manifest in
# the workspace. This is what turns `@kalisio/kdk/core.client` into a file,
# and it is where the edges that cross repositories come from -- the ones no
# local grep can produce. package.json is deliberately not indexed for
# retrieval, so these are read from disk rather than taken from the scan.
def _package_roots(repository_dirs):
    roots = {}
    for repository in repository_dirs.values():
        for location in MANIFEST_LOCATIONS:
            for manifest in sorted(Path(repository).glob(location)):
                name, main = _read_manifest(manifest)
                if name:
                    roots[name] = (manifest.parent, main)
    return roots


# The name a manifest declares, and its entry point -- "index.js" when it
# names none, the way Node falls back. A manifest we cannot read is a
# package we cannot resolve, never a run that stops.
def _read_manifest(manifest):
    try:
        data = json.loads(manifest.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    return data.get("name"), data.get("main") or "index.js"


# Every module specifier the file imports.
def _specifiers(path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if path.suffix.lower() == ".vue":
        text = "\n".join(_VUE_SCRIPT.findall(text))
    return _SPECIFIER.findall(text)


# Whether a specifier points inside the workspace at all.
def _is_internal(specifier):
    return specifier.startswith(".") or specifier.startswith(INTERNAL_SCOPES)


# The scanned file a specifier designates, or None when it points outside
# the corpus -- a package that is not cloned, an asset (.css, .json) the
# graph does not cover, a path that does not exist. Nothing is invented: a
# specifier that resolves to a file nobody scanned yields no edge at all,
# because an edge to a file that does not exist is worse than a missing one.
def _resolve(specifier, source, keys, packages):
    if specifier.startswith("."):
        return _resolve_path(source.parent / specifier, keys)
    # @kalisio/kdk/core.client -- the package names a directory, the rest of
    # the specifier is a path inside it. Every Kalisio import takes this
    # form; a bare @kalisio/kdk falls back on what "main" names.
    scope, _, remainder = specifier.partition("/")
    package_name, _, subpath = remainder.partition("/")
    package = packages.get(f"{scope}/{package_name}")
    if package is None:
        return None
    directory, main = package
    return _resolve_path(directory / (subpath or main), keys)


# A specifier leaves the extension out, and may name a directory holding an
# index file. Try what Node would try, in the same order.
def _resolve_path(base, keys):
    base = base.resolve()
    candidates = [base]
    candidates += [base.with_name(base.name + suffix)
                   for suffix in _IMPLICIT_SUFFIXES]
    candidates += [base / index for index in _IMPLICIT_INDEXES]
    for candidate in candidates:
        if candidate in keys:
            return keys[candidate]
    return None
