"""Cuts a JSON file into chunks, when its role is worth indexing."""

import json
import re

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

from ingestion.chunkers.line_locator import LineLocator
from ingestion.config import get_config

# JSON roles worth indexing; package manifests and fixtures are skipped.
_INDEXED_CATEGORIES = ("schema", "i18n", "docs")

# A validation schema file: under .../schemas/, named foo.create.json etc.
_SCHEMA_NAME = re.compile(r"\.(create|update|get)(?:-[a-z]+)?\.json$", re.IGNORECASE)

# The delimiters a JSON value can be wrapped in.
_CLOSING = {"{": "}", "[": "]"}


# Chunk one JSON file along boundaries that fit its role; skip other roles.
def chunk_json(text, path):
    config = get_config()
    category = _category(path)
    if category not in _INDEXED_CATEGORIES:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return _size_split(text, path, config)
    if category == "schema" and isinstance(data, dict):
        units = _schema_units(data, config)
    elif category == "i18n" and isinstance(data, dict):
        units = _i18n_units(data, config)
    elif category == "docs":
        units = [("<file>", text, [])]
    else:
        units = _tree_units(data, config)
    # Chunks are numbered in file order, whatever order the units were built
    # in, so chunk_index reads like the file does.
    locator = LineLocator(text)
    located = sorted(
        ((unit, body, _lines(text, keys, locator)) for unit, body, keys in units),
        key=lambda located_unit: located_unit[2])
    return [_chunk(path, unit, body, index, lines)
            for index, (unit, body, lines) in enumerate(located)]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

# Infer the file's role from its path and name.
def _category(path):
    parts = path.replace("\\", "/").lower().split("/")
    name = parts[-1]
    if name == "package.json":
        return "package"
    if "schemas" in parts and _SCHEMA_NAME.search(name):
        return "schema"
    if "i18n" in parts:
        return "i18n"
    if ".vitepress" in parts:
        return "docs"
    return "other"


# Schema: a metadata unit plus one unit per top-level property. Each unit
# carries the source keys it was built from, so it can be traced back to the
# lines it came from -- the body itself is re-serialised, not sliced.
def _schema_units(data, config):
    units = []
    meta_keys = [key for key in ("$id", "title", "description", "type",
                                 "required") if key in data]
    if meta_keys:
        units.append(("<schema-meta>",
                      _dumps({key: data[key] for key in meta_keys}),
                      meta_keys))
    for name, spec in (data.get("properties") or {}).items():
        units.append((name, _dumps({name: spec}), [name]))
    return units or _tree_units(data, config)


# i18n: one unit per nested section (re-split when oversized); flat labels
# grouped.
def _i18n_units(data, config):
    units = []
    flat_labels = {}
    for key, value in data.items():
        if not isinstance(value, (dict, list)):
            flat_labels[key] = value
            continue
        body = _dumps({key: value})
        if len(body) <= config.chunk_size * 2:
            units.append((key, body, [key]))
        else:
            splitter = RecursiveJsonSplitter(max_chunk_size=config.chunk_size)
            subtrees = splitter.split_json(json_data={key: value},
                                           convert_lists=True)
            for subtree in subtrees:
                units.append((key, json.dumps(subtree, ensure_ascii=False),
                              [key]))
    if flat_labels:
        units.append(("<labels>", _dumps(flat_labels), list(flat_labels)))
    return units or _tree_units(data, config)


# Any other JSON: split the tree into size-bounded subtrees.
def _tree_units(data, config):
    if not isinstance(data, (dict, list)):
        return [("<file>", _dumps(data), [])]
    splitter = RecursiveJsonSplitter(max_chunk_size=config.chunk_size)
    subtrees = splitter.split_json(json_data=data, convert_lists=True)
    return [("<part>", json.dumps(subtree, ensure_ascii=False),
             list(subtree) if isinstance(subtree, dict) else [])
            for subtree in subtrees]


# Malformed JSON: fall back to a plain character split. The pieces are cut
# from the text itself, so they can be located directly.
def _size_split(text, path, config):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    locator = LineLocator(text)
    return [_chunk(path, "<part>", piece, index,
                   locator.locate(piece) or locator.whole_file())
            for index, piece in enumerate(splitter.split_text(text))]


# Line range covering `keys` in the source text; the whole file when the unit
# is not tied to any key.
def _lines(text, keys, locator):
    spans = [span for span in (_key_span(text, key) for key in keys) if span]
    if not spans:
        return locator.whole_file()
    return locator.span(min(start for start, _ in spans),
                        max(end for _, end in spans))


# (start, end) offsets of `"key": <value>` in the source text.
def _key_span(text, key):
    needle = json.dumps(key, ensure_ascii=False)
    start = text.find(needle)
    if start < 0:
        return None
    colon = text.find(":", start + len(needle))
    if colon < 0:
        return None
    value = colon + 1
    while value < len(text) and text[value] in " \t\r\n":
        value += 1
    if value < len(text) and text[value] in _CLOSING:
        return start, _closing_offset(text, value)
    end = text.find("\n", value)
    return start, len(text) if end < 0 else end


# Offset just past the delimiter closing the one at `start`, skipping the
# delimiters that appear inside strings.
def _closing_offset(text, start):
    opening = text[start]
    closing = _CLOSING[opening]
    depth = 0
    in_string = False
    escaped = False
    for offset in range(start, len(text)):
        char = text[offset]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return offset + 1
    return len(text)


# Build one chunk dict from a unit label, its body text and its line range.
def _chunk(path, unit, body, index, lines):
    breadcrumb = "" if unit.startswith("<") else unit
    start_line, end_line = lines
    return {
        "text": _header(path, unit) + "\n" + body,
        "metadata": {
            "path": path,
            "chunk_index": index,
            "breadcrumb": breadcrumb,
            "start_line": start_line,
            "end_line": end_line,
        },
    }


# Build the "// <path> :: <unit>" header, or "// <path>" for a placeholder unit.
def _header(path, unit):
    if unit and not unit.startswith("<"):
        return f"// {path} :: {unit}"
    return f"// {path}"


# Dump JSON readably, preserving non-ASCII characters.
def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)
