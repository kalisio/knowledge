"""Split a JSON file into chunks for indexing.

JSON files play different roles (validation schemas, i18n translation tables,
package manifests, ...), so they are split by role rather than by raw size:
the role is inferred from the file path, then the payload is cut along
meaningful boundaries (one chunk per schema property, per translation
section, ...). A malformed file falls back to a plain size-based split.

Each chunk's text starts with a "// <path> :: <unit>" header, matching the
JS and Vue anchor style.

Returns a list of {"text", "metadata"} dicts using the shared chunk shape
(source_path / breadcrumb / chunk_index) plus a category field (the JSON
role); breadcrumb holds the unit (property/section/...) within the file.

Note: this file is json.py, but `import json` below resolves to the standard
library (absolute imports), not this module.
Note: which roles actually get indexed is a separate, upstream decision.
"""

import json
import re

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)


# Target chunk length and overlap, in characters.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# A validation schema file: under .../schemas/, named foo.create.json etc.
_SCHEMA_NAME = re.compile(r"\.(create|update|get)(?:-[a-z]+)?\.json$", re.I)


# Split one JSON file into chunks, cut along boundaries that fit its role.
def chunk_json(text, source_path):
    category = _category(source_path)
    try:
        data = json.loads(text)
    except ValueError:
        return _size_split(text, source_path, category)

    if category == "schema" and isinstance(data, dict):
        units = _schema_units(data)
    elif category == "i18n" and isinstance(data, dict):
        units = _i18n_units(data)
    elif category == "package" and isinstance(data, dict):
        units = [("<package>", _package_summary(data))]
    elif category == "docs":
        units = [("<file>", text)]
    else:
        units = _tree_units(data)

    return [
        _chunk(source_path, category, unit, body, index)
        for index, (unit, body) in enumerate(units)
    ]


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Infer the file's role from its path and name.
def _category(source_path):
    parts = source_path.replace("\\", "/").lower().split("/")
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


# Schema: a metadata chunk plus one chunk per top-level property.
def _schema_units(data):
    units = []
    meta = {k: data[k] for k in
            ("$id", "title", "description", "type", "required") if k in data}
    if meta:
        units.append(("<schema-meta>", _dumps(meta)))
    for name, spec in (data.get("properties") or {}).items():
        units.append((name, _dumps({name: spec})))
    return units or _tree_units(data)


# i18n: one chunk per nested section (re-split if a section is oversized);
# flat scalar labels grouped into a single chunk.
def _i18n_units(data):
    units = []
    flat = {}
    for key, value in data.items():
        if not isinstance(value, (dict, list)):
            flat[key] = value
            continue
        body = _dumps({key: value})
        if len(body) <= CHUNK_SIZE * 2:
            units.append((key, body))
        else:
            splitter = RecursiveJsonSplitter(max_chunk_size=CHUNK_SIZE)
            for sub in splitter.split_json(
                    json_data={key: value}, convert_lists=True):
                units.append((key, json.dumps(sub, ensure_ascii=False)))
    if flat:
        units.insert(0, ("<labels>", _dumps(flat)))
    return units or _tree_units(data)


# Package manifest: keep only descriptive fields and dependency names.
def _package_summary(data):
    keep = {
        "name": data.get("name"),
        "description": data.get("description"),
        "version": data.get("version"),
        "scripts": data.get("scripts"),
        "dependencies": sorted(data.get("dependencies") or {}) or None,
        "devDependencies": sorted(data.get("devDependencies") or {}) or None,
    }
    return _dumps({k: v for k, v in keep.items() if v is not None})


# Any other JSON: split the tree into size-bounded subtrees.
def _tree_units(data):
    if not isinstance(data, (dict, list)):
        return [("<file>", _dumps(data))]
    splitter = RecursiveJsonSplitter(max_chunk_size=CHUNK_SIZE)
    subtrees = splitter.split_json(json_data=data, convert_lists=True)
    return [
        ("<part>", json.dumps(sub, ensure_ascii=False))
        for sub in subtrees
    ]


# Malformed JSON: fall back to a plain character split.
def _size_split(text, source_path, category):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [
        _chunk(source_path, category, "<part>", piece, index)
        for index, piece in enumerate(splitter.split_text(text))
    ]


# Build one chunk dict from a unit label and its body text.
def _chunk(source_path, category, unit, body, index):
    breadcrumb = "" if unit.startswith("<") else unit
    return {
        "text": _header(source_path, unit) + "\n" + body,
        "metadata": {
            "source_path": source_path,
            "breadcrumb": breadcrumb,
            "chunk_index": index,
            "category": category,
        },
    }


# Header: "// <path> :: <unit>" for a real unit, else "// <path>".
def _header(source_path, unit):
    if unit and not unit.startswith("<"):
        return f"// {source_path} :: {unit}"
    return f"// {source_path}"


# json.dumps with readable, non-ASCII-preserving formatting.
def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)
