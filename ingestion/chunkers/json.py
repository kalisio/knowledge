import json
import re

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

# Target chunk length and overlap in characters.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# JSON roles worth indexing; package manifests and fixtures are skipped.
_INDEXED_CATEGORIES = ("schema", "i18n", "docs")

# A validation schema file: under .../schemas/, named foo.create.json etc.
_SCHEMA_NAME = re.compile(r"\.(create|update|get)(?:-[a-z]+)?\.json$", re.IGNORECASE)


# Chunk one JSON file along boundaries that fit its role; skip other roles.
def chunk_json(text, source_path):
    category = _category(source_path)
    if category not in _INDEXED_CATEGORIES:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return _size_split(text, source_path)
    if category == "schema" and isinstance(data, dict):
        units = _schema_units(data)
    elif category == "i18n" and isinstance(data, dict):
        units = _i18n_units(data)
    elif category == "docs":
        units = [("<file>", text)]
    else:
        units = _tree_units(data)
    return [_chunk(source_path, unit, body, index)
            for index, (unit, body) in enumerate(units)]


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


# Schema: a metadata unit plus one unit per top-level property.
def _schema_units(data):
    units = []
    schema_meta = {key: data[key]
                   for key in ("$id", "title", "description", "type", "required")
                   if key in data}
    if schema_meta:
        units.append(("<schema-meta>", _dumps(schema_meta)))
    for name, spec in (data.get("properties") or {}).items():
        units.append((name, _dumps({name: spec})))
    return units or _tree_units(data)


# i18n: one unit per nested section (re-split when oversized); flat labels grouped.
def _i18n_units(data):
    units = []
    flat_labels = {}
    for key, value in data.items():
        if not isinstance(value, (dict, list)):
            flat_labels[key] = value
            continue
        body = _dumps({key: value})
        if len(body) <= CHUNK_SIZE * 2:
            units.append((key, body))
        else:
            splitter = RecursiveJsonSplitter(max_chunk_size=CHUNK_SIZE)
            subtrees = splitter.split_json(json_data={key: value}, convert_lists=True)
            for subtree in subtrees:
                units.append((key, json.dumps(subtree, ensure_ascii=False)))
    if flat_labels:
        units.insert(0, ("<labels>", _dumps(flat_labels)))
    return units or _tree_units(data)


# Any other JSON: split the tree into size-bounded subtrees.
def _tree_units(data):
    if not isinstance(data, (dict, list)):
        return [("<file>", _dumps(data))]
    splitter = RecursiveJsonSplitter(max_chunk_size=CHUNK_SIZE)
    subtrees = splitter.split_json(json_data=data, convert_lists=True)
    return [("<part>", json.dumps(subtree, ensure_ascii=False)) for subtree in subtrees]


# Malformed JSON: fall back to a plain character split.
def _size_split(text, source_path):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [_chunk(source_path, "<part>", piece, index)
            for index, piece in enumerate(splitter.split_text(text))]


# Build one chunk dict from a unit label and its body text.
def _chunk(source_path, unit, body, index):
    breadcrumb = "" if unit.startswith("<") else unit
    return {
        "text": _header(source_path, unit) + "\n" + body,
        "metadata": {
            "source_path": source_path,
            "chunk_index": index,
            "breadcrumb": breadcrumb,
        },
    }


# Build the "// <path> :: <unit>" header, or "// <path>" for a placeholder unit.
def _header(source_path, unit):
    if unit and not unit.startswith("<"):
        return f"// {source_path} :: {unit}"
    return f"// {source_path}"


# Dump JSON readably, preserving non-ASCII characters.
def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)
