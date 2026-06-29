"""Split a Vue single-file component (SFC) into chunks for indexing.

A Vue SFC is built from <template>, <script> and <style> blocks. Each block
is split with a splitter suited to its language (the script block like
JavaScript, the template like HTML, the style by size). A readable component
name is derived from the file name (e.g. KZoomControl.vue -> "zoom control")
and added to every chunk so natural-language queries match more easily.

Each chunk's text starts with a header naming the file, component and block,
so the chunk is self-describing once embedded.

Returns a list of {"text", "metadata"} dicts using the shared chunk shape
(source_path / breadcrumb / chunk_index) plus a block_type field
("template" / "script" / "style"); breadcrumb holds the component name.
"""

import re
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


# Target chunk length and overlap, in characters.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Match each SFC block: tag name (1), attributes (2), body (3).
_SFC_BLOCK = re.compile(
    r"<(template|script|style)\b([^>]*)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)

# Split CamelCase / acronyms at word boundaries.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Read a block's lang attribute, e.g. <script lang="ts">.
_LANG_ATTR = re.compile(r"""lang\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


# Split one Vue SFC into per-block chunks.
def chunk_vue(text, source_path):
    component = _component_name(source_path)
    chunks = []
    for match in _SFC_BLOCK.finditer(text):
        block_type = match.group(1).lower()
        attrs = match.group(2)
        body = match.group(3).strip()
        if not body:
            continue
        header = _header(source_path, component, block_type)
        for piece in _split_block(block_type, body, attrs):
            chunks.append({
                "text": header + "\n" + piece,
                "metadata": {
                    "source_path": source_path,
                    "breadcrumb": component,
                    "chunk_index": len(chunks),
                    "block_type": block_type,
                },
            })
    return chunks


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Build the per-block header (a JS comment for script, else an HTML comment).
def _header(source_path, component, block_type):
    if component:
        label = f"{source_path} — {component} [{block_type}]"
    else:
        label = f"{source_path} [{block_type}]"
    if block_type == "script":
        return f"// {label}"
    return f"<!-- {label} -->"


# Split a block's body with a splitter suited to its type.
def _split_block(block_type, body, attrs):
    if block_type == "style" and len(body) <= CHUNK_SIZE * 2:
        return [body]
    if block_type == "script":
        language = Language.JS
    elif block_type == "template" and _is_html(attrs):
        language = Language.HTML
    else:
        language = None
    if language is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(body)


# True when the template has no lang attribute or declares html.
def _is_html(attrs):
    lang = _LANG_ATTR.search(attrs)
    return lang is None or lang.group(1).lower() == "html"


# Derive a readable component name from the file name; "" if none remains.
def _component_name(source_path):
    stem = Path(source_path).stem
    if stem.startswith("K") and len(stem) > 1:
        stem = stem[1:]
    words = [w.lower() for w in _CAMEL.split(stem) if len(w) > 1]
    return " ".join(words)
