import re
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from ingestion.chunkers.locator import Locator
from ingestion.config import get_config

# The opening tag of an SFC block: tag name (1), attributes (2).
_BLOCK_OPEN = re.compile(r"<(template|script|style)\b([^>]*)>", re.IGNORECASE)

# Split CamelCase and acronyms at word boundaries.
_CAMEL_CASE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Read a block's lang attribute, e.g. <script lang="ts">.
_LANG_ATTRIBUTE = re.compile(r"""lang\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


# Chunk one Vue single-file component: split each block, tag with the component.
def chunk_vue(text, path):
    config = get_config()
    component = _component_name(path)
    locator = Locator(text)
    chunks = []
    for block_type, attributes, body_start, body_end in _iter_blocks(text):
        body = text[body_start:body_end].strip()
        if not body:
            continue
        header = _header(path, component, block_type)
        locator.seek(body_start)
        for piece in _split_block(block_type, body, attributes, config):
            start_line, end_line = (
                locator.locate(piece) or locator.span(body_start, body_end))
            chunks.append({
                "text": header + "\n" + piece,
                "metadata": {
                    "path": path,
                    "chunk_index": len(chunks),
                    "breadcrumb": component,
                    "start_line": start_line,
                    "end_line": end_line,
                },
            })
    return chunks


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

# Walk the top-level SFC blocks as (tag, attributes, body_start, body_end).
# Nesting is counted rather than matched non-greedily: a <template v-for> or
# a named slot inside the template would otherwise close the root block early
# and everything after it would be dropped.
def _iter_blocks(text):
    position = 0
    while True:
        opening = _BLOCK_OPEN.search(text, position)
        if opening is None:
            return
        tag = opening.group(1).lower()
        body_start = opening.end()
        body_end, block_end = _matching_close(text, tag, body_start)
        yield tag, opening.group(2), body_start, body_end
        position = block_end


# End of the body and end of the block, for the close tag matching the open
# one. An unclosed block runs to the end of the file rather than being lost.
def _matching_close(text, tag, body_start):
    pattern = re.compile(rf"<(/?){tag}\b[^>]*>", re.IGNORECASE)
    depth = 1
    position = body_start
    while True:
        match = pattern.search(text, position)
        if match is None:
            return len(text), len(text)
        depth += -1 if match.group(1) else 1
        if depth == 0:
            return match.start(), match.end()
        position = match.end()


# Build the block header: a JS comment for script, an HTML comment otherwise.
def _header(path, component, block_type):
    if component:
        label = f"{path} — {component} [{block_type}]"
    else:
        label = f"{path} [{block_type}]"
    return f"// {label}" if block_type == "script" else f"<!-- {label} -->"


# Split a block's body with a splitter matched to its type.
def _split_block(block_type, body, attributes, config):
    if block_type == "style" and len(body) <= config.code_chunk_size * 2:
        return [body]
    if block_type == "script":
        language = Language.JS
    elif block_type == "template" and _is_html(attributes):
        language = Language.HTML
    else:
        language = None
    if language is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.code_chunk_size,
            chunk_overlap=config.code_chunk_overlap)
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=config.code_chunk_size,
            chunk_overlap=config.code_chunk_overlap)
    return splitter.split_text(body)


# Check the template declares html, or omits the lang attribute.
def _is_html(attributes):
    lang = _LANG_ATTRIBUTE.search(attributes)
    return lang is None or lang.group(1).lower() == "html"


# Derive a readable component name from the file name; empty when none remains.
def _component_name(path):
    stem = Path(path).stem
    if stem.startswith("K") and len(stem) > 1:
        stem = stem[1:]
    words = [word.lower() for word in _CAMEL_CASE.split(stem) if len(word) > 1]
    return " ".join(words)
