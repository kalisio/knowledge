import re
from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Target chunk length and overlap in characters.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Match each SFC block: tag name (1), attributes (2), body (3).
_SFC_BLOCK = re.compile(
    r"<(template|script|style)\b([^>]*)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)

# Split CamelCase and acronyms at word boundaries.
_CAMEL_CASE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Read a block's lang attribute, e.g. <script lang="ts">.
_LANG_ATTRIBUTE = re.compile(r"""lang\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


# Chunk one Vue single-file component: split each block, tag with the component.
def chunk_vue(text, source_path):
    component = _component_name(source_path)
    chunks = []
    for match in _SFC_BLOCK.finditer(text):
        block_type = match.group(1).lower()
        attributes = match.group(2)
        body = match.group(3).strip()
        if not body:
            continue
        header = _header(source_path, component, block_type)
        for piece in _split_block(block_type, body, attributes):
            chunks.append({
                "text": header + "\n" + piece,
                "metadata": {
                    "source_path": source_path,
                    "chunk_index": len(chunks),
                    "breadcrumb": component,
                },
            })
    return chunks


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

# Build the block header: a JS comment for script, an HTML comment otherwise.
def _header(source_path, component, block_type):
    if component:
        label = f"{source_path} — {component} [{block_type}]"
    else:
        label = f"{source_path} [{block_type}]"
    return f"// {label}" if block_type == "script" else f"<!-- {label} -->"


# Split a block's body with a splitter matched to its type.
def _split_block(block_type, body, attributes):
    if block_type == "style" and len(body) <= CHUNK_SIZE * 2:
        return [body]
    if block_type == "script":
        language = Language.JS
    elif block_type == "template" and _is_html(attributes):
        language = Language.HTML
    else:
        language = None
    if language is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(body)


# Check the template declares html, or omits the lang attribute.
def _is_html(attributes):
    lang = _LANG_ATTRIBUTE.search(attributes)
    return lang is None or lang.group(1).lower() == "html"


# Derive a readable component name from the file name; empty when none remains.
def _component_name(source_path):
    stem = Path(source_path).stem
    if stem.startswith("K") and len(stem) > 1:
        stem = stem[1:]
    words = [word.lower() for word in _CAMEL_CASE.split(stem) if len(word) > 1]
    return " ".join(words)
