"""Cuts a shell script into chunks, one per function."""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.chunkers.line_locator import LineLocator
from ingestion.config import get_config

# A function definition at column zero: `name() {`, `function name {`,
# `function name() {`. Column zero on purpose, the way the JavaScript
# chunker anchors its symbols: an indented definition sits inside another
# one, and the nearest match would pass a helper off as the function.
_FUNCTION = re.compile(
    r"^(?:function\s+)?([A-Za-z_][\w-]*)\s*(?:\(\s*\))?\s*\{[ \t]*$",
    re.M)

# What closes a function opened at column zero. Kalisio's shell is written
# that way throughout; a script that is not falls back to the next function
# as the boundary, which is the same thing for every function but the last.
_CLOSING = re.compile(r"^\}[ \t]*$", re.M)

# A comment line, or a blank one: the header of the function under it.
_COMMENT_OR_BLANK = re.compile(r"^\s*(?:#|$)")


# Chunk one shell script: a chunk per function, its comment header
# included, and the code between functions gathered into its own chunks.
def chunk_shell(text, path):
    config = get_config()
    locator = LineLocator(text)
    chunks = []
    for name, body in _units(text):
        body = body.strip("\n")
        if not body.strip():
            continue
        for piece in _sized(body, config):
            lines = locator.locate(piece) or locator.whole_file()
            chunks.append(_chunk(path, name, piece, len(chunks), lines))
    return chunks


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# [(function name or "", text)] in file order. A function runs from its
# header comment to its closing brace; whatever lies between two functions
# is top-level code and keeps no name.
def _units(text):
    functions = list(_FUNCTION.finditer(text))
    if not functions:
        return [("", text)]
    limit = get_config().code_chunk_size
    units = []
    cursor = 0
    for index, match in enumerate(functions):
        start = _header_start(text, match.start())
        # A header longer than a chunk is the file's, not the function's --
        # a licence, a usage block -- and stays top-level code.
        if match.start() - start > limit:
            start = match.start()
        next_start = (functions[index + 1].start()
                      if index + 1 < len(functions) else len(text))
        closing = _CLOSING.search(text, match.end(), next_start)
        end = closing.end() if closing else next_start
        if start > cursor and text[cursor:start].strip():
            units.append(("", text[cursor:start]))
        units.append((match.group(1), text[start:end]))
        cursor = end
    if text[cursor:].strip():
        units.append(("", text[cursor:]))
    return units


# The offset where a function's header starts: walk up through the comment
# and blank lines above the definition, stopping at the first line of code.
# Blank lines do not stop the walk, so a section title -- `### Secrets`,
# then a blank line, then the function's own comment -- travels with the
# function under it instead of being left as a chunk of one line.
def _header_start(text, definition):
    lines = text[:definition].splitlines(keepends=True)
    start = definition
    for line in reversed(lines):
        if not _COMMENT_OR_BLANK.match(line):
            break
        start -= len(line)
    return start


# A function that still exceeds the chunk size is cut on characters, keeping
# its name -- better a long function in three pieces than dropped.
def _sized(body, config):
    if len(body) <= config.code_chunk_size * 2:
        return [body]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.code_chunk_size,
        chunk_overlap=config.code_chunk_overlap)
    return splitter.split_text(body)


def _chunk(path, name, body, index, lines):
    header = f"# {path}" + (f" :: {name}" if name else "")
    start_line, end_line = lines
    return {
        "text": header + "\n" + body,
        "metadata": {
            "path": path,
            "chunk_index": index,
            "breadcrumb": name,
            "start_line": start_line,
            "end_line": end_line,
        },
    }
