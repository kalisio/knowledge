import re

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from ingestion.chunkers.locator import Locator
from ingestion.config import get_config

# A top-level declaration; the captured group is the symbol name. Anchored at
# column zero on purpose: `^\s*` would also match declarations indented inside
# a function body, and since the nearest match wins, a local variable would be
# passed off as the symbol the chunk belongs to.
_TOP_LEVEL_SYMBOL = re.compile(
    r"^(?:export\s+(?:default\s+)?)?"
    r"(?:async\s+)?"
    r"(?:function\s*\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


# Chunk one JavaScript file: split at JS boundaries, tag each with its symbol.
def chunk_javascript(text, path):
    config = get_config()
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=config.code_chunk_size,
        chunk_overlap=config.code_chunk_overlap)
    locator = Locator(text)
    symbols = _declared_symbols(text, locator)
    chunks = []
    for chunk_index, piece in enumerate(splitter.split_text(text)):
        start_line, end_line = locator.locate(piece) or locator.whole_file()
        symbol = _symbol_at(symbols, start_line)
        header = f"// {path}" + (f" :: {symbol}" if symbol else "")
        chunks.append({
            "text": header + "\n" + piece,
            "metadata": {
                "path": path,
                "chunk_index": chunk_index,
                "breadcrumb": symbol,
                "start_line": start_line,
                "end_line": end_line,
            },
        })
    return chunks

# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

# Every top-level declaration as (line, symbol), in file order.
def _declared_symbols(text, locator):
    return [(locator.line_of(match.start()), match.group(1))
            for match in _TOP_LEVEL_SYMBOL.finditer(text)]


# The last symbol declared at or before `line`, or "" above the first one.
def _symbol_at(symbols, line):
    symbol = ""
    for declared_line, name in symbols:
        if declared_line > line:
            break
        symbol = name
    return symbol
