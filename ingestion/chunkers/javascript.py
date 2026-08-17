import re

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Target chunk length and overlap in characters.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# A top-level declaration; the captured group is the symbol name.
_TOP_LEVEL_SYMBOL = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?"
    r"(?:async\s+)?"
    r"(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


# Chunk one JavaScript file: split at JS boundaries, tag each with its symbol.
def chunk_javascript(text, source_path):
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    cursor = 0
    for chunk_index, piece in enumerate(splitter.split_text(text)):
        offset = text.find(piece[:32], cursor) if piece else -1
        if offset < 0:
            offset = cursor
        symbol = _nearest_symbol(text, offset)
        header = f"// {source_path}" + (f" :: {symbol}" if symbol else "")
        chunks.append({
            "text": header + "\n" + piece,
            "metadata": {
                "source_path": source_path,
                "chunk_index": chunk_index,
                "breadcrumb": symbol,
            },
        })
        cursor = offset + len(piece)
    return chunks

# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

# Get the last top-level symbol declared at or before `offset`.
def _nearest_symbol(text, offset):
    symbol = ""
    for match in _TOP_LEVEL_SYMBOL.finditer(text):
        if match.start() > offset:
            break
        symbol = match.group(1)
    return symbol
