"""Split a JavaScript file into chunks for indexing.

The file is split at JavaScript boundaries (functions, classes, blocks) into
pieces of about CHUNK_SIZE characters, with a small overlap so a definition
that straddles a cut is not lost. Each chunk's text starts with a comment
line naming the file and the nearest top-level symbol, e.g.
``// path/to/file.js :: functionName``, so the chunk is self-describing once
embedded and later shown to the model.

Returns a list of ``{"text", "metadata"}`` dicts: ``text`` is the content to
embed (with that comment prefix); ``metadata`` repeats the location as
separate fields. Repository and commit info is added later, before the chunk
is stored in the vector database.
"""

import re

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


# Target chunk length and overlap, in characters.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# A top-level declaration ("export default function foo", "class Bar",
# "const baz" …); the captured group is the symbol name.
_TOP_LEVEL_SYMBOL = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?"
    r"(?:async\s+)?"
    r"(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)


# Split one JavaScript file into chunks, each tagged with its nearest symbol.
def chunk_js(text, source_path, chunk_size=CHUNK_SIZE,
             chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    cursor = 0
    for index, piece in enumerate(splitter.split_text(text)):
        offset = text.find(piece[:32], cursor) if piece else -1
        if offset < 0:
            offset = cursor
        symbol = _nearest_symbol(text, offset)
        header = f"// {source_path}" + (f" :: {symbol}" if symbol else "")
        chunks.append({
            "text": header + "\n" + piece,
            "metadata": {
                "source_path": source_path,
                "breadcrumb": symbol,
                "chunk_index": index,
            },
        })
        cursor = offset + len(piece)
    return chunks


# Name of the last top-level symbol declared at or before `offset`.
def _nearest_symbol(text, offset):
    symbol = ""
    for match in _TOP_LEVEL_SYMBOL.finditer(text):
        if match.start() > offset:
            break
        symbol = match.group(1)
    return symbol
