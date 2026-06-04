"""Split a JSON file into chunks for indexing.

JSON files play very different roles (validation schemas, translation
tables, package manifests, test fixtures …), so they are chunked by role
rather than uniformly. The role is inferred from the file path and name,
then the payload is split along meaningful boundaries (for example one
chunk per schema property, or per translation section) instead of by raw
character count. Each chunk's text starts with a ``// <path> :: <unit>``
header, matching the JS and Vue anchor style.

Returns a list of ``{"text", "metadata"}`` dicts using the shared chunk
shape (``source_path`` / ``breadcrumb`` / ``chunk_index``).

Skeleton for now: the flow is laid out as comments and filled in next.

Note: which JSON roles to actually index (keep schemas and translations,
drop test fixtures, etc.) is a content decision still pending the team.
Note: this file is ``json.py`` per the fixed layout; make sure ``import
json`` inside it still resolves to the standard library, not this module.
"""

# Will use the standard library ``json`` to parse, plus langchain JSON/text
# splitters for oversized payloads (uncomment when implemented):
#   import json
#   from langchain_text_splitters import RecursiveJsonSplitter


# Split one JSON file into chunks, according to the file's role.
def chunk_json(text, source_path):
    # 1. Classify the file's role from its path/name: schema, translations,
    #    package manifest, test fixture, or other.
    # 2. Parse the JSON text.
    # 3. Split along boundaries that fit that role:
    #        schema   -> one chunk per top-level property (+ a meta chunk)
    #        i18n     -> one chunk per top-level section
    #        manifest -> a single chunk of the useful fields
    #        other    -> structure-preserving split by size
    # 4. Prefix each piece with a ``// <path> :: <unit>`` header and tag
    #    metadata with source_path / breadcrumb / chunk_index.
    # 5. Return the list of {"text", "metadata"} chunks.
    raise NotImplementedError("json chunker skeleton; wired in next")
