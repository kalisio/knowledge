"""Split a Vue single-file component (SFC) into chunks for indexing.

A Vue SFC is built from ``<template>``, ``<script>`` and ``<style>`` blocks.
Each block is chunked with a splitter suited to its language (the script
block like JavaScript, the template like HTML, the style as CSS), and every
chunk is tagged with the block it came from. Like the other chunkers, each
chunk's text starts with a header naming the file and block so it is
self-describing once embedded.

Returns a list of ``{"text", "metadata"}`` dicts using the shared chunk
shape (``source_path`` / ``breadcrumb`` / ``chunk_index``) plus a
``block_type`` field ("template" / "script" / "style").

Skeleton for now: the flow is laid out as comments and filled in next.
"""

# Will reuse the JavaScript splitter for <script>, and HTML/CSS splitters
# for <template>/<style> (uncomment when implemented):
#   import re
#   from langchain_text_splitters import (
#       Language, RecursiveCharacterTextSplitter)


# Split one Vue SFC into per-block chunks.
def chunk_vue(text, source_path):
    # 1. Find the SFC blocks: each <template> / <script> / <style> section.
    # 2. Optionally derive a readable component name from the file name, to
    #    help chunks match natural-language queries.
    # 3. For each non-empty block, pick a splitter by block type:
    #        script   -> JavaScript-aware splitter
    #        template -> HTML-aware splitter
    #        style    -> kept whole if small, else split by size
    # 4. Prefix each piece with a header naming the file and block, and tag
    #    metadata with source_path / breadcrumb / chunk_index / block_type.
    # 5. Return the list of {"text", "metadata"} chunks.
    raise NotImplementedError("vue chunker skeleton; wired in next")
