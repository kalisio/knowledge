"""Best-effort line range computation for chunks.

Chunkers each inject their own breadcrumb header (``// path :: symbol`` for
JS/JSON, ``<!-- path [template] -->`` for Vue, ``Context: ...\\nSource: ...``
for Markdown D). After stripping the header, this helper locates the
chunk's first substantive line in the original source text and returns a
1-indexed ``(start_line, end_line)`` pair.

For chunkers that splice non-contiguous source regions (e.g. Markdown
``C_ast_merge`` / ``D_ast_breadcrumb`` joining atoms across paragraphs, or
JSON whose splitter re-serialises parsed data), the start_line is the
first atom's position and the end_line is ``start_line + newlines in
chunk content`` — a size estimate, not a literal source span.

When the anchor cannot be located (minified JSON, transformed content,
fallback splitters), returns ``(1, total_source_lines)`` so callers always
get a usable range.
"""

from __future__ import annotations


_ANCHOR_LEN = 64
_SHORT_ANCHOR_LEN = 16


def _total_lines(source_text: str) -> int:
    if not source_text:
        return 1
    return max(1, len(source_text.splitlines()))


def _strip_prefix(chunk_text: str) -> str:
    """Strip known chunker-injected breadcrumb headers."""
    lines = chunk_text.split("\n")
    if not lines:
        return chunk_text

    first = lines[0]

    if first.startswith("Context:"):
        drop = 1
        if len(lines) > drop and lines[drop].startswith("Source:"):
            drop += 1
        while drop < len(lines) and lines[drop] == "":
            drop += 1
        return "\n".join(lines[drop:])

    if first.startswith("// ") or first.startswith("<!-- "):
        return "\n".join(lines[1:])

    return chunk_text


def _anchor(text: str) -> str:
    stripped = text.lstrip()
    return stripped[:_ANCHOR_LEN] if stripped else ""


def compute_line_range(chunk_text: str, source_text: str) -> tuple[int, int]:
    """Return 1-indexed ``(start_line, end_line)`` for a chunk in source.

    Always returns a usable range. Falls back to ``(1, total_lines)`` if
    the chunk content cannot be located (e.g. JSON re-serialisation,
    minified source, transformed text).
    """
    total = _total_lines(source_text)
    content = _strip_prefix(chunk_text)
    anchor = _anchor(content)

    if not anchor or not source_text:
        return 1, total

    pos = source_text.find(anchor)
    if pos < 0 and len(anchor) > _SHORT_ANCHOR_LEN:
        pos = source_text.find(anchor[:_SHORT_ANCHOR_LEN])
    if pos < 0:
        return 1, total

    start_line = source_text.count("\n", 0, pos) + 1
    end_line = start_line + content.count("\n")
    if end_line > total:
        end_line = total
    return start_line, end_line
