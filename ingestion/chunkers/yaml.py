"""Cuts a YAML file into chunks, one per top-level key."""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.chunkers.line_locator import LineLocator
from ingestion.config import get_config

# The file is cut on indentation, never parsed: a values.yaml.gotmpl is not
# valid YAML until helmfile has rendered it, and a Helm template is not
# either. Indentation is the one structure both keep.

# A line that opens a block at a given column: a key, or a list item. The
# indentation is captured so the same pattern serves every nesting level.
_BLOCK_OPENER = re.compile(r"^( *)(?:-\s+)?([^\s#{][^:\n]*?)\s*:(?:\s|$)|^( *)-\s")

# What separates documents in one file.
_DOCUMENT_SEPARATOR = re.compile(r"^---\s*$", re.M)

# The resource kind a Kubernetes document declares. The corpus is Helm
# charts and cluster configurations before it is anything else, and
# "ServiceAccount > spec" says more than "document 2 > spec".
_KIND = re.compile(r"^kind:\s*([A-Za-z]+)\s*$", re.M)

# A Go template comment spans lines; everything inside attaches to the block
# that follows, like any other comment.
_TEMPLATE_COMMENT_OPEN = "{{/*"
_TEMPLATE_COMMENT_CLOSE = "*/}}"

# Lines that never open a block and attach to the block that follows: a
# comment, a template directive on its own line, an empty line.
_PREFIX_LINE = re.compile(r"^\s*(?:#|\{\{|$)")


# Chunk one YAML file: a chunk per top-level key, subdivided on the next
# level when a key is too large for one chunk -- `jobs` in a workflow, `env`
# in a values file.
def chunk_yaml(text, path):
    config = get_config()
    locator = LineLocator(text)
    chunks = []
    documents = _documents(text)
    for document_index, (start, end) in enumerate(documents):
        crumbs = _document_name(text, start, end, document_index,
                                len(documents))
        units = _merged(_units(text, start, end, config), config)
        for keys, unit_start, unit_end in units:
            body = _trimmed(text[unit_start:unit_end])
            if not body.strip():
                continue
            breadcrumb = " > ".join(crumbs + keys)
            for piece in _sized(body, config):
                lines = locator.locate(piece) or locator.whole_file()
                chunks.append(_chunk(path, breadcrumb, piece, len(chunks),
                                     lines))
    return chunks


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# (start, end) offsets of each document, split on `---`. A file with no
# separator is one document; an empty document (a leading `---`) is none.
def _documents(text):
    bounds = []
    start = 0
    for match in _DOCUMENT_SEPARATOR.finditer(text):
        bounds.append((start, match.start()))
        start = match.end()
    bounds.append((start, len(text)))
    return [(s, e) for s, e in bounds if text[s:e].strip()]


# What names a document in the breadcrumb: nothing for a single one, its
# kind when it declares one, its position otherwise.
def _document_name(text, start, end, index, count):
    kind = _KIND.search(text, start, end)
    if kind:
        return [kind.group(1)]
    return [f"document {index + 1}"] if count > 1 else []


# A unit's body without the blank lines around it. Leading ones come from
# the separator between the previous key and this one's comments.
def _trimmed(body):
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# The units of one document: [(parent keys, leaf key, start, end)]. A
# top-level block that fits stays whole; one that does not is cut on its own
# children, and a child that still does not fit is left to the size splitter.
def _units(text, start, end, config):
    limit = config.chunk_size * 2
    units = []
    for key, block_start, block_end in _blocks(text, start, end, indent=0,
                                               detach_over=config.chunk_size):
        if block_end - block_start <= limit or key is None:
            units.append(([], key, block_start, block_end))
            continue
        # The children start after the opener line itself: `jobs:` alone is
        # not a chunk, and the breadcrumb carries its name to every child.
        # A child is indented deeper than its parent, or it is not a child:
        # a block that is large only because of the comment attached to it
        # has none, and must not be cut on its own siblings.
        child_indent = _child_indent(text, block_start, block_end)
        body_start = _after_opener(text, block_start, block_end)
        children = (_blocks(text, body_start, block_end, indent=child_indent,
                            detach_over=config.chunk_size)
                    if child_indent > 0 else [])
        if len(children) < 2:
            units.append(([], key, block_start, block_end))
            continue
        for child, child_start, child_end in children:
            units.append(([key], child, child_start, child_end))
    return units


# The offset just past the line that opens a block -- past its attached
# comments and past the `key:` line.
def _after_opener(text, start, end):
    offset = start
    for line in text[start:end].splitlines(keepends=True):
        offset += len(line)
        if not _PREFIX_LINE.match(line):
            return offset
    return end


# Adjacent units small enough to share a chunk are merged: a values file
# opens on six one-line keys, and six one-line chunks say nothing on their
# own. Only siblings merge -- the same parent -- so a child of `jobs` never
# swallows a top-level key that follows it. The result is [(key path,
# start, end)], the merged leaves named together, up to four of them.
def _merged(units, config):
    groups = []
    for parent, leaf, start, end in units:
        if groups:
            previous_parent, leaves, group_start, _ = groups[-1]
            if (previous_parent == parent and leaf and leaves
                    and end - group_start <= config.chunk_size):
                groups[-1] = (parent, leaves + [leaf], group_start, end)
                continue
        groups.append((parent, [leaf] if leaf else [], start, end))
    return [(parent + _named(leaves), start, end)
            for parent, leaves, start, end in groups]


# "env" for one leaf, "image, forceRestart, replicaCount, hostOverride, …"
# for several, nothing for a block that has no key at all.
def _named(leaves):
    if not leaves:
        return []
    if len(leaves) == 1:
        return leaves
    return [", ".join(leaves[:4]) + (", …" if len(leaves) > 4 else "")]


# The blocks opened at `indent` between `start` and `end`, as
# [(key or None, start, end)]. Comments, blank lines and template directives
# right before an opener belong to it: a `# must stay in step with ...`
# describes the key under it, not the one above. A prefix longer than
# `detach_over` is a block of its own instead -- the twenty-line header of
# a template describes the file, not its first key, and would otherwise
# push that key past the size limit and have it cut on characters.
def _blocks(text, start, end, indent, detach_over):
    lines = text[start:end].splitlines(keepends=True)
    openers = []                         # (key, start) -- a None key is a
    offset = start                       # detached prefix
    pending = None                       # where the current prefix started
    in_comment = False                   # inside a {{/* ... */}}
    for line in lines:
        stripped = line.strip()
        if in_comment or stripped.startswith(_TEMPLATE_COMMENT_OPEN):
            in_comment = _TEMPLATE_COMMENT_CLOSE not in stripped
            if pending is None:
                pending = offset
        elif _PREFIX_LINE.match(line):
            if pending is None:
                pending = offset
        else:
            match = _BLOCK_OPENER.match(line)
            column = len(match.group(1) or match.group(3) or "") if match else -1
            if match and column == indent:
                key = (match.group(2) or "").strip() or None
                block_start = offset
                if pending is not None:
                    if offset - pending > detach_over:
                        openers.append((None, pending))
                    else:
                        block_start = pending
                openers.append((key, block_start))
            pending = None
        offset += len(line)

    if not openers:
        return [(None, start, end)]
    blocks = []
    # What sits before the first opener -- a leading comment, a template
    # header -- is a block of its own.
    if openers[0][1] > start and text[start:openers[0][1]].strip():
        blocks.append((None, start, openers[0][1]))
    for index, (key, block_start) in enumerate(openers):
        block_end = openers[index + 1][1] if index + 1 < len(openers) else end
        blocks.append((key, block_start, block_end))
    return blocks


# The indentation of the first child line of a block, or -1 if it has none.
def _child_indent(text, start, end):
    first = True
    for line in text[start:end].splitlines():
        if _PREFIX_LINE.match(line):
            continue
        if first:
            first = False                # the opener itself
            continue
        return len(line) - len(line.lstrip(" "))
    return -1


# A unit that still exceeds the chunk size is cut on characters, keeping its
# breadcrumb -- better a large key in three pieces than a large key dropped.
def _sized(body, config):
    if len(body) <= config.chunk_size * 2:
        return [body]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    return splitter.split_text(body)


def _chunk(path, breadcrumb, body, index, lines):
    header = f"# {path}" + (f" :: {breadcrumb}" if breadcrumb else "")
    start_line, end_line = lines
    return {
        "text": header + "\n" + body,
        "metadata": {
            "path": path,
            "chunk_index": index,
            "breadcrumb": breadcrumb,
            "start_line": start_line,
            "end_line": end_line,
        },
    }
