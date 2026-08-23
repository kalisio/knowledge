"""Chunking checked against reference files, one supported type each.

For every sample in test/data/sources there is a reference in
test/data/expected listing the chunks it must produce: their order, their
breadcrumb, their line range and their exact text. Two things are checked
against it:

  1. the chunker still produces exactly those chunks (a regression fence);
  2. the line range each chunk claims really points at that chunk's content
     in the SOURCE file -- the reference cannot drift away from reality
     without the source and the reference disagreeing.

Regenerate the references after an intentional chunking change:

    UPDATE_CHUNK_REFERENCES=1 pytest test/unit/test_chunk_references.py
"""

import json
import os
from pathlib import Path

import pytest

from ingestion.chunkers import _CHUNKERS

DATA = Path(__file__).resolve().parent.parent / "data"

# Each sample and the repo-relative path it is chunked as. The path shows up
# in the chunk headers, so it is part of what the reference pins down.
SAMPLES = {
    "guide.md": "docs/catalog/guide.md",
    "geolocation.js": "core/client/geolocation.js",
    "KLayerList.vue": "client/components/KLayerList.vue",
    "en.json": "client/i18n/en.json",
}


# The sample text, the path it is chunked as, and the chunks it produces.
@pytest.fixture(params=sorted(SAMPLES), ids=sorted(SAMPLES))
def sample(request):
    name = request.param
    path = SAMPLES[name]
    text = (DATA / "sources" / name).read_text(encoding="utf-8")
    chunker = _CHUNKERS[Path(name).suffix]
    chunks = chunker(text, path)
    reference = _reference(name, path, chunks, text.splitlines())
    return SimpleSample(name, path, text, chunks, reference)


class SimpleSample:
    def __init__(self, name, path, text, chunks, reference):
        self.name = name
        self.path = path
        self.text = text
        self.lines = text.splitlines()
        self.chunks = chunks
        self.reference = reference


# --- the chunks are the ones the reference describes -----------------------


def test_the_chunk_count_matches_the_reference(sample):
    assert len(sample.chunks) == len(sample.reference["chunks"])


def test_every_chunk_matches_the_reference(sample):
    for chunk, expected in zip(sample.chunks, sample.reference["chunks"]):
        metadata = chunk["metadata"]
        assert metadata["chunk_index"] == expected["chunk_index"]
        assert metadata["breadcrumb"] == expected["breadcrumb"]
        assert metadata["start_line"] == expected["start_line"]
        assert metadata["end_line"] == expected["end_line"]
        assert chunk["text"] == expected["text"]


def test_every_chunk_states_the_path_it_was_cut_from(sample):
    for chunk in sample.chunks:
        assert chunk["metadata"]["path"] == sample.path


# --- the line ranges point back at the source file -------------------------


def test_the_line_range_is_inside_the_file(sample):
    for chunk in sample.chunks:
        start = chunk["metadata"]["start_line"]
        end = chunk["metadata"]["end_line"]
        assert 1 <= start <= end <= len(sample.lines), chunk["text"][:60]


def test_the_range_of_each_chunk_holds_what_the_reference_says(sample):
    # The reference stores the excerpt each range points at. Reading it back
    # from the source file is what ties the two files together: the sample
    # cannot be edited without this failing until the reference is redone.
    for expected in sample.reference["chunks"]:
        excerpt = "\n".join(
            sample.lines[expected["start_line"] - 1:expected["end_line"]])
        assert excerpt == expected["excerpt"], (
            f"lines {expected['start_line']}-{expected['end_line']} of "
            f"{sample.name} no longer hold what the reference expects")


def test_each_chunk_opens_the_file_on_something_recognisable(sample):
    # Whatever the chunker did to the text, the first line of the range has
    # to be where a reader would want the file opened: the heading, the
    # declaration, the tag or the key the chunk is about.
    for chunk in sample.chunks:
        opening = sample.lines[chunk["metadata"]["start_line"] - 1].strip()
        breadcrumb = chunk["metadata"]["breadcrumb"]
        body = _body_lines(chunk)
        assert opening, "a chunk points at a blank line"
        assert (opening in chunk["text"]
                or breadcrumb.split(" > ")[-1].lower() in opening.lower()
                or opening.strip("{[ ") in body[0]), (
            f"{sample.name} chunk {chunk['metadata']['chunk_index']} starts "
            f"on {opening!r}, unrelated to the chunk")


def test_the_content_of_the_range_covers_the_chunk(sample):
    # Everything the chunk quotes from the file has to be inside the range it
    # claims -- that is what lets a reader open the file there and find it.
    for chunk in sample.chunks:
        metadata = chunk["metadata"]
        excerpt = "\n".join(
            sample.lines[metadata["start_line"] - 1:metadata["end_line"]])
        for line in _source_lines(chunk, sample):
            assert line.strip() in excerpt, (
                f"{sample.name} chunk {metadata['chunk_index']}: "
                f"{line.strip()!r} is outside lines "
                f"{metadata['start_line']}-{metadata['end_line']}")


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# The chunk's lines that were taken verbatim from the file, and that say
# something. JSON units are re-serialised rather than sliced, so a line made
# only of delimiters ("{", "},", "]") is structure the serialiser produced,
# not a line traceable to one place in the file.
def _source_lines(chunk, sample):
    source = set(line.strip() for line in sample.lines if line.strip())
    return [line for line in chunk["text"].splitlines()[1:]
            if line.strip() in source and _is_content(line)]


# Whether a line carries content rather than pure structure.
def _is_content(line):
    return bool(line.strip(" \t{}[](),;"))


# The chunk's own lines, without the generated header.
def _body_lines(chunk):
    return [line for line in chunk["text"].splitlines()[1:] if line.strip()]


# One chunk as the reference records it: what the chunker produced, plus the
# excerpt of the source file its line range points at.
def _expected(chunk, source_lines):
    metadata = chunk["metadata"]
    return {
        "chunk_index": metadata["chunk_index"],
        "breadcrumb": metadata["breadcrumb"],
        "start_line": metadata["start_line"],
        "end_line": metadata["end_line"],
        "text": chunk["text"],
        "excerpt": "\n".join(
            source_lines[metadata["start_line"] - 1:metadata["end_line"]]),
    }


# Load the reference, or rewrite it when UPDATE_CHUNK_REFERENCES is set.
def _reference(name, path, chunks, source_lines):
    reference_path = DATA / "expected" / f"{name}.json"
    if os.environ.get("UPDATE_CHUNK_REFERENCES"):
        reference_path.write_text(json.dumps({
            "source": f"sources/{name}",
            "path": path,
            "chunks": [_expected(chunk, source_lines) for chunk in chunks],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json.loads(reference_path.read_text(encoding="utf-8"))
