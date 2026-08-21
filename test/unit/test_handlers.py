import pytest
from fastapi import HTTPException

import api.handlers as handlers


# A search-result chunk as read_payload returns it (what /search serves and
# what _build_llm_context consumes).
def make_chunk(content="export function center () {}",
               source_path="map/base.js", breadcrumb="base > center",
               score=0.5):
    return {"source_path": source_path, "breadcrumb": breadcrumb,
            "content": content, "score": score}


def block(index, chunk):
    return handlers._format_chunk_block(index, chunk)


# --- _format_chunk_block: the exact shape the LLM sees --------------------

def test_format_chunk_block_renders_header_then_content():
    text = block(1, make_chunk(score=0.5))
    assert text == ("[Chunk 1]\n"
                    "Source: map/base.js\n"
                    "Breadcrumb: base > center\n"
                    "Score: 0.5000\n"
                    "Content:\nexport function center () {}")


# --- _build_llm_context: the character budget -----------------------------
# The only place in the project that silently drops data: chunks beyond the
# budget never reach the LLM.

def test_context_keeps_every_chunk_within_the_budget():
    chunks = [make_chunk(content="aaa"), make_chunk(content="bbb")]

    context = handlers._build_llm_context(chunks, max_chars=10_000)

    assert context == block(1, chunks[0]) + "\n\n" + block(2, chunks[1])


def test_context_drops_the_chunks_beyond_the_budget():
    chunks = [make_chunk(content="aaa"), make_chunk(content="bbb")]
    budget = len(block(1, chunks[0]))  # exactly one block fits

    context = handlers._build_llm_context(chunks, budget)

    assert context == block(1, chunks[0])


def test_context_is_empty_when_the_first_chunk_alone_overflows():
    # Documented current behaviour: no partial chunk, no error -- the LLM is
    # then called with an empty context. If this ever changes to raising or
    # truncating inside a chunk, this test is the place to update.
    chunk = make_chunk(content="x" * 100)

    assert handlers._build_llm_context([chunk], max_chars=10) == ""


def test_context_stops_at_the_first_overflow_instead_of_skipping_it():
    # Chunks arrive ordered by relevance, so the cut is a break, not a
    # skip: a smaller, less relevant chunk must not leapfrog a bigger, more
    # relevant one that missed the budget.
    small_1 = make_chunk(content="aaa")
    big = make_chunk(content="b" * 500)
    small_2 = make_chunk(content="ccc")
    budget = len(block(1, small_1)) + len(block(2, small_2)) + 2

    context = handlers._build_llm_context([small_1, big, small_2], budget)

    assert context == block(1, small_1)


# --- _ensure_indexed: empty index vs genuine no-match ---------------------

def test_results_pass_without_consulting_the_collection(monkeypatch):
    # With hits in hand the guard must not cost a Qdrant round-trip.
    def unexpected_call():
        raise AssertionError("get_code_collection_length was called")
    monkeypatch.setattr(handlers.vectordb, "get_code_collection_length",
                        unexpected_call)

    assert handlers._ensure_indexed([make_chunk()]) is None


def test_no_hits_on_a_populated_index_is_a_genuine_no_match(monkeypatch):
    monkeypatch.setattr(handlers.vectordb, "get_code_collection_length",
                        lambda: 42)

    assert handlers._ensure_indexed([]) is None


def test_no_hits_on_an_empty_index_answers_503_naming_ingestion(monkeypatch):
    monkeypatch.setattr(handlers.vectordb, "get_code_collection_length",
                        lambda: 0)

    with pytest.raises(HTTPException) as excinfo:
        handlers._ensure_indexed([])

    assert excinfo.value.status_code == 503
    assert "ingestion" in excinfo.value.detail
