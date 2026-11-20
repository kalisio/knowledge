import pytest
from fastapi import HTTPException

import api.services.retrieval as retrieval


# A search-result chunk as read_payload returns it (what /search serves and
# what _build_llm_context consumes).
def make_chunk(content="export function center () {}",
               path="map/base.js", lines="1-3", breadcrumb="base > center",
               score=0.5):
    return {"path": path, "lines": lines, "breadcrumb": breadcrumb,
            "content": content, "score": score}


def block(index, chunk):
    return retrieval._format_chunk_block(index, chunk)


# --- _format_chunk_block: the exact shape the LLM sees --------------------

def test_format_chunk_block_renders_header_then_content():
    text = block(1, make_chunk(score=0.5))
    assert text == ("[Chunk 1]\n"
                    "Source: map/base.js:1-3\n"
                    "Breadcrumb: base > center\n"
                    "Score: 0.5000\n"
                    "Content:\nexport function center () {}")


# --- _build_llm_context: the character budget -----------------------------
# The only place in the project that silently drops data: chunks beyond the
# budget never reach the LLM.

def test_context_keeps_every_chunk_within_the_budget():
    chunks = [make_chunk(content="aaa"), make_chunk(content="bbb")]

    context = retrieval._build_llm_context(chunks, max_chars=10_000)

    assert context == block(1, chunks[0]) + "\n\n" + block(2, chunks[1])


def test_context_drops_the_chunks_beyond_the_budget():
    chunks = [make_chunk(content="aaa"), make_chunk(content="bbb")]
    budget = len(block(1, chunks[0]))  # exactly one block fits

    context = retrieval._build_llm_context(chunks, budget)

    assert context == block(1, chunks[0])


def test_context_is_empty_when_the_first_chunk_alone_overflows():
    # Documented current behaviour: no partial chunk, no error -- the LLM is
    # then called with an empty context. If this ever changes to raising or
    # truncating inside a chunk, this test is the place to update.
    chunk = make_chunk(content="x" * 100)

    assert retrieval._build_llm_context([chunk], max_chars=10) == ""


def test_context_stops_at_the_first_overflow_instead_of_skipping_it():
    # Chunks arrive ordered by relevance, so the cut is a break, not a
    # skip: a smaller, less relevant chunk must not leapfrog a bigger, more
    # relevant one that missed the budget.
    small_1 = make_chunk(content="aaa")
    big = make_chunk(content="b" * 500)
    small_2 = make_chunk(content="ccc")
    budget = len(block(1, small_1)) + len(block(2, small_2)) + 2

    context = retrieval._build_llm_context([small_1, big, small_2], budget)

    assert context == block(1, small_1)


# --- _ensure_indexed: empty index vs genuine no-match ---------------------

def test_results_pass_without_consulting_the_collection(monkeypatch):
    # With hits in hand the guard must not cost a Qdrant round-trip.
    def unexpected_call():
        raise AssertionError("count_chunks was called")
    monkeypatch.setattr(retrieval.vectordb, "count_chunks",
                        unexpected_call)

    assert retrieval._ensure_indexed([make_chunk()]) is None


def test_no_hits_on_a_populated_index_is_a_genuine_no_match(monkeypatch):
    monkeypatch.setattr(retrieval.vectordb, "count_chunks",
                        lambda: 42)

    assert retrieval._ensure_indexed([]) is None


def test_no_hits_on_an_empty_index_answers_503_naming_ingestion(monkeypatch):
    monkeypatch.setattr(retrieval.vectordb, "count_chunks",
                        lambda: 0)

    with pytest.raises(HTTPException) as excinfo:
        retrieval._ensure_indexed([])

    assert excinfo.value.status_code == 503
    assert "ingestion" in excinfo.value.detail


# --- _phrasings: the query and its translations ---------------------------

def test_without_translation_the_query_is_searched_as_typed(monkeypatch):
    monkeypatch.setattr(retrieval.translation, "translations",
                        lambda query: [])

    assert retrieval._phrasings("how to add a layer") == ["how to add a layer"]


# The original comes first and stays: it carries the technical terms, and it
# is the phrasing that reaches the pages written in the question's language.
def test_each_translation_is_searched_after_the_original(monkeypatch):
    monkeypatch.setattr(
        retrieval.translation, "translations",
        lambda query: [("en", "how to add a layer"),
                       ("es", "cómo añadir una capa")])

    assert retrieval._phrasings("comment ajouter une couche") == [
        "comment ajouter une couche", "how to add a layer",
        "cómo añadir una capa"]


# A translation that only gives the query back -- an identifier, a product
# name -- would cost a second search for nothing.
def test_a_translation_identical_to_the_query_is_dropped(monkeypatch):
    monkeypatch.setattr(retrieval.translation, "translations",
                        lambda query: [("en", " KDK Planet ")])

    assert retrieval._phrasings("kdk planet") == ["kdk planet"]


# --- _search_every_phrasing: one ranking out of several -------------------

def hit(path, score, repo="kdk", chunk_index=0):
    return {"repo": repo, "path": path, "chunk_index": chunk_index,
            "score": score, "lines": "1", "breadcrumb": "", "content": ""}


def stub_search(monkeypatch, results):
    monkeypatch.setattr(retrieval.embeddings, "encode", lambda text: text)
    monkeypatch.setattr(retrieval.vectordb, "search",
                        lambda vector, top_k: results[vector])


# A chunk both phrasings find is returned once, with the best score it
# reached -- the scores come from one model and one cosine, so they compare.
def test_a_chunk_found_twice_keeps_its_best_score(monkeypatch):
    stub_search(monkeypatch, {
        "fr": [hit("a.md", 0.61), hit("b.md", 0.55)],
        "en": [hit("a.md", 0.83), hit("c.md", 0.70)],
    })

    ranked = retrieval._search_every_phrasing(["fr", "en"], top_k=5)

    assert [(c["path"], c["score"]) for c in ranked] == [
        ("a.md", 0.83), ("c.md", 0.70), ("b.md", 0.55)]


def test_the_merged_ranking_is_cut_at_top_k(monkeypatch):
    stub_search(monkeypatch, {
        "fr": [hit("a.md", 0.9), hit("b.md", 0.5)],
        "en": [hit("c.md", 0.8), hit("d.md", 0.4)],
    })

    ranked = retrieval._search_every_phrasing(["fr", "en"], top_k=2)

    assert [c["path"] for c in ranked] == ["a.md", "c.md"]


# Two files may share a path in two repositories, and two chunks a file:
# neither is the same chunk.
def test_chunks_are_told_apart_by_repo_path_and_index(monkeypatch):
    stub_search(monkeypatch, {
        "q": [hit("README.md", 0.9, repo="kdk"),
              hit("README.md", 0.8, repo="kano"),
              hit("README.md", 0.7, repo="kdk", chunk_index=1)],
    })

    assert len(retrieval._search_every_phrasing(["q"], top_k=5)) == 3


# --- answer_question: /ask retrieves the way /search does -----------------

# Before, /ask searched the question as typed while /search also searched
# its translation: the same French question reached the English page from
# one endpoint and missed it from the other.
def test_ask_searches_the_translations_too(monkeypatch):
    monkeypatch.setattr(retrieval.translation, "translations",
                        lambda query: [("en", "how to add a layer")])
    stub_search(monkeypatch, {
        "comment ajouter une couche": [hit("fr.md", 0.40)],
        "how to add a layer": [hit("layers.md", 0.82)],
    })
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return retrieval.llm.LLMResponse(answer="ok", provider="local",
                                         model="test-llm")
    monkeypatch.setattr(retrieval.llm, "ask", ask)

    response = retrieval.answer_question("comment ajouter une couche")

    assert [c["path"] for c in response["sources"]] == ["layers.md", "fr.md"]
    assert "Source: layers.md" in prompts[0]
    # The LLM is asked the question as the user typed it, not its
    # translation: it answers in the user's language.
    assert "Question: comment ajouter une couche" in prompts[0]


def test_search_and_ask_share_one_retrieval(monkeypatch):
    seen = []

    def retrieve(query, top_k):
        seen.append((query, top_k))
        return [hit("a.md", 0.9)]
    monkeypatch.setattr(retrieval, "retrieve", retrieve)
    monkeypatch.setattr(
        retrieval.llm, "ask",
        lambda prompt: retrieval.llm.LLMResponse("ok", "local", "test-llm"))

    retrieval.search_chunks("how to add a layer", 3)
    retrieval.answer_question("how to add a layer")

    assert seen == [("how to add a layer", 3), ("how to add a layer", 6)]
