"""Answer a question over the Kalisio corpus: embed, retrieve, generate."""

from fastapi import HTTPException

import api.clients.embeddings as embeddings
import api.clients.llm as llm
import api.clients.translation as translation
import api.clients.vectordb as vectordb
from api.config import get_config
from api.logger import get_logger

log = get_logger("api")


# Run /ask end-to-end: embed -> search -> context -> LLM -> answer.
def answer_question(question):
    config = get_config()

    # 1. Embed the question into a query vector
    vector = embeddings.encode(question)

    # 2. Retrieve the top-k most relevant chunks from Qdrant
    chunks = vectordb.search(vector, top_k=config.top_k)
    _ensure_indexed(chunks)

    # 3. Build the LLM context, respecting the character budget
    context = _build_llm_context(chunks, config.max_context_chars)

    # 4. Call the LLM and return the answer with its sources
    prompt = config.prompt_template.format(context=context, question=question)
    response = llm.ask(prompt)
    return {
        "answer": response.answer,
        "sources": chunks,
        "provider": response.provider,
        "model": response.model,
    }


# Run /search: embed the query, return top-k raw chunks without calling the
# LLM. The chunks are returned as a bare list -- that is the agent-facing
# contract of the endpoint.
def search_chunks(query, top_k):
    chunks = _search_every_phrasing(_phrasings(query), top_k)
    _ensure_indexed(chunks)
    return chunks


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# The query as typed, plus its translation when the corpus is written in
# another language. Both are kept: the translation is what usually finds
# the right page, but the original still carries the technical terms, which
# are spelled the same in either language and often match on their own.
def _phrasings(query):
    way = translation.direction()
    if way is None:
        return [query]
    source, _ = way
    if not translation.looks_like(query, source):
        return [query]
    translated = translation.translate(query)
    if not translated or translated.strip().lower() == query.strip().lower():
        return [query]
    log.info("query translated from %s for retrieval", source)
    return [query, translated]


# Search once per phrasing and merge, keeping the best score each chunk
# reached. Searching twice costs one more embedding and one more query --
# milliseconds -- against a search that would otherwise miss its page.
def _search_every_phrasing(phrasings, top_k):
    best = {}
    for phrasing in phrasings:
        for chunk in vectordb.search(embeddings.encode(phrasing), top_k=top_k):
            key = (chunk["repo"], chunk["path"], chunk["chunk_index"])
            if key not in best or best[key]["score"] < chunk["score"]:
                best[key] = chunk
    ranked = sorted(best.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]



# Guard against querying before ingestion has run. Empty results plus an
# empty collection means the corpus was never indexed: answer 503 naming the
# ingestion command. A non-empty collection with no hits is a genuine
# "no match" and passes.
def _ensure_indexed(chunks):
    if chunks or vectordb.count_chunks() > 0:
        return
    log.warning("query hit an empty index -- ingestion has not run yet")
    raise HTTPException(
        status_code=503,
        detail=("the knowledge index is empty -- run "
                "`python -m ingestion.bin` to index the corpus first"),
    )


# Concatenate chunks (header + content) while staying under the char budget.
def _build_llm_context(chunks, max_chars):
    parts = []
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        block = _format_chunk_block(index, chunk)

        # Stop before the next block would exceed the budget
        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


# Format a single chunk's metadata header followed by its content.
def _format_chunk_block(index, chunk):
    return (
        f"[Chunk {index}]\n"
        f"Source: {chunk['path']}:{chunk['lines']}\n"
        f"Breadcrumb: {chunk['breadcrumb']}\n"
        f"Score: {chunk['score']:.4f}\n"
        f"Content:\n{chunk['content']}"
    )
