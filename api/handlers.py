import logging

import utils.embeddings as embeddings
import utils.vectordb as vectordb
import api.llm as llm
from api.config import get_config
from fastapi import HTTPException


log = logging.getLogger("knowledge.api")

PROMPT_TEMPLATE = """Answer the question based on the context below. \
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


# Run /ask end-to-end: embed → search → context → LLM → answer
def answer_question(question):
    config = get_config()

    # 1. Embed the question into a query vector
    vector = embeddings.encode(question)

    # 2. Retrieve the top-k most relevant chunks from Qdrant
    chunks = vectordb.search(vector, top_k=config.top_k)
    _ensure_indexed(chunks)

    # 3. Build the LLM context, respecting the character budget
    context = build_llm_context(chunks, config.max_context_chars)

    # 4. Call the LLM and return the answer with its sources
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.ask(prompt)
    return {
        "answer": response.answer,
        "sources": chunks,
        "provider": response.provider,
        "model": response.model,
    }


# Run /search: embed the query, return top-k raw chunks without calling the LLM
def search_chunks(query, top_k):
    vector = embeddings.encode(query)
    chunks = vectordb.search(vector, top_k=top_k)
    _ensure_indexed(chunks)
    return {"results": chunks}


# Guard against querying before ingestion has run. Empty results plus an empty
# collection means the corpus was never indexed -- surface an actionable hint
# (503) instead of silently returning nothing. A non-empty collection with no
# hits is a genuine "no match", so we let it through.
def _ensure_indexed(chunks):
    if chunks or vectordb.count() > 0:
        return
    log.warning("query hit an empty index -- ingestion has not run yet")
    raise HTTPException(
        status_code=503,
        detail=("the knowledge index is empty -- run "
                "`python -m ingestion.main` to index the corpus first"),
    )


# Concatenate chunks (header + content) while staying under the char budget
def build_llm_context(chunks, max_chars):
    parts = []
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        block = format_chunk_block(index, chunk)

        # Stop before the next block would exceed the budget
        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


# Format a single chunk's metadata header followed by its content
def format_chunk_block(index, chunk):
    return (
        f"[Chunk {index}]\n"
        f"Source: {chunk['source_path']}\n"
        f"Breadcrumb: {chunk['breadcrumb']}\n"
        f"Score: {chunk['score']:.4f}\n"
        f"Content:\n{chunk['content']}"
    )
