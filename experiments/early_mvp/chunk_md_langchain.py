"""
Chunker for Markdown files using LangChain RecursiveCharacterTextSplitter.

Mirrors chunk_md.py but uses LangChain's splitter instead of manual ## splitting.
Produces the same output format (JSONL) for use with embed_chunks.py and
ingest_chunks_to_qdrant.py.

Usage:
    conda activate knowledge && python scripts/chunk_md_langchain.py
"""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "data" / "kdk" / "docs"
OUTPUT_PATH = ROOT_DIR / "outputs" / "md_chunks_langchain.jsonl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)


def get_doc_title(lines: list[str], fallback: str) -> str:
    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return fallback


def chunk_markdown_file(filepath: Path) -> list[dict]:
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()
    rel_path = str(filepath.relative_to(DOCS_DIR))
    doc_title = get_doc_title(lines, filepath.stem)

    docs = splitter.create_documents(
        texts=[text],
        metadatas=[{
            "source": rel_path,
            "doc_title": doc_title,
            "section_title": doc_title,
        }],
    )

    chunks = []
    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        if not content:
            continue
        chunks.append({
            "text": content,
            "metadata": {
                **dict(doc.metadata),
                "chunk_index": i,
                "char_count": len(content),
            },
        })
    return chunks


def process_all_markdown_files():
    md_files = sorted(DOCS_DIR.rglob("*.md"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for filepath in md_files:
            chunks = chunk_markdown_file(filepath)
            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)

    print(f"Processed {len(md_files)} files")
    print(f"Wrote {total_chunks} chunks")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    process_all_markdown_files()
