"""
Chunker for Markdown files.

Usage:
    conda activate knowledge && python scripts/chunk_md.py
"""

from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "data" / "kdk" / "docs"
OUTPUT_PATH = ROOT_DIR / "outputs" / "md_chunks.jsonl"
MAX_CHARS = 1500
OVERLAP_CHARS = 200

def read_one_md_file():
    test_file = DOCS_DIR / "about" / "introduction.md"
    text = test_file.read_text(encoding="utf-8")
    print(f"File path: {test_file}")
    print(f"File size: {len(text)} characters")
    print("First 200 characters:")
    print(text[:200])

def find_all_md_files():
    list_md_files = []
    md_files = list(DOCS_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} Markdown files in {DOCS_DIR}")
    for md_file in md_files:
        list_md_files.append(md_file)
    print(f"Total Markdown files found: {len(list_md_files)}")    
    print(f"First 5 Markdown files: {list_md_files[:5]}")

def split_md_file_by_two_dieses():
    test_file = DOCS_DIR / "about" / "introduction.md"
    text = test_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    chunks = []
    current_title = None
    current_chunk = []
    
    for line in lines:
        if line.startswith("## "):
            if current_chunk:
                chunks.append(
                    {
                        "title": current_title,
                        "content": "\n".join(current_chunk)
                    }

                )
            current_title = line[3:].strip()  # Remove "## " prefix
            current_chunk = []
        else:
            current_chunk.append(line)

    # Append the last chunk if it exists
    if current_chunk:
        chunks.append(
            {
                "title": current_title,
                "content": "\n".join(current_chunk)
            }
            
        )
    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1} (length: {len(chunk['content'])} characters):")
        print(f"Title: {chunk['title']}")
        print(chunk['content'][:200])  # Print first 200 characters of each chunk
        print("\n---\n")

def preview_chunks_with_metadata():
    test_file = DOCS_DIR / "about" / "introduction.md"
    chunks = chunk_markdown_file(test_file)

    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1} (length: {len(chunk['text'])} characters):")
        print(f"Metadata: {chunk['metadata']}")
        print(chunk['text'][:200])  # Print first 200 characters of each chunk
        print("\n---\n")

def get_doc_title(lines: list[str], fallback_title: str) -> str:
    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return fallback_title


def split_oversized_text(text: str, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)

    return [part for part in parts if part]


def build_chunk(rel_path: Path, doc_title: str, section_title: str, text: str, chunk_index: int) -> dict:
    return {
        "text": text,
        "metadata": {
            "source": str(rel_path),
            "doc_title": doc_title,
            "section_title": section_title,
            "chunk_index": chunk_index,
            "char_count": len(text),
        },
    }


def chunk_markdown_file(filepath: Path) -> list[dict]:
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()
    rel_path = filepath.relative_to(DOCS_DIR)
    doc_title = get_doc_title(lines, filepath.stem)

    sections = []
    current_title = doc_title
    current_lines = []

    for line in lines:
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    chunks = []
    chunk_index = 0
    for section_title, section_text in sections:
        for part in split_oversized_text(section_text):
            chunks.append(build_chunk(rel_path, doc_title, section_title, part, chunk_index))
            chunk_index += 1

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
    read_one_md_file()
    find_all_md_files()
    split_md_file_by_two_dieses()
    preview_chunks_with_metadata()
    process_all_markdown_files()
