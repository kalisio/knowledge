"""Build an auto-generated query set with gold source files for nb02 evaluation.

Why this exists: `outputs/test_questions.json` has empty `relevant_sources`
for all 46 questions, so it can't be used as ground truth for deterministic
retrieval metrics. This script mines `data/kdk/docs/*.md` to emit queries of
three shapes, each with a gold source file attached:

  - symbol  : an API identifier (e.g. "setupGlobe", "addLayer")
  - section : a section heading text (e.g. "Scene post processing")
  - title   : the page title (h1 or first h2)

Each query's `relevant_sources` is the file it was mined from. If the same
query text is mined from multiple files, their sources are merged.

Output: `outputs/gold_queries.json`

The resulting query set is meant to approximate what an AI coder agent would
actually send to the retriever (identifier- and task-centric), not FAQ-style
questions. That matches the downstream use case: feeding retrieved chunks as
context into a code-generation model.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_filter import scan_corpus  # noqa: E402


# Per-file caps to avoid a single large file dominating the query set.
MAX_SYMBOLS_PER_FILE = 4
MAX_SECTIONS_PER_FILE = 2
MIN_SYMBOL_LEN = 4  # drop noise like `f`, `x`, `id`

# Symbols we don't want as queries (language keywords, too generic).
SYMBOL_BLOCKLIST = {
    "function", "return", "const", "let", "var", "class", "import",
    "export", "async", "await", "true", "false", "null", "undefined",
    "this", "new", "typeof", "switch", "case", "break", "default",
    "string", "number", "boolean", "object", "array", "console", "log",
    "if", "else", "for", "while", "throw", "catch", "try", "require",
    "module", "exports", "Object", "Array", "String", "Number", "Math",
    "Date", "Promise", "JSON", "Error", "Boolean",
}

BOLD_SIG = re.compile(r"\*\*([a-zA-Z_$][\w$]*)\s*\(")
INLINE_CODE_SIG = re.compile(r"`([a-zA-Z_$][\w$]*)\s*\(")
FENCED_CODE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
FN_DECL = re.compile(r"\b(?:function\s+|const\s+|let\s+|class\s+)([a-zA-Z_$][\w$]*)")
H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)
H2 = re.compile(r"^##\s+(.+?)\s*$", re.M)
H3 = re.compile(r"^###\s+(.+?)\s*$", re.M)


def extract_symbols(text: str) -> list[str]:
    """Return symbols seen in the file, in first-appearance order, deduped."""
    seen: dict[str, int] = {}  # symbol -> count
    order: list[str] = []

    def add(sym: str) -> None:
        if len(sym) < MIN_SYMBOL_LEN:
            return
        if sym in SYMBOL_BLOCKLIST:
            return
        if sym not in seen:
            order.append(sym)
        seen[sym] = seen.get(sym, 0) + 1

    # Bold API signatures (**funcName(args)**) — strongest signal in KDK docs.
    for m in BOLD_SIG.finditer(text):
        add(m.group(1))

    # Inline code signatures (`funcName(args)`).
    for m in INLINE_CODE_SIG.finditer(text):
        add(m.group(1))

    # Function/class/const declarations inside fenced code blocks.
    for block in FENCED_CODE.findall(text):
        for m in FN_DECL.finditer(block):
            add(m.group(1))

    # Rank by count desc; ties broken by first-appearance order (captured now
    # because order.index() would be wrong once the list starts mutating).
    first_idx = {s: i for i, s in enumerate(order)}
    order.sort(key=lambda s: (-seen[s], first_idx[s]))
    return order


def clean_markdown_text(text: str) -> str:
    """Remove markdown artifacts like backticks, bold, italic, and links."""
    # Remove links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove backticks
    text = text.replace("`", "")
    # Remove bold/italic markers
    text = text.replace("**", "").replace("*", "")
    return text.strip()

def extract_title(text: str, fallback: str) -> str:
    m = H1.search(text)
    if m:
        return clean_markdown_text(m.group(1))
    m = H2.search(text)
    if m:
        return clean_markdown_text(m.group(1))
    return fallback


def extract_sections(text: str, exclude_title: str) -> list[str]:
    """Return section headings (h2 + h3) other than the title, in order."""
    headings = [clean_markdown_text(h) for h in H2.findall(text)]
    headings += [clean_markdown_text(h) for h in H3.findall(text)]
    out = []
    seen = set()
    for h in headings:
        if h == exclude_title or h in seen:
            continue
        if len(h) < 4:
            continue
        seen.add(h)
        out.append(h)
    return out


def build_queries(md_files) -> list[dict]:
    """Mine queries from every markdown file. Returns list of query dicts."""
    # query_text_lower -> {query, source_type, sources: set, first_seen_title}
    bucket: dict[str, dict] = {}

    def put(query: str, source_type: str, source: str, file_title: str) -> None:
        key = query.lower().strip()
        if not key:
            return
        if key not in bucket:
            bucket[key] = {
                "query": query.strip(),
                "source_type": source_type,
                "sources": set(),
                "file_title": file_title,
            }
        bucket[key]["sources"].add(source)

    for rec in md_files:
        text = rec.path.read_text(encoding="utf-8")
        rel = rec.rel_path
        title = extract_title(text, Path(rel).stem)

        put(title, "title", rel, title)

        symbols = extract_symbols(text)[:MAX_SYMBOLS_PER_FILE]
        for sym in symbols:
            put(sym, "symbol", rel, title)

        sections = extract_sections(text, exclude_title=title)[:MAX_SECTIONS_PER_FILE]
        for sec in sections:
            put(sec, "section", rel, title)

    queries = []
    for i, (_, data) in enumerate(sorted(bucket.items())):
        queries.append({
            "id": f"gq-{i:04d}",
            "query": data["query"],
            "source_type": data["source_type"],
            "relevant_sources": sorted(data["sources"]),
            "file_title": data["file_title"],
        })
    return queries


def main() -> int:
    scan = scan_corpus(ROOT / "data" / "kdk")
    md_files = [
        rec for rec in scan.included
        if rec.extension == ".md" and rec.rel_path.startswith("docs/")
    ]
    print(f"Scanning {len(md_files)} markdown files under docs/")

    queries = build_queries(md_files)
    by_type = defaultdict(int)
    for q in queries:
        by_type[q["source_type"]] += 1

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "gold_queries.json"

    payload = {
        "version": "1.0",
        "description": (
            "Auto-generated query set for nb02 deterministic evaluation. "
            "Each query is mined from a markdown file and its gold source(s) "
            "point back to the file(s) it was extracted from."
        ),
        "source": "experiments/nb02_chunking_md/build_gold_query_set.py",
        "counts": dict(by_type),
        "total": len(queries),
        "queries": queries,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"{len(queries)} queries generated "
          f"(title={by_type['title']}, symbol={by_type['symbol']}, "
          f"section={by_type['section']})")
    print(f"Wrote {out_path.relative_to(ROOT)}")

    # Quick integrity check: every query should have >= 1 gold source.
    missing = sum(1 for q in queries if not q["relevant_sources"])
    if missing:
        print(f"WARNING: {missing} queries have empty relevant_sources")
        return 1
    print("All queries have at least one gold source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
