"""Run the real product-level experiment for nb02 Part 7.

For each (task, strategy) pair:
  1. Retrieve Top-K chunks from the strategy's Qdrant collection using the
     task's query.
  2. Build a Top-K "bundle" (concatenation) — the exact shape an agent would
     stuff into its context window.
  3. Ask the LLM to generate code for the task using only that bundle.
  4. Save a markdown file capturing task, query, retrieved sources, bundle,
     and generated code. These files are the input to manual/qualitative
     scoring in nb02 Part 7.

No automatic LLM-as-judge scoring is done here — the point of this experiment
is to produce artifacts a human (or a second LLM call) can review to answer
the real question: does this strategy's retrieval output actually let a
code-generation model write correct Kalisio code?

Usage:
    # Full run (needs LLM provider configured via LLM_PROVIDER + either
    # OLLAMA_BASE_URL/OLLAMA_MODEL or ANTHROPIC_API_KEY):
    python scripts/nb02_codegen_experiment.py

    # Retrieval only — builds bundles but skips LLM calls. Useful for
    # inspecting what the agent would see without burning tokens.
    python scripts/nb02_codegen_experiment.py --dry-run

    # Limit strategies or tasks:
    python scripts/nb02_codegen_experiment.py --strategy D_ast_breadcrumb
    python scripts/nb02_codegen_experiment.py --task task-01-timestamp-hook
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from embedding_utils import load_embedding_model  # noqa: E402

load_dotenv(ROOT / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
TOP_K = 5

STRATEGY_COLLECTIONS = {
    "A_rct": "nb02_A_rct",
    "B_mhs_rct": "nb02_B_mhs_rct",
    "C_ast_merge": "nb02_C_ast_merge",
    "D_ast_breadcrumb": "nb02_D_ast_breadcrumb",
}

SYSTEM_PROMPT = (
    "You are a senior engineer writing code for the Kalisio KDK platform. "
    "Use ONLY the provided documentation context; do not invent imports, "
    "functions, or APIs that are not mentioned in the context. If the context "
    "does not contain enough information to complete the task, write the code "
    "you can justify from the context and add a short comment pointing out "
    "which part you had to guess. Return the code in a single fenced block, "
    "followed by a one-sentence note on what you relied on."
)


def build_bundle(points) -> tuple[str, list[dict]]:
    """Concatenate Top-K payloads into a bundle. Returns (bundle_text, sources)."""
    parts: list[str] = []
    sources: list[dict] = []
    for i, p in enumerate(points, start=1):
        md = p.payload.get("metadata", {})
        src = md.get("source", "?")
        text = p.payload.get("text", "")
        sources.append({
            "rank": i,
            "score": round(float(p.score), 4),
            "source": src,
            "breadcrumb": md.get("breadcrumb", ""),
        })
        parts.append(
            f"[Chunk {i}] source: {src}\n{text}"
        )
    return "\n\n---\n\n".join(parts), sources


def build_prompt(task: dict, bundle: str) -> str:
    return (
        f"Task: {task['title']}\n\n"
        f"Description:\n{task['description']}\n\n"
        f"Documentation context:\n"
        f"<<<\n{bundle}\n>>>\n\n"
        f"Generate the code now."
    )


def retrieve(qdrant, embed_model, collection: str, query: str, k: int = TOP_K):
    vec = embed_model.encode("search_query: " + query).tolist()
    return qdrant.query_points(
        collection_name=collection, query=vec, limit=k
    ).points


def write_result(out_dir: Path, strategy: str, task: dict,
                 sources: list[dict], bundle: str, code: str | None) -> Path:
    md = [
        f"# {task['id']} — {strategy}",
        "",
        f"**Task:** {task['title']}",
        "",
        f"**Description:** {task['description']}",
        "",
        f"**Query:** `{task['query']}`",
        "",
        "## Retrieved Top-K",
        "",
        "| Rank | Score | Source | Breadcrumb |",
        "|-----:|------:|--------|------------|",
    ]
    for s in sources:
        breadcrumb = s["breadcrumb"].replace("|", "\\|") if s["breadcrumb"] else ""
        md.append(
            f"| {s['rank']} | {s['score']:.4f} | `{s['source']}` | {breadcrumb} |"
        )
    md += [
        "",
        "## Bundle (what the LLM saw as context)",
        "",
        "```markdown",
        bundle,
        "```",
        "",
        "## LLM output",
        "",
    ]
    if code is None:
        md.append("_(dry-run: LLM call skipped)_")
    else:
        md.append(code)
    md += [
        "",
        "## Evaluation criteria",
        "",
    ]
    for c in task["evaluation_criteria"]:
        md.append(f"- [ ] {c}")
    md.append("")

    out_path = out_dir / f"{strategy}__{task['id']}.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, write retrieval bundles only")
    parser.add_argument("--strategy", choices=list(STRATEGY_COLLECTIONS))
    parser.add_argument("--task")
    parser.add_argument(
        "--tasks-file",
        default=str(ROOT / "outputs" / "codegen_tasks.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "nb02_codegen_results"),
    )
    args = parser.parse_args()

    tasks_payload = json.loads(Path(args.tasks_file).read_text())
    tasks = tasks_payload["tasks"]
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"No task matches id={args.task}")
            return 1

    strategies = (
        {args.strategy: STRATEGY_COLLECTIONS[args.strategy]}
        if args.strategy else STRATEGY_COLLECTIONS
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qdrant = QdrantClient(url=QDRANT_URL)
    embed_model = load_embedding_model(EMBEDDING_MODEL)

    llm = None
    if not args.dry_run:
        from token_tracker import TrackedClient  # noqa: E402
        llm = TrackedClient(source="nb02_codegen_experiment")
        print(f"LLM: provider={llm.provider} model={llm.model}")

    total = len(tasks) * len(strategies)
    print(f"Running {total} (task × strategy) pairs, "
          f"{'DRY-RUN' if args.dry_run else 'LIVE'}")

    for strat, coll in strategies.items():
        for task in tasks:
            points = retrieve(qdrant, embed_model, coll, task["query"])
            bundle, sources = build_bundle(points)

            code: str | None = None
            if llm is not None:
                prompt = build_prompt(task, bundle)
                try:
                    code = llm.ask(prompt, system=SYSTEM_PROMPT, max_tokens=1500)
                except Exception as exc:
                    code = f"_(LLM call failed: {exc})_"

            out_path = write_result(out_dir, strat, task, sources, bundle, code)
            mark = "DRY" if code is None else "OK"
            print(f"  [{mark}] {out_path.relative_to(ROOT)}")

    if llm is not None:
        llm.summary()
        llm.save()

    print(f"\nWrote {total} result files to {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
