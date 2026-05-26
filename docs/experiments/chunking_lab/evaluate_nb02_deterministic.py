"""Deterministic retrieval evaluation for nb02 chunking strategies.

For each strategy collection in Qdrant, retrieve Top-K chunks for every query
in `outputs/gold_queries.json` and compute source-level metrics:

  - recall@k        : fraction of gold sources present in Top-K
  - hit@k           : 1 if any gold source is in Top-K, else 0
  - code_integrity  : fraction of retrieved chunks with balanced ``` fences
  - bundle_efficiency : chars-on-topic / total chars in the Top-K bundle

No LLM judge, no human scoring. Writes `outputs/nb02_deterministic_metrics.json`.

Usage:
    python scripts/evaluate_nb02_deterministic.py              # all 4 strategies
    python scripts/evaluate_nb02_deterministic.py --collection nb02_A_rct  # one
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from tqdm import tqdm

_HELPER = Path(__file__).resolve().parents[1] / "experiment_helper"
sys.path.insert(0, str(_HELPER))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_utils import load_embedding_model  # noqa: E402
from retrieval_metrics import (  # noqa: E402
    bundle_token_efficiency,
    code_integrity_rate,
    hit_at_k,
    recall_at_k,
)

load_dotenv(ROOT / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
TOP_K = 5

DEFAULT_COLLECTIONS = [
    "nb02_A_rct",
    "nb02_B_mhs_rct",
    "nb02_C_ast_merge",
    "nb02_D_ast_breadcrumb",
]


def load_queries(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["queries"]


def retrieve(qdrant: QdrantClient, embed_model, collection: str,
             query_text: str, k: int = TOP_K) -> list[dict]:
    """Run one Top-K retrieval, return list of dicts with source + text."""
    vec = embed_model.encode("search_query: " + query_text).tolist()
    points = qdrant.query_points(
        collection_name=collection, query=vec, limit=k
    ).points
    out = []
    for rank, p in enumerate(points, start=1):
        md = p.payload.get("metadata", {})
        out.append({
            "rank": rank,
            "score": round(float(p.score), 4),
            "source": md.get("source", "?"),
            "text": p.payload.get("text", ""),
        })
    return out


def evaluate_collection(qdrant: QdrantClient, embed_model,
                        collection: str, queries: list[dict]) -> dict:
    """Run every query through one collection and aggregate metrics."""
    per_query = []
    for q in tqdm(queries, desc=collection, unit="q"):
        retrieved = retrieve(qdrant, embed_model, collection, q["query"])
        sources = [r["source"] for r in retrieved]
        texts = [r["text"] for r in retrieved]
        bundle = "\n\n".join(texts)

        # Use the query text itself as a "symbol" for bundle efficiency when
        # the source_type is 'symbol'; otherwise use the gold file titles as
        # a looser signal. For symbol queries this gives a meaningful number;
        # for title/section queries it still tells us roughly how much of the
        # bundle mentions the topic.
        target_terms = [q["query"]] if q["source_type"] == "symbol" else [
            q["query"].split()[0]
        ]

        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "source_type": q["source_type"],
            "gold": q["relevant_sources"],
            "retrieved_sources": sources,
            "recall_at_5": recall_at_k(sources, q["relevant_sources"], TOP_K),
            "hit_at_5": hit_at_k(sources, q["relevant_sources"], TOP_K),
            "code_integrity": code_integrity_rate(texts),
            "bundle_efficiency": bundle_token_efficiency(bundle, target_terms),
        })

    def mean(field: str) -> float:
        vals = [row[field] for row in per_query]
        return sum(vals) / len(vals) if vals else 0.0

    # Per-source-type breakdown (lets us see which strategy wins on symbols
    # versus sections versus titles).
    by_type: dict[str, dict] = {}
    for t in ("title", "symbol", "section"):
        subset = [r for r in per_query if r["source_type"] == t]
        if not subset:
            continue
        by_type[t] = {
            "n": len(subset),
            "recall_at_5": sum(r["recall_at_5"] for r in subset) / len(subset),
            "hit_at_5": sum(r["hit_at_5"] for r in subset) / len(subset),
            "code_integrity": sum(r["code_integrity"] for r in subset) / len(subset),
            "bundle_efficiency": sum(r["bundle_efficiency"] for r in subset) / len(subset),
        }

    return {
        "collection": collection,
        "n_queries": len(per_query),
        "overall": {
            "recall_at_5": mean("recall_at_5"),
            "hit_at_5": mean("hit_at_5"),
            "code_integrity": mean("code_integrity"),
            "bundle_efficiency": mean("bundle_efficiency"),
        },
        "by_source_type": by_type,
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", help="Only evaluate one collection")
    parser.add_argument(
        "--queries",
        default=str(ROOT / "outputs" / "gold_queries.json"),
        help="Path to gold query set JSON",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "nb02_deterministic_metrics.json"),
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run first N queries (smoke test)"
    )
    args = parser.parse_args()

    queries = load_queries(Path(args.queries))
    if args.limit:
        queries = queries[: args.limit]
    print(f"Loaded {len(queries)} queries from {args.queries}")

    collections = [args.collection] if args.collection else DEFAULT_COLLECTIONS

    qdrant = QdrantClient(url=QDRANT_URL)
    embed_model = load_embedding_model(EMBEDDING_MODEL)

    results = {}
    for coll in collections:
        res = evaluate_collection(qdrant, embed_model, coll, queries)
        results[coll] = res
        overall = res["overall"]
        print(
            f"  {coll}: "
            f"recall@5={overall['recall_at_5']:.3f} "
            f"hit@5={overall['hit_at_5']:.3f} "
            f"integrity={overall['code_integrity']:.3f} "
            f"eff={overall['bundle_efficiency']:.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
