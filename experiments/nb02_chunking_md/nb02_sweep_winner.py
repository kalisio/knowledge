"""Parameter sweep on the nb02 winner strategy.

Re-chunks `data/kdk/docs/*.md` at 3 chunk sizes using the winner strategy
exported in `src/chunking.py`, ingests each into a dedicated Qdrant
collection, evaluates against `outputs/gold_queries.json`, and produces:

  - outputs/nb02_sweep_metrics.json
  - outputs/nb02_sweep_deterministic.png

Run after `scripts/evaluate_nb02_deterministic.py` has picked the winner.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chunking import WINNER_STRATEGY, chunk_files  # noqa: E402
from corpus_filter import scan_corpus  # noqa: E402
from embedding_utils import embed_batch_size, load_embedding_model  # noqa: E402
from evaluate_nb02_deterministic import evaluate_collection  # noqa: E402

load_dotenv(ROOT / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
SWEEP_SIZES = [500, 1000, 1500]


def encode_microbatch_size(texts: list[str]) -> int:
    """Cap encode batch by longest text — strategy D prefixes inflate sequences."""
    n = len(texts)
    m = max((len(t) for t in texts), default=1)
    if not torch.cuda.is_available():
        return min(32, n)
    if m > 20_000: return 1
    if m > 12_000: return min(2, n)
    if m > 8_000: return min(4, n)
    if m > 5_000: return min(6, n)
    if m > 3_500: return min(8, n)
    if m > 2_000: return min(12, n)
    if m > 1_200: return min(24, n)
    return min(48, n)


def ingest(qdrant: QdrantClient, embed_model, coll_name: str,
           chunks: list[dict], upsert_batch: int = 128) -> None:
    """Recreate a collection and ingest all chunks."""
    if qdrant.collection_exists(coll_name):
        qdrant.delete_collection(coll_name)
    qdrant.create_collection(
        coll_name, VectorParams(size=768, distance=Distance.COSINE)
    )

    for i in tqdm(range(0, len(chunks), upsert_batch), desc=coll_name, unit="batch"):
        batch = chunks[i : i + upsert_batch]
        texts = ["search_document: " + c["text"] for c in batch]
        micro = encode_microbatch_size(texts)
        with torch.inference_mode():
            vectors = embed_model.encode(
                texts, batch_size=micro, show_progress_bar=False
            ).tolist()
        points = [
            PointStruct(id=i + j, vector=vectors[j], payload=batch[j])
            for j in range(len(batch))
        ]
        qdrant.upsert(collection_name=coll_name, points=points)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    scan = scan_corpus(ROOT / "data" / "kdk")
    md_files = [
        r for r in scan.included
        if r.extension == ".md" and r.rel_path.startswith("docs/")
    ]
    print(f"Corpus: {len(md_files)} markdown files")
    print(f"Winner strategy: {WINNER_STRATEGY}")

    qdrant = QdrantClient(url=QDRANT_URL)
    embed_model = load_embedding_model(EMBEDDING_MODEL)

    # Drop any sweep collections left over from earlier runs (including those
    # built with a different winner strategy).
    for coll in qdrant.get_collections().collections:
        if coll.name.startswith("nb02_sweep_"):
            qdrant.delete_collection(coll.name)
            print(f"  dropped stale {coll.name}")

    with open(ROOT / "outputs" / "gold_queries.json", "r", encoding="utf-8") as f:
        gold_queries = json.load(f)["queries"]
    print(f"Gold queries: {len(gold_queries)}")

    sweep_metrics: dict[int, dict] = {}
    upsert_batch = embed_batch_size()

    for size in SWEEP_SIZES:
        coll_name = f"nb02_sweep_{WINNER_STRATEGY}_sz{size}"
        print(f"\n=== {coll_name} ===")
        chunks = chunk_files(md_files, strategy=WINNER_STRATEGY, chunk_size=size)
        print(f"  chunks: {len(chunks)}")
        ingest(qdrant, embed_model, coll_name, chunks, upsert_batch)
        res = evaluate_collection(qdrant, embed_model, coll_name, gold_queries)
        sweep_metrics[size] = res["overall"]
        o = res["overall"]
        print(f"  recall@5={o['recall_at_5']:.3f} "
              f"integrity={o['code_integrity']:.3f} "
              f"eff={o['bundle_efficiency']:.3f}")

    # Persist
    out_json = ROOT / "outputs" / "nb02_sweep_metrics.json"
    out_json.write_text(
        json.dumps(
            {"strategy": WINNER_STRATEGY, "sizes": SWEEP_SIZES, "metrics": sweep_metrics},
            indent=2,
        )
    )
    print(f"\nWrote {out_json}")

    # Plot
    sweep_df = pd.DataFrame([
        {"chunk_size": size, **m} for size, m in sweep_metrics.items()
    ]).sort_values("chunk_size")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plots = [
        ("recall_at_5", "Recall@5"),
        ("code_integrity", "Code Integrity"),
        ("bundle_efficiency", "Bundle Efficiency"),
    ]
    for ax, (metric, title) in zip(axes, plots):
        ax.plot(sweep_df["chunk_size"], sweep_df[metric],
                marker="o", linewidth=2, color="#2a6f97")
        ax.set_xlabel("chunk_size")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        for _, row in sweep_df.iterrows():
            ax.annotate(
                f"{row[metric]:.3f}",
                (row["chunk_size"], row[metric]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

    fig.suptitle(
        f"Parameter Sweep — {WINNER_STRATEGY} (439 gold queries)", fontsize=13
    )
    plt.tight_layout()
    out_png = ROOT / "outputs" / "nb02_sweep_deterministic.png"
    plt.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")

    # Pick best chunk_size by bundle_efficiency, tie-broken by integrity.
    best = max(
        sweep_metrics.items(),
        key=lambda kv: (kv[1]["bundle_efficiency"], kv[1]["code_integrity"]),
    )
    print(f"\nBest chunk_size for {WINNER_STRATEGY}: {best[0]}")
    print(f"  {best[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
