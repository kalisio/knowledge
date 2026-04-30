from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "nb06_qdrant_index"))

from nb06_helpers import (  # noqa: E402
    breadcrumb_text,
    chunk_type,
    evaluate_file_rankings,
    file_rank_from_points,
    normalize_chunk,
    source_file_type,
    source_repository,
    stable_point_id,
)


def test_stable_point_id_is_deterministic() -> None:
    first = stable_point_id("kdk/docs/api/map.md", 3, "same text")
    second = stable_point_id("kdk/docs/api/map.md", 3, "same text")
    changed = stable_point_id("kdk/docs/api/map.md", 3, "different text")

    assert first == second
    assert first != changed


def test_source_metadata_helpers() -> None:
    assert source_repository("kdk/docs/api/map.md") == "kdk"
    assert source_file_type("kdk/docs/api/map.md") == "md"
    assert source_file_type("kdk/core/client/map") == ""


def test_normalize_chunk_payload_shape() -> None:
    chunk = {
        "text": "Context: Map\n\naddLayer(...)",
        "metadata": {
            "source": "kdk/docs/api/map/map-mixins.md",
            "strategy": "D_ast_breadcrumb",
            "chunk_index": 2,
            "breadcrumb": "Map > Mixins",
        },
    }

    record = normalize_chunk(chunk, 99)

    assert record["id"]
    assert record["vector"] is None
    assert record["payload"]["repository"] == "kdk"
    assert record["payload"]["file_type"] == "md"
    assert record["payload"]["chunk_type"] == "markdown"
    assert record["payload"]["chunk_index"] == 2
    assert record["payload"]["source_path"] == "kdk/docs/api/map/map-mixins.md"
    assert record["payload"]["text_sha1"]


def test_breadcrumb_and_chunk_type_from_dict_metadata() -> None:
    metadata = {
        "breadcrumb": {"path": "kdk/map/client/foo.js", "symbol": "addLayer", "block": ""},
        "strategy": "D_js_breadcrumb",
    }

    assert breadcrumb_text(metadata) == "kdk/map/client/foo.js > addLayer"
    assert chunk_type(metadata, "kdk/map/client/foo.js") == "javascript"


class _Point:
    def __init__(self, source_path: str):
        self.payload = {"source_path": source_path}


class _Query:
    id = "Q-001"
    layer = "B_docs"
    is_negative = False
    gold_sources = ("kdk/docs/api/map.md",)


class _NegativeQuery:
    id = "N-001"
    layer = "negative"
    is_negative = True
    gold_sources = ()


def test_file_rank_from_points_deduplicates_sources() -> None:
    ranking = file_rank_from_points([
        _Point("kdk/docs/api/map.md"),
        _Point("kdk/docs/api/map.md"),
        _Point("kdk/map/client/map.js"),
    ])

    assert ranking == ["kdk/docs/api/map.md", "kdk/map/client/map.js"]


def test_evaluate_file_rankings_positive_and_negative() -> None:
    rows = evaluate_file_rankings(
        [
            ["kdk/docs/api/map.md", "kdk/map/client/map.js"],
            ["outside.md"],
        ],
        [_Query(), _NegativeQuery()],
        language="en",
        approach="qdrant_qwen3",
    )

    assert rows[0]["hit@5"] == 1
    assert rows[0]["mrr"] == 1.0
    assert rows[1]["is_negative"] is True
    assert rows[1]["hit@5"] == 1
