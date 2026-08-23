"""The exact shape /search returns, checked against the corpus it came from.

Issue #6 pins the agent-facing contract:

    [
      {
        "path": "src/store/layers.js",
        "lines": "45-78",
        "score": 0.92,
        "content": "...",
        "commit_history": ["fix: ...", "feat: ..."]
      },
      ...
    ]

So the response is a bare list, every field is checked for what it actually
holds, and `lines` is read back against the file in the workspace: a range
that does not open the file on the chunk's own content is a broken promise.
"""

import json

import pytest

from helpers import SAMPLES, requires_qdrant

CONTRACT_FIELDS = ["path", "lines", "score", "content", "commit_history"]


# The corpus ingested, with a second commit on one file so the history has
# something to order.
@pytest.fixture
def served(pipeline):
    pipeline.workspace.install_samples("feat: import the corpus")
    pipeline.workspace.commit(
        "kdk/core/client/geolocation.js",
        pipeline.workspace.path("kdk/core/client/geolocation.js").read_text()
        + "\n// tracked\n",
        "fix(geo): guard against a missing position")
    assert pipeline.run() == 0
    return pipeline


# One search response, as JSON.
@pytest.fixture
def results(served):
    response = served.client.post(
        "/search", json={"query": "layer catalog geolocation", "top_k": 10})
    assert response.status_code == 200
    return response.json()


@pytest.mark.Contract
class TestResponseShape:
    # The response is a list, not an object wrapping one.
    @requires_qdrant
    def test_the_response_is_a_bare_list(self, results):
        assert isinstance(results, list)
        assert results

    # Every contract field is present on every result.
    @requires_qdrant
    def test_every_result_carries_the_contract_fields(self, results):
        for result in results:
            assert CONTRACT_FIELDS == [field for field in CONTRACT_FIELDS
                                       if field in result]

    # The contract fields come first, in the documented order, so the
    # response reads like the issue.
    @requires_qdrant
    def test_the_contract_fields_come_first_and_in_order(self, results):
        assert list(results[0])[:len(CONTRACT_FIELDS)] == CONTRACT_FIELDS

    # Nothing internal leaks: the digests and the raw line numbers are
    # bookkeeping for the ingestion job.
    @requires_qdrant
    def test_no_bookkeeping_field_is_exposed(self, results):
        for result in results:
            assert "file_sha1" not in result
            assert "content_sha1" not in result
            assert "start_line" not in result
            assert "end_line" not in result


@pytest.mark.Contract
class TestFieldValues:
    # path is repo-relative; repo names the repository separately.
    @requires_qdrant
    def test_the_path_is_repository_relative(self, results):
        expected = {path.split("/", 1)[1] for path in SAMPLES}

        for result in results:
            assert result["path"] in expected
            assert result["repo"] in {"kdk", "kano"}
            assert not result["path"].startswith(result["repo"] + "/")

    # lines is "start-end", or a single number for a one-line chunk.
    @requires_qdrant
    def test_the_lines_field_is_a_range(self, results):
        for result in results:
            parts = result["lines"].split("-")
            assert 1 <= len(parts) <= 2
            assert all(part.isdigit() for part in parts), result["lines"]
            if len(parts) == 2:
                assert int(parts[0]) <= int(parts[1])

    # score is a similarity, and the list comes back best first.
    @requires_qdrant
    def test_the_results_are_ordered_by_score(self, results):
        scores = [result["score"] for result in results]

        assert all(0.0 <= score <= 1.0 for score in scores)
        assert scores == sorted(scores, reverse=True)

    # content is the chunk itself, never empty.
    @requires_qdrant
    def test_the_content_is_the_chunk_text(self, results):
        for result in results:
            assert result["content"].strip()

    # commit_history is a list of subjects, newest first.
    @requires_qdrant
    def test_the_commit_history_is_a_list_of_subjects_newest_first(
            self, results):
        edited = [result for result in results
                  if result["path"] == "core/client/geolocation.js"]
        assert edited

        for result in edited:
            assert result["commit_history"] == [
                "fix(geo): guard against a missing position",
                "feat: import the corpus"]

    # top_k is honoured.
    @requires_qdrant
    def test_top_k_caps_the_number_of_results(self, served):
        response = served.client.post(
            "/search", json={"query": "layer", "top_k": 2})

        assert len(response.json()) == 2


@pytest.mark.Contract
class TestLinesPointAtTheFile:
    # The whole point of `lines`: opening the file there shows the chunk.
    @requires_qdrant
    def test_every_range_opens_the_file_on_the_chunk(self, served, results):
        for result in results:
            excerpt = _excerpt(served, result)
            quoted = _quoted_source_lines(served, result)
            assert quoted, f"nothing to check in {result['path']}"
            for line in quoted:
                assert line in excerpt, (
                    f"{result['path']}:{result['lines']} does not hold "
                    f"{line!r}")

    # The range is tight: it does not point at the whole file.
    @requires_qdrant
    def test_a_range_does_not_swallow_the_whole_file(self, served, results):
        for result in results:
            source = _source(served, result)
            start, end = _bounds(result)
            if len(source.splitlines()) < 10:
                continue
            assert (end - start + 1) < len(source.splitlines())

    # A chunk that starts further down the file says so.
    @requires_qdrant
    def test_the_ranges_of_a_multi_chunk_file_differ(self, served, results):
        by_file = {}
        for result in results:
            by_file.setdefault(result["path"], set()).add(result["lines"])

        multi = [ranges for ranges in by_file.values() if len(ranges) > 1]
        assert multi, "the corpus should have a file cut into several chunks"

    # The response can be serialised back to the JSON of the issue.
    @requires_qdrant
    def test_a_result_reads_like_the_documented_example(self, results):
        example = {field: results[0][field] for field in CONTRACT_FIELDS}

        rendered = json.loads(json.dumps(example))
        assert isinstance(rendered["path"], str)
        assert isinstance(rendered["lines"], str)
        assert isinstance(rendered["score"], float)
        assert isinstance(rendered["content"], str)
        assert isinstance(rendered["commit_history"], list)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# The file in the workspace a result points at.
def _source(pipeline, result):
    return pipeline.workspace.path(
        f"{result['repo']}/{result['path']}").read_text(encoding="utf-8")


# The (start, end) the result claims.
def _bounds(result):
    parts = result["lines"].split("-")
    return int(parts[0]), int(parts[-1])


# The lines of the file the result's range covers.
def _excerpt(pipeline, result):
    start, end = _bounds(result)
    return "\n".join(_source(pipeline, result).splitlines()[start - 1:end])


# The content lines the result quotes verbatim from its file. A JSON unit is
# rebuilt from the parsed tree, so only what the file really holds is
# compared -- and structural lines ("{", "},") are not evidence of anything.
def _quoted_source_lines(pipeline, result):
    source = {line.strip()
              for line in _source(pipeline, result).splitlines()
              if line.strip(" \t{}[](),;")}
    return [line.strip() for line in result["content"].splitlines()[1:]
            if line.strip() in source]
