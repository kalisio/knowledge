"""Tests for kli wrapper. Stubs node + kli with a fake shell script."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ingestion_job.kli_clone import clone_with_kli


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture()
def fake_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a fake `node` on PATH that records its argv to /tmp."""
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    record = tmp_path / "node_argv.log"
    _make_executable(
        fake_bin / "node",
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {record}\n",
    )
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    return record


def test_invokes_node_with_expected_args(tmp_path: Path, fake_node: Path) -> None:
    root = tmp_path / "root"
    kli_dir = root / "kli"
    kli_dir.mkdir(parents=True)
    (kli_dir / "index.js").write_text("// fake")
    config = root / "ws.js"
    config.write_text("module.exports = {}")

    clone_with_kli(config, root)

    argv = fake_node.read_text().splitlines()
    assert argv[0] == str(kli_dir / "index.js")
    assert argv[1] == str(config)
    assert argv[2] == "--no-fail-on-error"
    assert argv[3] == "--clone"


def test_raises_when_kli_missing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    config = root / "ws.js"
    config.write_text("{}")
    with pytest.raises(RuntimeError, match="kli not found"):
        clone_with_kli(config, root)


def test_raises_when_config_missing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    kli_dir = root / "kli"
    kli_dir.mkdir(parents=True)
    (kli_dir / "index.js").write_text("// fake")
    with pytest.raises(RuntimeError, match="KLI_CONFIG file does not exist"):
        clone_with_kli(root / "missing.js", root)


def test_respects_custom_kli_dir(tmp_path: Path, fake_node: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    custom = tmp_path / "elsewhere" / "kli"
    custom.mkdir(parents=True)
    (custom / "index.js").write_text("// fake")
    config = root / "ws.js"
    config.write_text("{}")

    clone_with_kli(config, root, kli_dir=custom)

    argv = fake_node.read_text().splitlines()
    assert argv[0] == str(custom / "index.js")
