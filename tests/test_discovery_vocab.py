"""Tests for YOLOE discovery vocabulary file parsing."""
from pathlib import Path

import pytest

from perception.models.instance import discovery_vocab as dv
from perception.models.instance.discovery_vocab import load_discovery_prompts


def test_load_skips_comments_and_blank_and_dedupes(tmp_path: Path):
    f = tmp_path / "v.txt"
    f.write_text(
        "\n  person  \n# ignored\nPedestrian\ncar\ncar\nperson\n\n",
        encoding="utf-8",
    )
    out = load_discovery_prompts(f)
    assert out == ["person", "Pedestrian", "car"]


def test_empty_file_raises(tmp_path: Path):
    f = tmp_path / "e.txt"
    f.write_text("# only comment\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_discovery_prompts(f)


def test_over_max_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(dv, "_DISCOVERY_MAX_LINES", 4)
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line{k}" for k in range(6)), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        load_discovery_prompts(f)
