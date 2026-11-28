from __future__ import annotations

import json
from pathlib import Path

import pytest

from mangoleaf.config import CLASSES
from mangoleaf.manifest import (
    ManifestError,
    ManifestRow,
    class_index_document,
    load_manifest,
    manifest_sha256,
    read_json,
    render_summary_markdown,
    write_json,
    write_manifest,
)


def _rows() -> list[ManifestRow]:
    return [
        ManifestRow("Anthracnose/a.jpg", "Anthracnose", 0, "train", "a" * 64, "g1"),
        ManifestRow("Healthy/b.jpg", "Healthy", 5, "val", "b" * 64, "g2"),
        ManifestRow("Sooty Mould/c.jpg", "Sooty Mould", 7, "test", "c" * 64, "g3"),
    ]


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(_rows(), path)

    manifest = load_manifest(path)

    assert manifest.rows == tuple(_rows())
    assert manifest.fingerprint == manifest_sha256(path)
    assert len(manifest.fingerprint) == 64
    assert manifest.split_counts() == {"train": 1, "val": 1, "test": 1}
    assert manifest.class_counts()["Healthy"]["val"] == 1


def test_load_rejects_wrong_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("path,label\nx.jpg,Healthy\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="Unexpected CSV header"):
        load_manifest(path)


def test_load_rejects_class_idx_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    write_manifest([ManifestRow("a.jpg", "Healthy", 0, "train", "a" * 64, "g")], path)

    with pytest.raises(ManifestError, match="does not match"):
        load_manifest(path)


def test_load_rejects_unknown_class(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    write_manifest([ManifestRow("a.jpg", "Mango", 0, "train", "a" * 64, "g")], path)

    with pytest.raises(ManifestError, match="unknown class"):
        load_manifest(path)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.csv")


def test_class_index_document() -> None:
    document = class_index_document()

    assert document["dataset"] == "MangoLeafBD"
    assert document["version"] == "v1"
    assert [entry["class"] for entry in document["classes"]] == list(CLASSES)
    assert [entry["class_idx"] for entry in document["classes"]] == list(range(8))


def test_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    document = {"clase": "Añil", "n": 3}
    write_json(document, path)

    assert read_json(path) == document
    assert json.loads(path.read_text(encoding="utf-8")) == document


def test_render_summary_markdown_contains_tables_and_findings() -> None:
    meta = {
        "dataset": {
            "name": "MangoLeafBD",
            "version": "v1",
            "source_images": 4000,
            "source_images_per_class": 500,
        },
        "generation": {
            "seed": 42,
            "command": "python scripts/build_manifest.py --seed 42",
            "python": "3.13.5",
            "pillow": "11.1.0",
            "dhash_threshold": 6,
        },
        "selection": {
            "per_class": [
                {
                    "class_idx": 0,
                    "class": "Anthracnose",
                    "train": 88,
                    "val": 18,
                    "test": 19,
                    "total": 125,
                },
                {
                    "class_idx": 1,
                    "class": "Healthy",
                    "train": 87,
                    "val": 19,
                    "test": 19,
                    "total": 125,
                },
            ]
        },
        "totals": {"by_split": {"train": 700, "val": 150, "test": 150}, "rows": 1000},
        "duplicates": {
            "exact": {"removed": 21, "groups": 21, "cross_class_conflicts": 0},
            "near_duplicate": {
                "groups": 3000,
                "multi_member_groups": 400,
                "max_members": 4,
                "cross_class_groups": 0,
                "signals": {"asset_stem": 10, "dhash": 3},
            },
        },
        "validation": {"images_checked": 4000, "invalid": []},
        "fingerprint": "f" * 64,
        "residual_risk": "Riesgo documentado.",
    }

    markdown = render_summary_markdown(meta)

    assert markdown.startswith("# Manifiesto compartido — MangoLeafBD v1")
    assert "| 0 | Anthracnose | 88 | 18 | 19 | 125 |" in markdown
    assert "| **Total** | 700 | 150 | 150 | 1000 |" in markdown
    assert "**21**" in markdown
    assert "**4000**" in markdown
    assert "Riesgo documentado." in markdown
    assert "no disponibles" in markdown  # hallazgo sobre identificadores de hoja
