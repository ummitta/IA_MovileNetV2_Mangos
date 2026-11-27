from __future__ import annotations

from pathlib import Path

import pytest

from mangoleaf.config import CLASSES
from mangoleaf.dataset import (
    DatasetError,
    extract_capture_timestamp,
    normalize_asset_stem,
    scan_dataset,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("20211011_140211.jpg", "20211011_140211"),
        ("IMG_20211011_140211.jpg", "20211011_140211"),
        ("IMG_20211011_140211 (Custom) (Custom).jpg", "20211011_140211"),
        ("20211011_140211 - Copy - Copy.jpg", "20211011_140211"),
        ("20211011_140211 (2).jpg", "20211011_140211"),
        ("weird name.jpg", "weird name"),
    ],
)
def test_normalize_asset_stem(filename: str, expected: str) -> None:
    assert normalize_asset_stem(filename) == expected


def test_extract_capture_timestamp() -> None:
    assert extract_capture_timestamp("IMG_20211011_140211 (Custom).jpg") == "20211011_140211"
    assert extract_capture_timestamp("20211011_140211.jpg") == "20211011_140211"
    assert extract_capture_timestamp("no_timestamp.jpg") is None


def test_scan_dataset_records_metadata(make_dataset) -> None:
    root = make_dataset(2)
    result = scan_dataset(root)

    assert len(result.records) == 16
    assert not result.invalid
    assert not result.ignored
    assert [record.relative_path for record in result.records] == sorted(
        record.relative_path for record in result.records
    )
    first = result.records[0]
    assert first.class_name in CLASSES
    assert first.class_idx == CLASSES.index(first.class_name)
    assert first.width > 0 and first.height > 0
    assert len(first.sha256) == 64
    assert first.relative_path.startswith(first.class_name + "/")


def test_scan_dataset_reports_invalid_and_ignored(make_dataset) -> None:
    root = make_dataset(1)
    (root / CLASSES[0] / "broken.jpg").write_bytes(b"not an image")
    (root / CLASSES[0] / "notes.txt").write_text("ignored")

    result = scan_dataset(root)

    assert len(result.records) == 8
    assert [path for path, _ in result.invalid] == [f"{CLASSES[0]}/broken.jpg"]
    assert result.invalid[0][1].startswith(("UnidentifiedImageError", "OSError"))
    assert f"{CLASSES[0]}/notes.txt" in result.ignored


def test_scan_dataset_requires_every_class_folder(make_dataset) -> None:
    root = make_dataset(1)
    for entry in (root / CLASSES[3]).iterdir():
        entry.unlink()
    (root / CLASSES[3]).rmdir()

    with pytest.raises(DatasetError, match="missing 1 class folder"):
        scan_dataset(root)


def test_scan_dataset_rejects_unexpected_folder(make_dataset) -> None:
    root = make_dataset(1)
    (root / "Unknown").mkdir()

    with pytest.raises(DatasetError, match="Unexpected folder"):
        scan_dataset(root)


def test_scan_dataset_without_pillow_validation(make_dataset) -> None:
    root = make_dataset(1)
    (root / CLASSES[0] / "broken.jpg").write_bytes(b"not an image")

    result = scan_dataset(root, validate_images=False)

    assert len(result.records) == 9
    assert not result.invalid
    assert all(record.width == 0 for record in result.records)


def test_scan_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="does not exist"):
        scan_dataset(tmp_path / "nope")
