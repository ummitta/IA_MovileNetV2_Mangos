from __future__ import annotations

from pathlib import Path

from PIL import Image

from mangoleaf.config import CLASSES
from mangoleaf.dataset import scan_dataset
from mangoleaf.dedup import (
    choose_representative,
    deduplicate_exact,
    dhash64,
    dhash_orientations,
    find_duplicate_groups,
    min_rotation_hamming,
)


def test_dhash_is_stable_and_rotation_aware(write_image, tmp_path: Path) -> None:
    path = write_image(tmp_path / "leaf.jpg", seed=7, size=48)
    with Image.open(path) as image:
        base = dhash64(image.convert("L"))
        orientations = dhash_orientations(image)

    assert len(orientations) == 4
    assert base == orientations[0]

    with Image.open(path) as image:
        for angle in (90, 180, 270):
            rotated = image.rotate(angle, expand=True)
            assert min_rotation_hamming(orientations, dhash_orientations(rotated)) == 0


def test_min_rotation_hamming_detects_unrelated_images(write_image, tmp_path: Path) -> None:
    first = write_image(tmp_path / "a.jpg", seed=1)
    second = write_image(tmp_path / "b.jpg", seed=2)
    with Image.open(first) as left, Image.open(second) as right:
        distance = min_rotation_hamming(dhash_orientations(left), dhash_orientations(right))
    assert distance > 6


def test_exact_duplicates_are_removed(make_dataset) -> None:
    root = make_dataset(2)
    original = root / CLASSES[0] / "img_00_0000.jpg"
    duplicate = root / CLASSES[0] / "copy of original.jpg"
    duplicate.write_bytes(original.read_bytes())

    result = scan_dataset(root)
    dedup = deduplicate_exact(result.records)

    assert len(result.records) == 17
    assert len(dedup.records) == 16
    assert len(dedup.removed) == 1
    kept, dropped = dedup.removed[0]
    assert kept.relative_path == original.relative_to(root).as_posix()
    assert dropped.relative_path == duplicate.relative_to(root).as_posix()
    assert not dedup.cross_class_conflicts


def test_find_duplicate_groups_unites_rotated_variant(make_dataset) -> None:
    root = make_dataset(1)
    with Image.open(root / CLASSES[0] / "img_00_0000.jpg") as image:
        image.rotate(90, expand=True).save(root / CLASSES[0] / "different name.jpg")

    records = scan_dataset(root).records
    groups = find_duplicate_groups(list(records), dhash_threshold=6)

    assert len(groups) == 8  # 9 images; two are a rotated pair
    multi = [group for group in groups if len(group.members) > 1]
    assert len(multi) == 1
    assert set(multi[0].members) == {
        f"{CLASSES[0]}/img_00_0000.jpg",
        f"{CLASSES[0]}/different name.jpg",
    }
    assert "dhash" in multi[0].signals


def test_find_duplicate_groups_unites_re_saved_suffixes(write_image, tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for class_name in CLASSES:
        (root / class_name).mkdir(parents=True)
    write_image(root / CLASSES[1] / "20211011_140211.jpg", seed=11)
    write_image(root / CLASSES[1] / "IMG_20211011_140211 (Custom) - Copy.jpg", seed=12)

    records = scan_dataset(root).records
    groups = find_duplicate_groups(list(records), dhash_threshold=-1)

    assert len(groups) == 1
    assert len(groups[0].members) == 2
    assert "asset_stem" in groups[0].signals
    assert "capture_timestamp" in groups[0].signals


def test_choose_representative_prefers_clean_name(write_image, tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for class_name in CLASSES:
        (root / class_name).mkdir(parents=True)
    write_image(root / CLASSES[0] / "20211011_140211.jpg", seed=1)
    write_image(root / CLASSES[0] / "IMG_20211011_140211 (Custom) - Copy.jpg", seed=2)

    by_name = {record.filename: record for record in scan_dataset(root).records}
    chosen = choose_representative(
        [
            by_name["IMG_20211011_140211 (Custom) - Copy.jpg"],
            by_name["20211011_140211.jpg"],
        ]
    )
    assert chosen.filename == "20211011_140211.jpg"
