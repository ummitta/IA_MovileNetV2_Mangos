from __future__ import annotations

import random

import pytest

import mangoleaf.build as build_module
from mangoleaf.build import BuildError, _subset_for_target, assign_splits, build_manifest
from mangoleaf.config import CLASSES, SPLIT_ALLOCATION, SplitAllocation


def test_subset_for_target_exact_and_impossible() -> None:
    assert _subset_for_target([1, 1, 1], 2) is not None
    chosen = _subset_for_target([2, 1, 1], 2)
    assert chosen is not None and sum([2, 1, 1][index] for index in chosen) == 2
    assert _subset_for_target([3, 3], 4) is None
    assert _subset_for_target([1, 2], 0) == []


def test_assign_splits_with_singletons() -> None:
    units = [[f"img-{index}"] for index in range(125)]
    assignment = assign_splits(units, SPLIT_ALLOCATION["Anthracnose"], random.Random(0))

    counts = {"train": 0, "val": 0, "test": 0}
    for split, unit in assignment:
        counts[split] += len(unit)
    assert counts == {"train": 88, "val": 18, "test": 19}
    assert len(assignment) == 125


def test_assign_splits_keeps_groups_indivisible() -> None:
    units = [["a", "b"]] + [[f"img-{index}"] for index in range(123)]
    assignment = assign_splits(units, SPLIT_ALLOCATION["Cutting Weevil"], random.Random(7))

    counts = {"train": 0, "val": 0, "test": 0}
    split_of: dict[str, set[str]] = {}
    for split, unit in assignment:
        counts[split] += len(unit)
        for item in unit:
            split_of.setdefault(item, set()).add(split)
    assert counts == {"train": 88, "val": 19, "test": 18}
    assert split_of["a"] == split_of["b"]
    assert len(split_of["a"]) == 1


def test_assign_splits_fails_when_target_is_unreachable() -> None:
    units = [["x"] * 100, ["y"] * 25]
    with pytest.raises(BuildError, match="Cannot reach the train target"):
        assign_splits(units, SPLIT_ALLOCATION["Gall Midge"], random.Random(0))


def test_assign_splits_fails_on_wrong_total() -> None:
    with pytest.raises(BuildError, match="does not match"):
        assign_splits([["x"] * 124], SPLIT_ALLOCATION["Gall Midge"], random.Random(0))


def _patch_small_contract(monkeypatch) -> None:
    monkeypatch.setattr(build_module, "IMAGES_PER_CLASS", 6)
    monkeypatch.setattr(
        build_module,
        "SPLIT_ALLOCATION",
        {name: SplitAllocation(3, 2, 1) for name in CLASSES},
    )
    monkeypatch.setattr(build_module, "validate_config", lambda: None)


def test_build_manifest_small_contract_is_deterministic(make_dataset, monkeypatch) -> None:
    _patch_small_contract(monkeypatch)
    root = make_dataset(12)

    first = build_manifest(root, seed=42)
    second = build_manifest(root, seed=42)

    assert len(first.rows) == 48
    assert first.rows == second.rows
    assert first.totals == {"train": 24, "val": 16, "test": 8, "total": 48}
    assert {row.class_name for row in first.rows} == set(CLASSES)
    assert len({row.relative_path for row in first.rows}) == 48
    assert len({row.sha256 for row in first.rows}) == 48


def test_build_manifest_seed_changes_selection(make_dataset, monkeypatch) -> None:
    _patch_small_contract(monkeypatch)
    root = make_dataset(12)

    first = build_manifest(root, seed=1)
    second = build_manifest(root, seed=2)

    assert first.rows != second.rows


def test_build_manifest_realistic_size(make_dataset) -> None:
    # 130 images per class leaves slack above the 125 required groups.
    root = make_dataset(130)
    result = build_manifest(root, seed=42, dhash_threshold=6)

    assert len(result.rows) == 1000
    assert result.totals == {"train": 700, "val": 150, "test": 150, "total": 1000}
    for class_name in CLASSES:
        allocation = SPLIT_ALLOCATION[class_name]
        counts = result.counts_by_class[class_name]
        assert counts["train"] == allocation.train
        assert counts["val"] == allocation.val
        assert counts["test"] == allocation.test
        assert counts["total"] == 125

    groups_by_split: dict[str, set[str]] = {}
    for row in result.rows:
        groups_by_split.setdefault(row.near_dup_group, set()).add(row.split)
    assert all(len(splits) == 1 for splits in groups_by_split.values())
    assert len({row.relative_path for row in result.rows}) == 1000
    assert len({row.sha256 for row in result.rows}) == 1000
