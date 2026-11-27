"""Deterministic stratified selection and split assignment.

Pipeline (same seed + same dataset => byte-identical manifest):

1. Scan every class folder, decode each image with Pillow and hash it.
2. Remove exact SHA-256 duplicates.
3. Group near-duplicates (name stem, capture timestamp, dHash).
4. Per class, draw groups with a seeded RNG and keep one representative each.
5. Split the selected units per class with a second seeded RNG, using the
   explicit 700/150/150 allocation from :mod:`mangoleaf.config`.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mangoleaf.config import (
    CLASSES,
    DEFAULT_DHASH_THRESHOLD,
    IMAGES_PER_CLASS,
    SPLIT_ALLOCATION,
    SPLITS,
    SplitAllocation,
    validate_config,
)
from mangoleaf.dataset import ImageRecord, ScanResult, scan_dataset
from mangoleaf.dedup import (
    DuplicateGroup,
    ExactDedupResult,
    deduplicate_exact,
    find_duplicate_groups,
    ordered_members,
)
from mangoleaf.manifest import ManifestRow


class BuildError(RuntimeError):
    """Raised when the dataset cannot satisfy the requested manifest."""


@dataclass(frozen=True)
class ClassSelection:
    class_name: str
    source_files: int  # valid images found in the class folder
    valid_unique: int  # after exact dedup
    eligible_groups: int  # groups not already consumed by an earlier class
    groups_used: int
    extra_representatives: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class BuildResult:
    rows: tuple[ManifestRow, ...]
    scan: ScanResult
    dedup: ExactDedupResult
    groups: tuple[DuplicateGroup, ...]
    members_by_group: dict[str, tuple[ImageRecord, ...]]
    selections: tuple[ClassSelection, ...]
    counts_by_class: dict[str, dict[str, int]]
    totals: dict[str, int]


def _class_rng(seed: int, purpose: str, class_name: str) -> random.Random:
    """Deterministic RNG independent across classes and purposes.

    ``hash()`` is salted per process and the global ``random`` state is shared,
    so neither is acceptable here.
    """
    digest = hashlib.sha256(f"{seed}:{purpose}:{class_name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _subset_for_target(sizes: Sequence[int], target: int) -> list[int] | None:
    """0/1 subset-sum: indices of units whose sizes add up to ``target``."""
    if target == 0:
        return []
    reachable = [False] * (target + 1)
    reachable[0] = True
    choice = [-1] * (target + 1)
    for index, size in enumerate(sizes):
        if size > target:
            continue
        for total in range(target, size - 1, -1):
            if not reachable[total] and reachable[total - size]:
                reachable[total] = True
                choice[total] = index
    if not reachable[target]:
        return None
    selected: list[int] = []
    total = target
    while total > 0:
        index = choice[total]
        selected.append(index)
        total -= sizes[index]
    return selected


def assign_splits(
    units: Sequence[Sequence[ImageRecord]],
    allocation: SplitAllocation,
    rng: random.Random,
) -> list[tuple[str, tuple[ImageRecord, ...]]]:
    """Assign whole units to train/val/test honouring the exact allocation."""
    train_target = allocation.train
    val_target = allocation.val
    test_target = allocation.test

    order = [tuple(unit) for unit in units]
    rng.shuffle(order)
    if sum(len(unit) for unit in order) != train_target + val_target + test_target:
        raise BuildError("Split allocation does not match the number of selected images")

    if all(len(unit) == 1 for unit in order):
        assignment: list[tuple[str, tuple[ImageRecord, ...]]] = []
        for index, unit in enumerate(order):
            if index < train_target:
                split = "train"
            elif index < train_target + val_target:
                split = "val"
            else:
                split = "test"
            assignment.append((split, unit))
        return assignment

    sizes = [len(unit) for unit in order]
    train_indices = _subset_for_target(sizes, train_target)
    if train_indices is None:
        raise BuildError("Cannot reach the train target without splitting a near-duplicate group")
    train_set = set(train_indices)

    remaining = [index for index in range(len(order)) if index not in train_set]
    val_local = _subset_for_target([sizes[index] for index in remaining], val_target)
    if val_local is None:
        raise BuildError("Cannot reach the val target without splitting a near-duplicate group")
    val_set = {remaining[local] for local in val_local}

    assignment = []
    for index, unit in enumerate(order):
        if index in train_set:
            split = "train"
        elif index in val_set:
            split = "val"
        else:
            split = "test"
        assignment.append((split, unit))
    if sum(len(unit) for split, unit in assignment if split == "test") != test_target:
        raise BuildError("Test split does not match the allocation after assignment")
    return assignment


@dataclass(frozen=True)
class _SelectionDraft:
    units: list[list[ImageRecord]]
    extra_representatives: int
    eligible_groups: int


def _select_class_units(
    class_name: str,
    groups: Sequence[DuplicateGroup],
    members_by_group: dict[str, tuple[ImageRecord, ...]],
    seed: int,
    used_groups: set[str],
) -> _SelectionDraft:
    """Pick ``IMAGES_PER_CLASS`` images, at most one near-duplicate group each.

    If a class cannot provide 125 distinct groups, extra members of already
    selected groups are appended (they stay in the same unit, hence the same
    split).  A class that still falls short fails loudly.
    """
    rng = _class_rng(seed, "select", class_name)
    candidates = [group for group in groups if group.group_id not in used_groups]
    eligible = len(candidates)
    rng.shuffle(candidates)

    units: list[list[ImageRecord]] = []
    unit_by_group: dict[str, list[ImageRecord]] = {}
    for group in candidates:
        if group.group_id in used_groups:
            continue
        members = ordered_members(
            [
                member
                for member in members_by_group[group.group_id]
                if member.class_name == class_name
            ]
        )
        if not members:
            continue
        used_groups.add(group.group_id)
        unit = [members[0]]
        units.append(unit)
        unit_by_group[group.group_id] = unit
        if len(units) == IMAGES_PER_CLASS:
            break

    selected_total = sum(len(unit) for unit in units)
    extras = 0
    if selected_total < IMAGES_PER_CLASS:
        for group in candidates:
            unit = unit_by_group.get(group.group_id)
            if unit is None:
                continue
            for member in members_by_group[group.group_id][1:]:
                if member.class_name != class_name or member in unit:
                    continue
                unit.append(member)
                extras += 1
                selected_total += 1
                if selected_total == IMAGES_PER_CLASS:
                    break
            if selected_total == IMAGES_PER_CLASS:
                break

    if selected_total != IMAGES_PER_CLASS:
        raise BuildError(
            f"{class_name}: only {selected_total} distinct images available after "
            f"duplicate grouping, need {IMAGES_PER_CLASS}"
        )
    return _SelectionDraft(units=units, extra_representatives=extras, eligible_groups=eligible)


def _check_invariants(result_rows: Sequence[ManifestRow]) -> None:
    if len(result_rows) != IMAGES_PER_CLASS * len(CLASSES):
        raise BuildError(
            f"Manifest has {len(result_rows)} rows, expected {IMAGES_PER_CLASS * len(CLASSES)}"
        )

    per_class: dict[str, dict[str, int]] = {name: dict.fromkeys(SPLITS, 0) for name in CLASSES}
    for row in result_rows:
        per_class[row.class_name][row.split] += 1
    for class_name in CLASSES:
        allocation = SPLIT_ALLOCATION[class_name]
        expected = {"train": allocation.train, "val": allocation.val, "test": allocation.test}
        if per_class[class_name] != expected:
            raise BuildError(f"{class_name}: split counts {per_class[class_name]} != {expected}")

    splits_by_group: dict[str, set[str]] = {}
    for row in result_rows:
        splits_by_group.setdefault(row.near_dup_group, set()).add(row.split)
    leaking = {group: splits for group, splits in splits_by_group.items() if len(splits) > 1}
    if leaking:
        raise BuildError(f"Near-duplicate groups leaked across splits: {sorted(leaking)[:5]}")

    if len({row.relative_path for row in result_rows}) != len(result_rows):
        raise BuildError("Duplicate relative_path values in the manifest")
    if len({row.sha256 for row in result_rows}) != len(result_rows):
        raise BuildError("Duplicate SHA-256 values in the manifest")


def build_manifest(
    dataset_dir: Path,
    seed: int,
    *,
    dhash_threshold: int = DEFAULT_DHASH_THRESHOLD,
    validate_images: bool = True,
) -> BuildResult:
    """Build the complete 1,000-image manifest (deterministic for a given seed)."""
    validate_config()
    dataset_dir = Path(dataset_dir)
    scan = scan_dataset(dataset_dir, validate_images=validate_images)
    if not scan.records:
        raise BuildError(f"No valid images found under {dataset_dir}")

    dedup = deduplicate_exact(scan.records)
    groups = find_duplicate_groups(dedup.records, dhash_threshold=dhash_threshold)

    records_by_path = {record.relative_path: record for record in dedup.records}
    members_by_group = {
        group.group_id: tuple(records_by_path[path] for path in group.members) for group in groups
    }
    group_id_by_path = {path: group.group_id for group in groups for path in group.members}

    groups_by_class: dict[str, list[DuplicateGroup]] = {name: [] for name in CLASSES}
    for group in groups:
        classes = {member.class_name for member in members_by_group[group.group_id]}
        for class_name in classes:
            groups_by_class[class_name].append(group)

    rows: list[ManifestRow] = []
    selections: list[ClassSelection] = []
    used_groups: set[str] = set()

    for class_name in CLASSES:
        source_files = sum(1 for record in scan.records if record.class_name == class_name)
        valid_unique = sum(1 for record in dedup.records if record.class_name == class_name)
        draft = _select_class_units(
            class_name,
            groups_by_class[class_name],
            members_by_group,
            seed,
            used_groups,
        )
        assignment = assign_splits(
            draft.units,
            SPLIT_ALLOCATION[class_name],
            _class_rng(seed, "split", class_name),
        )
        split_counts = dict.fromkeys(SPLITS, 0)
        for split, unit in assignment:
            for record in unit:
                split_counts[split] += 1
                rows.append(
                    ManifestRow(
                        relative_path=record.relative_path,
                        class_name=class_name,
                        class_idx=record.class_idx,
                        split=split,
                        sha256=record.sha256,
                        near_dup_group=group_id_by_path[record.relative_path],
                    )
                )
        selections.append(
            ClassSelection(
                class_name=class_name,
                source_files=source_files,
                valid_unique=valid_unique,
                eligible_groups=draft.eligible_groups,
                groups_used=len(draft.units),
                extra_representatives=draft.extra_representatives,
                split_counts=split_counts,
            )
        )

    split_order = {split: index for index, split in enumerate(SPLITS)}
    rows.sort(key=lambda row: (row.class_idx, split_order[row.split], row.relative_path))
    _check_invariants(rows)

    counts_by_class: dict[str, dict[str, int]] = {}
    totals = dict.fromkeys(SPLITS, 0)
    for class_name in CLASSES:
        counts = dict.fromkeys(SPLITS, 0)
        for row in rows:
            if row.class_name == class_name:
                counts[row.split] += 1
        counts["total"] = sum(counts[split] for split in SPLITS)
        counts_by_class[class_name] = counts
        for split in SPLITS:
            totals[split] += counts[split]
    totals["total"] = sum(totals[split] for split in SPLITS)

    return BuildResult(
        rows=tuple(rows),
        scan=scan,
        dedup=dedup,
        groups=groups,
        members_by_group=members_by_group,
        selections=tuple(selections),
        counts_by_class=counts_by_class,
        totals=totals,
    )
