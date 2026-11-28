"""Duplicate and near-duplicate detection.

Signals, applied to every valid source image before any sampling:

1. Exact SHA-256 equality (cryptographic, exact).
2. Normalised file-name stem + capture timestamp (the authors re-saved some
   images as ``- Copy`` / ``(Custom)`` variants keeping the original name).
3. 64-bit dHash (perceptual) compared over four 90-degree orientations.

Groups produced here are *indivisible* for the stratified split, which is what
prevents identical or derived images from leaking across train/val/test.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from mangoleaf.config import DEFAULT_DHASH_THRESHOLD
from mangoleaf.dataset import ImageRecord, sha256_file

ProgressCallback = Callable[[int, int], None]

# Distance threshold on 64 hash bits.  6/64 keeps re-saved, zoomed and
# 90-degree-rotated variants together while staying far from unrelated leaves
# (which sit around 32/64).
_HASH_BITS = 64


@dataclass(frozen=True)
class ExactDedupResult:
    records: tuple[ImageRecord, ...]
    removed: tuple[tuple[ImageRecord, ImageRecord], ...]  # (kept, dropped)

    @property
    def cross_class_conflicts(self) -> tuple[tuple[ImageRecord, ImageRecord], ...]:
        return tuple(
            (kept, dropped)
            for kept, dropped in self.removed
            if kept.class_name != dropped.class_name
        )


@dataclass(frozen=True)
class DuplicateGroup:
    """A set of images that must stay in the same split."""

    group_id: str
    members: tuple[str, ...]  # sorted relative paths
    signals: tuple[str, ...]  # which signals united the members


def deduplicate_exact(records: Sequence[ImageRecord]) -> ExactDedupResult:
    """Keep one deterministic representative per SHA-256 (cleanest filename)."""
    by_hash: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_hash.setdefault(record.sha256, []).append(record)

    kept: list[ImageRecord] = []
    removed: list[tuple[ImageRecord, ImageRecord]] = []
    for group in by_hash.values():
        ordered = sorted(group, key=_representative_rank)
        kept.append(ordered[0])
        removed.extend((ordered[0], record) for record in ordered[1:])
    kept.sort(key=lambda record: record.relative_path)
    removed.sort(key=lambda pair: pair[1].relative_path)
    return ExactDedupResult(records=tuple(kept), removed=tuple(removed))


def dhash64(gray: Image.Image) -> int:
    """Difference hash of an 8x8 gradient grid (64 bits)."""
    small = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())  # 'L' mode -> one byte per pixel
    bits = 0
    for row in range(8):
        row_start = row * 9
        for column in range(8):
            bits = (bits << 1) | int(pixels[row_start + column] > pixels[row_start + column + 1])
    return bits


def dhash_orientations(image: Image.Image) -> tuple[int, int, int, int]:
    """dHash of the image rotated by 0, 90, 180 and 270 degrees."""
    gray = image.convert("L")
    return tuple(  # type: ignore[return-value]
        dhash64(gray if angle == 0 else gray.rotate(angle, expand=True))
        for angle in (0, 90, 180, 270)
    )


def min_rotation_hamming(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    """Smallest Hamming distance over the 16 orientation pairs."""
    best = _HASH_BITS
    for left_hash in left:
        for right_hash in right:
            distance = (left_hash ^ right_hash).bit_count()
            if distance < best:
                best = distance
                if best == 0:
                    return 0
    return best


def _representative_rank(record: ImageRecord) -> tuple[int, int, int, int, str]:
    """Prefer the cleanest file name: fewer copies, fewer suffixes, shorter."""
    name = record.filename
    lowered = name.lower()
    copies = lowered.count("- copy")
    customs = lowered.count("(custom)")
    numbered = len(re.findall(r"\(\d+\)", Path(name).stem))
    return (copies, customs, numbered, len(name), record.relative_path)


def choose_representative(members: Sequence[ImageRecord]) -> ImageRecord:
    return min(members, key=_representative_rank)


def ordered_members(members: Sequence[ImageRecord]) -> tuple[ImageRecord, ...]:
    """Group members from the cleanest (preferred representative) to the rest."""
    return tuple(sorted(members, key=_representative_rank))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size
        self.reasons: dict[int, set[str]] = {}

    def find(self, item: int) -> int:
        parent = self._parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int, reason: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            self.reasons.setdefault(root_left, set()).add(reason)
            return
        if self._rank[root_left] < self._rank[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        if self._rank[root_left] == self._rank[root_right]:
            self._rank[root_left] += 1
        merged = self.reasons.pop(root_right, set())
        reasons = self.reasons.setdefault(root_left, set())
        reasons.update(merged)
        reasons.add(reason)


def group_id_for(members: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()
    return digest[:12]


def find_duplicate_groups(
    records: Sequence[ImageRecord],
    *,
    dhash_threshold: int = DEFAULT_DHASH_THRESHOLD,
    hashes: Sequence[tuple[int, int, int, int]] | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[DuplicateGroup, ...]:
    """Union exact-name, timestamp and perceptual signals into groups.

    Groups never span classes: the splits are stratified per class, so a group
    that mixed classes could not be assigned to a single split and would starve
    the later class. Cross-class identical bytes are still reported by
    :func:`deduplicate_exact` (label conflicts) and by the verifier.

    ``dhash_threshold`` below 0 disables the perceptual signal (exact-name and
    timestamp grouping are still applied). The default of 0 only unites images
    whose dHash is identical under some 90-degree rotation; raising it produces
    transitive chains that merge visually similar but distinct leaves.
    """
    if not records:
        return ()

    size = len(records)
    union_find = _UnionFind(size)

    by_stem: dict[tuple[str, str], int] = {}
    by_timestamp: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        if record.asset_stem:
            key = (record.class_name, record.asset_stem)
            previous = by_stem.setdefault(key, index)
            if previous != index:
                union_find.union(previous, index, "asset_stem")
        if record.capture_timestamp:
            key = (record.class_name, record.capture_timestamp)
            previous = by_timestamp.setdefault(key, index)
            if previous != index:
                union_find.union(previous, index, "capture_timestamp")

    if dhash_threshold >= 0:
        if hashes is None:
            computed: list[tuple[int, int, int, int]] = []
            for index, record in enumerate(records):
                with Image.open(record.absolute_path) as image:
                    computed.append(dhash_orientations(image))
                if on_progress is not None:
                    on_progress(index + 1, size)
            hashes = computed
        assert hashes is not None
        for left in range(size - 1):
            left_hash = hashes[left]
            left_class = records[left].class_name
            for right in range(left + 1, size):
                if records[right].class_name != left_class:
                    continue
                if min_rotation_hamming(left_hash, hashes[right]) <= dhash_threshold:
                    union_find.union(left, right, "dhash")

    buckets: dict[int, list[int]] = {}
    for index in range(size):
        buckets.setdefault(union_find.find(index), []).append(index)

    groups: list[DuplicateGroup] = []
    for root, members in buckets.items():
        paths = tuple(sorted(records[index].relative_path for index in members))
        groups.append(
            DuplicateGroup(
                group_id=group_id_for(paths),
                members=paths,
                signals=tuple(sorted(union_find.reasons.get(root, set()))),
            )
        )

    groups.sort(key=lambda group: group.members[0])
    return tuple(groups)


def hash_records(
    records: Sequence[ImageRecord], on_progress: ProgressCallback | None = None
) -> list[tuple[int, int, int, int]]:
    """Compute the four dHash orientations for every record (helper for callers)."""
    hashes: list[tuple[int, int, int, int]] = []
    for index, record in enumerate(records):
        with Image.open(record.absolute_path) as image:
            hashes.append(dhash_orientations(image))
        if on_progress is not None:
            on_progress(index + 1, len(records))
    return hashes


def rehash(record: ImageRecord) -> str:
    """Recompute the on-disk hash (used by the verifier)."""
    return sha256_file(record.absolute_path)
