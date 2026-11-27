"""Single source of truth for the dataset, the classes and the shared partitions.

The ``class_idx`` mapping (0-7) and the split allocation are a contract shared by
every training/evaluation run of the group.  Nothing else in the project may
redefine them: import them from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Dataset source (Mendeley Data, official publication) -------------------
DATASET_NAME = "MangoLeafBD"
DATASET_VERSION = "v1"
DATASET_DOI = "10.17632/hxsnvwty3r.1"
MENDELEY_DATASET_ID = "hxsnvwty3r"
MENDELEY_VERSION = "1"
MENDELEY_PAGE_URL = "https://data.mendeley.com/datasets/hxsnvwty3r/1"
MENDELEY_METADATA_URL = "https://data.mendeley.com/public-api/datasets/hxsnvwty3r"
MENDELEY_ARCHIVE_URL = "https://data.mendeley.com/public-api/zip/hxsnvwty3r/download/1"

ARCHIVE_FILENAME = "MangoLeafBD Dataset.zip"
ZIP_ROOT_DIR = "MangoLeafBD Dataset"
EXPECTED_ARCHIVE_BYTES = 108_424_956
EXPECTED_SOURCE_IMAGES = 4_000
EXPECTED_SOURCE_IMAGES_PER_CLASS = 500

# --- Fixed class order (alphabetical) ---------------------------------------
CLASSES: tuple[str, ...] = (
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
)
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {idx: name for name, idx in CLASS_TO_IDX.items()}

SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class SplitAllocation:
    """Per-class split sizes.  ``train + val + test`` is always 125."""

    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


# The exact rounding requested by Issue #1: 700/150/150 globally, with
# 87-88 images per class in train and 18-19 in val/test.
SPLIT_ALLOCATION: dict[str, SplitAllocation] = {
    "Anthracnose": SplitAllocation(88, 18, 19),
    "Bacterial Canker": SplitAllocation(88, 18, 19),
    "Cutting Weevil": SplitAllocation(88, 19, 18),
    "Die Back": SplitAllocation(88, 19, 18),
    "Gall Midge": SplitAllocation(87, 19, 19),
    "Healthy": SplitAllocation(87, 19, 19),
    "Powdery Mildew": SplitAllocation(87, 19, 19),
    "Sooty Mould": SplitAllocation(87, 19, 19),
}

IMAGES_PER_CLASS = 125
TOTAL_IMAGES = 1000
SPLIT_TOTALS: dict[str, int] = {
    split: sum(allocation.__getattribute__(split) for allocation in SPLIT_ALLOCATION.values())
    for split in SPLITS
}

# --- Defaults ---------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_RAW_DIR = Path("data/raw/mangoleafbd")
DEFAULT_MANIFEST_PATH = Path("data/manifests/mangoleaf_1000_manifest.csv")
# Exact dHash match under one of the four 90-degree rotations. A larger
# tolerance chains perceptually similar leaves into one group (single-linkage
# transitive closure), which both under-counts distinct leaves and would make
# the 125-per-class target unreachable.
DEFAULT_DHASH_THRESHOLD = 0

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "relative_path",
    "class",
    "class_idx",
    "split",
    "sha256",
    "near_dup_group",
)


def validate_config() -> None:
    """Fail fast if the shared contract is internally inconsistent."""
    if len(CLASSES) != 8:
        raise ValueError(f"Expected exactly 8 classes, found {len(CLASSES)}")

    if set(CLASS_TO_IDX) != set(CLASSES) or len(CLASS_TO_IDX) != len(CLASSES):
        raise ValueError("CLASS_TO_IDX must be a bijection with CLASSES")

    if set(SPLIT_ALLOCATION) != set(CLASSES):
        raise ValueError("SPLIT_ALLOCATION must define every class exactly once")

    for name, allocation in SPLIT_ALLOCATION.items():
        if allocation.total != IMAGES_PER_CLASS:
            raise ValueError(
                f"{name}: split allocation sums to {allocation.total}, expected {IMAGES_PER_CLASS}"
            )
        if not 87 <= allocation.train <= 88:
            raise ValueError(f"{name}: train={allocation.train} is outside [87, 88]")
        if not 18 <= allocation.val <= 19:
            raise ValueError(f"{name}: val={allocation.val} is outside [18, 19]")
        if not 18 <= allocation.test <= 19:
            raise ValueError(f"{name}: test={allocation.test} is outside [18, 19]")

    if SPLIT_TOTALS != {"train": 700, "val": 150, "test": 150}:
        raise ValueError(f"Global split totals are wrong: {SPLIT_TOTALS}")

    if sum(SPLIT_TOTALS.values()) != TOTAL_IMAGES:
        raise ValueError("Split totals do not add up to TOTAL_IMAGES")
