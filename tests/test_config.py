from __future__ import annotations

from mangoleaf.config import (
    CLASS_TO_IDX,
    CLASSES,
    IDX_TO_CLASS,
    IMAGES_PER_CLASS,
    MANIFEST_COLUMNS,
    SPLIT_ALLOCATION,
    SPLIT_TOTALS,
    TOTAL_IMAGES,
    validate_config,
)


def test_config_is_internally_consistent() -> None:
    validate_config()


def test_exactly_eight_classes_in_fixed_alphabetical_order() -> None:
    assert CLASSES == (
        "Anthracnose",
        "Bacterial Canker",
        "Cutting Weevil",
        "Die Back",
        "Gall Midge",
        "Healthy",
        "Powdery Mildew",
        "Sooty Mould",
    )
    assert [CLASS_TO_IDX[name] for name in CLASSES] == list(range(8))
    assert all(IDX_TO_CLASS[index] == name for name, index in CLASS_TO_IDX.items())


def test_allocation_matches_issue_1() -> None:
    assert {name: (a.train, a.val, a.test) for name, a in SPLIT_ALLOCATION.items()} == {
        "Anthracnose": (88, 18, 19),
        "Bacterial Canker": (88, 18, 19),
        "Cutting Weevil": (88, 19, 18),
        "Die Back": (88, 19, 18),
        "Gall Midge": (87, 19, 19),
        "Healthy": (87, 19, 19),
        "Powdery Mildew": (87, 19, 19),
        "Sooty Mould": (87, 19, 19),
    }
    assert all(allocation.total == IMAGES_PER_CLASS for allocation in SPLIT_ALLOCATION.values())
    assert SPLIT_TOTALS == {"train": 700, "val": 150, "test": 150}
    assert sum(SPLIT_TOTALS.values()) == TOTAL_IMAGES == 1000


def test_manifest_columns_are_the_documented_contract() -> None:
    assert MANIFEST_COLUMNS == (
        "relative_path",
        "class",
        "class_idx",
        "split",
        "sha256",
        "near_dup_group",
    )
