"""Fixtures compartidos: imágenes sintéticas y filas de manifiesto válidas."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image

from mangoleaf.config import CLASS_TO_IDX, CLASSES, SPLIT_ALLOCATION
from mangoleaf.manifest import ManifestRow


@pytest.fixture
def write_image():
    """Fábrica de imágenes JPEG sintéticas (ruido determinista o color plano)."""

    def factory(
        path: Path,
        *,
        size: int = 32,
        seed: int = 0,
        color: tuple[int, int, int] | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if color is None:
            data = random.Random(seed).randbytes(size * size * 3)
            image = Image.frombytes("RGB", (size, size), data)
        else:
            image = Image.new("RGB", (size, size), color)
        image.save(path, format="JPEG", quality=85)
        return path

    return factory


@pytest.fixture
def make_dataset(tmp_path, write_image):
    """Crea la estructura MangoLeafBD (8 carpetas) con ``n`` imágenes por clase."""

    def factory(images_per_class: int = 1, *, size: int = 32, prefix: str = "img") -> Path:
        root = tmp_path / "raw"
        for class_idx, class_name in enumerate(CLASSES):
            for index in range(images_per_class):
                write_image(
                    root / class_name / f"{prefix}_{class_idx:02d}_{index:04d}.jpg",
                    size=size,
                    seed=class_idx * 100_000 + index,
                )
        return root

    return factory


@pytest.fixture
def valid_rows():
    """1.000 filas que cumplen exactamente el contrato de la Issue #1."""

    def factory() -> list[ManifestRow]:
        rows: list[ManifestRow] = []
        for class_name in CLASSES:
            allocation = SPLIT_ALLOCATION[class_name]
            class_idx = CLASS_TO_IDX[class_name]
            for split, count in (
                ("train", allocation.train),
                ("val", allocation.val),
                ("test", allocation.test),
            ):
                for _ in range(count):
                    serial = len(rows)
                    rows.append(
                        ManifestRow(
                            relative_path=f"{class_name}/2021_{serial:04d}.jpg",
                            class_name=class_name,
                            class_idx=class_idx,
                            split=split,
                            sha256=f"{serial:064x}",
                            near_dup_group=f"g{serial:08d}",
                        )
                    )
        return rows

    return factory
