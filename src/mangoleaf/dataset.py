"""Dataset discovery, file naming and image integrity validation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from mangoleaf.config import CLASS_TO_IDX, CLASSES, IMAGE_EXTENSIONS

ProgressCallback = Callable[[int, int], None]

# ``20211011_140211`` / ``IMG_20211011_140211`` capture timestamps.
_TIMESTAMP_RE = re.compile(r"(20\d{6}_\d{6})")

# Processing suffixes added by the authors: "(Custom)", "- Copy", "(1)", ...
_SUFFIX_PATTERNS = (
    re.compile(r"\s*-\s*copy(?:\s*-\s*copy)*\s*$", re.IGNORECASE),
    re.compile(r"(?:\s*\(\s*custom\s*\))+$", re.IGNORECASE),
    re.compile(r"\s*\(\d+\)$"),
)


class DatasetError(RuntimeError):
    """Raised when the raw dataset is missing, incomplete or malformed."""


@dataclass(frozen=True)
class ImageRecord:
    """One source image with the metadata needed to build the manifest."""

    relative_path: str  # POSIX path relative to the dataset root.
    class_name: str
    class_idx: int
    absolute_path: Path
    size_bytes: int
    sha256: str
    width: int
    height: int
    asset_stem: str
    capture_timestamp: str | None

    @property
    def filename(self) -> str:
        return self.absolute_path.name


@dataclass(frozen=True)
class ScanResult:
    records: tuple[ImageRecord, ...]
    invalid: tuple[tuple[str, str], ...]  # (relative_path, reason)
    ignored: tuple[str, ...]  # non-image files found inside class folders


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_asset_stem(filename: str) -> str:
    """Strip processing suffixes and the ``IMG_`` prefix from a file name.

    ``"IMG_20211011_140211 (Custom) (Custom) - Copy.jpg"`` -> ``"20211011_140211"``
    """
    stem = Path(filename).stem
    changed = True
    while changed:
        changed = False
        for pattern in _SUFFIX_PATTERNS:
            new_stem = pattern.sub("", stem).strip()
            if new_stem != stem:
                stem = new_stem
                changed = True
    if stem.upper().startswith("IMG_"):
        stem = stem[4:]
    return stem.strip()


def extract_capture_timestamp(filename: str) -> str | None:
    """Return the capture timestamp encoded in the file name, if any."""
    match = _TIMESTAMP_RE.search(Path(filename).stem)
    return match.group(1) if match else None


def _validate_layout(dataset_dir: Path) -> dict[str, Path]:
    if not dataset_dir.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {dataset_dir}")

    class_dirs: dict[str, Path] = {}
    missing: list[str] = []
    for class_name in CLASSES:
        candidate = dataset_dir / class_name
        if candidate.is_dir():
            class_dirs[class_name] = candidate
        else:
            missing.append(class_name)

    if missing:
        raise DatasetError(
            f"{dataset_dir} is missing {len(missing)} class folder(s): " + ", ".join(missing)
        )

    unexpected = sorted(
        entry.name
        for entry in dataset_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in CLASSES
    )
    if unexpected:
        raise DatasetError(f"Unexpected folder(s) in {dataset_dir}: " + ", ".join(unexpected))
    return class_dirs


def _iter_image_files(class_dir: Path) -> Iterator[Path]:
    for path in sorted(class_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def scan_dataset(
    dataset_dir: Path,
    *,
    validate_images: bool = True,
    on_progress: ProgressCallback | None = None,
) -> ScanResult:
    """Discover every class image, hash it and (optionally) decode it with Pillow.

    Invalid files are reported and excluded; they never reach the manifest.
    """
    dataset_dir = Path(dataset_dir)
    class_dirs = _validate_layout(dataset_dir)

    paths: list[tuple[str, Path]] = []
    ignored: list[str] = []
    for class_name in CLASSES:
        for path in class_dirs[class_name].iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append((class_name, path))
            else:
                ignored.append(path.relative_to(dataset_dir).as_posix())

    records: list[ImageRecord] = []
    invalid: list[tuple[str, str]] = []
    total = len(paths)

    for index, (class_name, path) in enumerate(
        sorted(paths, key=lambda item: (item[0], item[1].name))
    ):
        relative_path = path.relative_to(dataset_dir).as_posix()
        if validate_images:
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                invalid.append((relative_path, f"{type(exc).__name__}: {exc}"))
                if on_progress is not None:
                    on_progress(index + 1, total)
                continue
        else:
            width = height = 0

        records.append(
            ImageRecord(
                relative_path=relative_path,
                class_name=class_name,
                class_idx=CLASS_TO_IDX[class_name],
                absolute_path=path,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                width=width,
                height=height,
                asset_stem=normalize_asset_stem(path.name),
                capture_timestamp=extract_capture_timestamp(path.name),
            )
        )
        if on_progress is not None:
            on_progress(index + 1, total)

    return ScanResult(
        records=tuple(records), invalid=tuple(invalid), ignored=tuple(sorted(ignored))
    )
