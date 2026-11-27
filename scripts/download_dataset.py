#!/usr/bin/env python3
"""Descarga reproducible de MangoLeafBD v1 desde Mendeley Data.

Uso típico::

    python scripts/download_dataset.py --dest data/raw/mangoleafbd

Descarga el ZIP oficial (108.424.956 bytes), verifica su tamaño, lo extrae
eliminando la carpeta raíz ``MangoLeafBD Dataset/`` y comprueba el SHA-256 de
cada imagen contra el metadato publicado por Mendeley.  Si Mendeley bloqueara
la descarga automatizada, el script imprime el enlace directo y la ruta exacta
donde debe colocarse el ZIP para continuar con ``--archive``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

from mangoleaf.config import (  # noqa: E402
    ARCHIVE_FILENAME,
    CLASSES,
    DATASET_DOI,
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_RAW_DIR,
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_SOURCE_IMAGES,
    EXPECTED_SOURCE_IMAGES_PER_CLASS,
    IMAGE_EXTENSIONS,
    MENDELEY_ARCHIVE_URL,
    MENDELEY_METADATA_URL,
    MENDELEY_PAGE_URL,
    ZIP_ROOT_DIR,
)
from mangoleaf.dataset import sha256_file  # noqa: E402

USER_AGENT = "mangoleaf-issue1/1.0 (academic project; MangoLeafBD v1 downloader)"
CHUNK_SIZE = 1 << 16


class DownloadError(RuntimeError):
    """Raised when the archive or the extracted dataset is not as expected."""


def _strip_root(member: str) -> str | None:
    """Map a ZIP member to its path relative to the dataset root."""
    normalized = member.replace("\\", "/").lstrip("/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts:
        return None
    if parts[0] == ZIP_ROOT_DIR:
        parts = parts[1:]
    elif parts[0] not in CLASSES:
        return None
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def download_archive(url: str, destination: Path, *, timeout: float) -> Path:
    """Download the archive with HTTP range resume; return its final size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")

    if destination.is_file():
        size = destination.stat().st_size
        if size == EXPECTED_ARCHIVE_BYTES:
            print(f"[download] ZIP ya presente: {destination} ({size:,} bytes)")
            return destination
        print(f"[download] ZIP incompleto ({size:,} bytes); se vuelve a descargar")
        destination.unlink()

    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if part.is_file() and part.stat().st_size > 0:
        headers["Range"] = f"bytes={part.stat().st_size}-"

    print(f"[download] GET {url}")
    with requests.get(
        url, stream=True, timeout=timeout, headers=headers, allow_redirects=True
    ) as response:
        if response.status_code not in (200, 206):
            raise DownloadError(
                f"Mendeley respondió HTTP {response.status_code}. Enlace directo: {url}"
            )
        resuming = response.status_code == 206
        if resuming:
            content_range = response.headers.get("Content-Range", "")
            total = int(content_range.split("/")[-1]) if "/" in content_range else None
            mode = "ab"
            initial = part.stat().st_size
        else:
            length = response.headers.get("Content-Length")
            total = int(length) if length else None
            mode = "wb"
            initial = 0
            if part.exists():
                part.unlink()

        with (
            part.open(mode) as handle,
            tqdm(
                total=total,
                initial=initial,
                unit="B",
                unit_scale=True,
                desc="ZIP MangoLeafBD v1",
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))

    size = part.stat().st_size
    if size != EXPECTED_ARCHIVE_BYTES:
        raise DownloadError(
            f"Tamaño del ZIP inesperado: {size:,} bytes (esperado {EXPECTED_ARCHIVE_BYTES:,}). "
            "¿Descarga interrumpida? Vuelve a ejecutar el script para reanudar."
        )
    part.replace(destination)
    print(f"[download] OK: {destination} ({size:,} bytes)")
    return destination


def load_publisher_metadata(url: str, *, timeout: float, cache: Path | None) -> dict:
    """Fetch (or reuse) the Mendeley metadata with per-file SHA-256 hashes."""
    if cache is not None and cache.is_file():
        print(f"[metadata] usando caché {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))
    print(f"[metadata] GET {url}")
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    document = response.json()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(document), encoding="utf-8")
    return document


def expected_hashes(document: dict) -> dict[str, str]:
    """Basename -> SHA-256 published by Mendeley (None when unavailable)."""
    hashes: dict[str, str] = {}
    for entry in document.get("files", []):
        name = str(entry.get("filename", ""))
        digest = (entry.get("content_details") or {}).get("sha256_hash")
        if name and digest:
            hashes[Path(name).name] = str(digest).lower()
    return hashes


def _extract(archive: Path, temp_root: Path) -> tuple[dict[str, int], list[str]]:
    counts = {name: 0 for name in CLASSES}
    ignored: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise DownloadError(f"ZIP corrupto: primer miembro con CRC inválido = {bad}")
        for info in bundle.infolist():
            if info.is_dir():
                continue
            relative = _strip_root(info.filename)
            if relative is None:
                ignored.append(info.filename)
                continue
            target = temp_root / relative
            if target.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored.append(info.filename)
                continue
            class_name = relative.split("/", 1)[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=CHUNK_SIZE)
            counts[class_name] += 1
    return counts, ignored


def verify_extracted(dataset_dir: Path, hashes: dict[str, str], *, skip: bool) -> int:
    """Compare every extracted image against the publisher's SHA-256."""
    if skip or not hashes:
        reason = "--skip-hash-check" if skip else "metadato sin hashes"
        print(f"[verify] comprobación SHA-256 omitida ({reason})")
        return 0

    mismatches: list[str] = []
    unchecked: list[str] = []
    files = sorted(dataset_dir.rglob("*"))
    image_files = [
        path for path in files if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    for path in tqdm(image_files, desc="SHA-256 vs Mendeley", unit="img"):
        expected = hashes.get(path.name)
        if expected is None:
            unchecked.append(path.name)
            continue
        if sha256_file(path) != expected:
            mismatches.append(path.relative_to(dataset_dir).as_posix())
    if unchecked:
        print(f"[verify] AVISO: {len(unchecked)} archivos sin hash publicado (no comprobables)")
    if mismatches:
        preview = "\n  ".join(mismatches[:10])
        raise DownloadError(
            f"{len(mismatches)} imágenes no coinciden con el SHA-256 de Mendeley:\n  {preview}"
        )
    print(f"[verify] {len(image_files) - len(unchecked)} imágenes verificadas contra Mendeley")
    return len(image_files) - len(unchecked)


def write_source_marker(
    dataset_dir: Path,
    archive: Path,
    *,
    archive_sha256: str,
    counts: dict[str, int],
    verified: int,
    metadata_url: str,
) -> None:
    marker = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "doi": DATASET_DOI,
        "page_url": MENDELEY_PAGE_URL,
        "archive_url": MENDELEY_ARCHIVE_URL,
        "archive_filename": ARCHIVE_FILENAME,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "metadata_url": metadata_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_images": sum(counts.values()),
        "images_per_class": counts,
        "publisher_hashes_verified": verified,
        "extraction": f"ZIP root '{ZIP_ROOT_DIR}/' stripped into {dataset_dir.as_posix()}/<Class>/",
    }
    (dataset_dir / ".source.json").write_text(
        json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _already_complete(dataset_dir: Path) -> bool:
    if not (dataset_dir / ".source.json").is_file():
        return False
    for class_name in CLASSES:
        folder = dataset_dir / class_name
        if not folder.is_dir():
            return False
        images = sum(1 for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if images != EXPECTED_SOURCE_IMAGES_PER_CLASS:
            return False
    return True


def _print_counts(counts: dict[str, int]) -> None:
    print(f"\n{'Clase':<20}{'imágenes':>10}")
    print("-" * 30)
    for class_name in CLASSES:
        print(f"{class_name:<20}{counts[class_name]:>10}")
    print("-" * 30)
    print(f"{'TOTAL':<20}{sum(counts.values()):>10}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_RAW_DIR, help="carpeta destino")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="ruta del ZIP (por defecto <dest-parent>/'MangoLeafBD Dataset.zip'; "
        "si existe se reutiliza, útil si la descarga automática está bloqueada)",
    )
    parser.add_argument("--url", default=MENDELEY_ARCHIVE_URL, help="URL del ZIP oficial")
    parser.add_argument("--metadata-url", default=MENDELEY_METADATA_URL)
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="JSON de metadatos cacheado (se guarda/lee aquí)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true", help="reemplazar un dataset existente")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--keep-archive", action="store_true", default=True)
    args = parser.parse_args(argv)

    archive = args.archive or (args.dest.parent / ARCHIVE_FILENAME)
    destination = args.dest
    counts: dict[str, int] | None = None
    verified = 0

    try:
        if not args.overwrite and _already_complete(destination):
            print(f"[download] {destination} ya contiene las 4.000 imágenes y su marcador.")
            print("[download] Nada que hacer (usa --overwrite para reconstruir).")
            return 0

        if not archive.is_file() or archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
            download_archive(args.url, archive, timeout=args.timeout)
        else:
            print(f"[download] reutilizando ZIP local: {archive}")

        if args.metadata_json is None:
            args.metadata_json = archive.parent / "mendeley_metadata.json"
        document = load_publisher_metadata(
            args.metadata_url, timeout=args.timeout, cache=args.metadata_json
        )
        hashes = expected_hashes(document)

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".extract-") as tmp:
            temp_root = Path(tmp)
            counts, ignored = _extract(archive, temp_root)
            if ignored:
                print(
                    f"[extract] {len(ignored)} entradas ignoradas (p. ej. __MACOSX): {ignored[:3]}"
                )
            for class_name in CLASSES:
                if counts[class_name] != EXPECTED_SOURCE_IMAGES_PER_CLASS:
                    raise DownloadError(
                        f"{class_name}: {counts[class_name]} imágenes extraídas, "
                        f"esperadas {EXPECTED_SOURCE_IMAGES_PER_CLASS}"
                    )
            if sum(counts.values()) != EXPECTED_SOURCE_IMAGES:
                raise DownloadError(
                    f"Total extraído {sum(counts.values())}, esperado {EXPECTED_SOURCE_IMAGES}"
                )
            _print_counts(counts)

            if destination.exists():
                if not args.overwrite:
                    raise DownloadError(
                        f"{destination} ya existe. Usa --overwrite para reemplazarlo."
                    )
                for class_name in CLASSES:
                    shutil.rmtree(destination / class_name, ignore_errors=True)
            destination.mkdir(parents=True, exist_ok=True)
            for class_name in CLASSES:
                shutil.move(str(temp_root / class_name), str(destination / class_name))

        verified = verify_extracted(destination, hashes, skip=args.skip_hash_check)
        write_source_marker(
            destination,
            archive,
            archive_sha256=sha256_file(archive),
            counts=counts,
            verified=verified,
            metadata_url=args.metadata_url,
        )
        print(f"[done] Dataset listo en {destination}")
        print(f"[done] Marcador de procedencia: {destination / '.source.json'}")
        return 0
    except (DownloadError, requests.RequestException, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("\nDescarga manual (si el acceso automatizado está bloqueado):", file=sys.stderr)
        print(f"  1. Abre {MENDELEY_PAGE_URL} (DOI {DATASET_DOI}).", file=sys.stderr)
        print(
            f"  2. Descarga '{ARCHIVE_FILENAME}' ({EXPECTED_ARCHIVE_BYTES:,} bytes).",
            file=sys.stderr,
        )
        print(f"  3. Déjalo en: {archive}", file=sys.stderr)
        print("  4. Ejecuta: python scripts/download_dataset.py --archive", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
