"""Mandatory verification of the shared manifest (Issue #1 acceptance criteria)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from mangoleaf.config import (
    CLASS_TO_IDX,
    CLASSES,
    MANIFEST_COLUMNS,
    SPLIT_ALLOCATION,
    SPLITS,
    TOTAL_IMAGES,
)
from mangoleaf.manifest import (
    Manifest,
    ManifestError,
    ManifestRow,
    load_manifest,
    manifest_sha256,
    read_json,
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class VerificationReport:
    checks: tuple[Check, ...]
    manifest: Manifest | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _check(name: str, passed: bool, detail: str = "") -> Check:
    return Check(name=name, passed=passed, detail=detail)


def _pairwise_disjoint(rows_by_split: dict[str, list[ManifestRow]], attribute: str) -> list[str]:
    problems: list[str] = []
    for first_index, first in enumerate(SPLITS):
        for second in SPLITS[first_index + 1 :]:
            values_first = {getattr(row, attribute) for row in rows_by_split[first]}
            values_second = {getattr(row, attribute) for row in rows_by_split[second]}
            overlap = values_first & values_second
            if overlap:
                sample = sorted(overlap)[:3]
                problems.append(f"{first}∩{second}: {len(overlap)} (e.g. {sample})")
    return problems


def _allocation_row(class_name: str) -> dict[str, int]:
    allocation = SPLIT_ALLOCATION[class_name]
    return {"train": allocation.train, "val": allocation.val, "test": allocation.test}


def verify_manifest(
    manifest_path: Path,
    *,
    dataset_dir: Path | None = None,
    compare_path: Path | None = None,
    rehash: bool = False,
    check_images: bool = False,
) -> VerificationReport:
    manifest_path = Path(manifest_path)
    checks: list[Check] = []

    try:
        manifest = load_manifest(manifest_path)
    except (ManifestError, OSError) as exc:
        return VerificationReport(
            checks=(_check("Schema y carga del CSV", False, f"{type(exc).__name__}: {exc}"),)
        )

    checks.append(
        _check(
            "Esquema del CSV",
            True,
            f"columnas = {', '.join(MANIFEST_COLUMNS)}",
        )
    )
    checks.append(
        _check(
            "Filas == 1.000",
            len(manifest.rows) == TOTAL_IMAGES,
            f"{len(manifest.rows)} filas",
        )
    )

    rows_by_split: dict[str, list[ManifestRow]] = {split: [] for split in SPLITS}
    for row in manifest.rows:
        rows_by_split[row.split].append(row)

    class_counts: dict[str, dict[str, int]] = {name: dict.fromkeys(SPLITS, 0) for name in CLASSES}
    for row in manifest.rows:
        class_counts[row.class_name][row.split] += 1

    allocation_problems = [
        f"{name}: {class_counts[name]} != {_allocation_row(name)}"
        for name in CLASSES
        if class_counts[name] != _allocation_row(name)
    ]
    checks.append(
        _check(
            "Reparto por clase (125 y 87-88 / 18-19 / 18-19)",
            not allocation_problems,
            "; ".join(allocation_problems) if allocation_problems else "8 clases correctas",
        )
    )

    split_totals = {split: len(rows_by_split[split]) for split in SPLITS}
    checks.append(
        _check(
            "Totales 700 / 150 / 150",
            split_totals == {"train": 700, "val": 150, "test": 150},
            f"{split_totals}",
        )
    )

    idx_problems = [
        f"{row.relative_path}: {row.class_idx} != {CLASS_TO_IDX[row.class_name]}"
        for row in manifest.rows
        if CLASS_TO_IDX[row.class_name] != row.class_idx
    ]
    checks.append(
        _check(
            "class_idx fijo 0-7 coherente con la clase",
            not idx_problems,
            "; ".join(idx_problems[:3]) if idx_problems else f"{len(CLASSES)} clases",
        )
    )

    empty_fields = [
        row.relative_path
        for row in manifest.rows
        if not row.relative_path or not row.sha256 or not row.near_dup_group or not row.split
    ]
    checks.append(
        _check(
            "Sin campos vacíos",
            not empty_fields,
            f"{len(empty_fields)} filas incompletas" if empty_fields else "ok",
        )
    )

    bad_hashes = [
        row.relative_path
        for row in manifest.rows
        if len(row.sha256) != 64 or any(char not in "0123456789abcdef" for char in row.sha256)
    ]
    checks.append(
        _check(
            "SHA-256 con formato hexadecimal de 64 caracteres",
            not bad_hashes,
            f"{len(bad_hashes)} hashes inválidos" if bad_hashes else "ok",
        )
    )

    duplicate_paths = len(manifest.rows) - len({row.relative_path for row in manifest.rows})
    checks.append(
        _check(
            "Rutas relativas únicas",
            duplicate_paths == 0,
            f"{duplicate_paths} duplicadas" if duplicate_paths else "ok",
        )
    )

    duplicate_hashes = len(manifest.rows) - len({row.sha256 for row in manifest.rows})
    checks.append(
        _check(
            "SHA-256 únicos en el manifiesto",
            duplicate_hashes == 0,
            f"{duplicate_hashes} repetidos" if duplicate_hashes else "ok",
        )
    )

    path_problems = []
    for first_index, first in enumerate(SPLITS):
        for second in SPLITS[first_index + 1 :]:
            overlap = {row.relative_path for row in rows_by_split[first]} & {
                row.relative_path for row in rows_by_split[second]
            }
            if overlap:
                path_problems.append(f"{first}∩{second}={len(overlap)}")
    checks.append(
        _check(
            "Intersección de rutas entre particiones = ∅",
            not path_problems,
            "; ".join(path_problems) if path_problems else "train/val/test disjuntos",
        )
    )

    hash_problems = _pairwise_disjoint(rows_by_split, "sha256")
    checks.append(
        _check(
            "Intersección de SHA-256 entre particiones = ∅",
            not hash_problems,
            "; ".join(hash_problems) if hash_problems else "sin hashes compartidos",
        )
    )

    group_problems = _pairwise_disjoint(rows_by_split, "near_dup_group")
    checks.append(
        _check(
            "Intersección de near_dup_group entre particiones = ∅",
            not group_problems,
            "; ".join(group_problems) if group_problems else "sin grupos compartidos",
        )
    )

    groups_by_split: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            groups_by_split.setdefault(row.near_dup_group, set()).add(split)
    leaking_groups = [group for group, splits in groups_by_split.items() if len(splits) > 1]
    checks.append(
        _check(
            "Cada grupo near-duplicate vive en una sola partición",
            not leaking_groups,
            f"{len(leaking_groups)} grupos repartidos" if leaking_groups else "ok",
        )
    )

    if dataset_dir is not None:
        root = Path(dataset_dir).resolve()
        missing: list[str] = []
        escaping: list[str] = []
        for row in manifest.rows:
            candidate = (root / row.relative_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                escaping.append(row.relative_path)
                continue
            if not candidate.is_file():
                missing.append(row.relative_path)
        checks.append(
            _check(
                "Todas las rutas existen dentro de --dataset-dir",
                not missing and not escaping,
                (
                    f"{len(missing)} ausentes, {len(escaping)} fuera del root"
                    if (missing or escaping)
                    else f"{len(manifest.rows)} archivos encontrados"
                ),
            )
        )

        if rehash:
            mismatches: list[str] = []
            for row in manifest.rows:
                digest = hashlib.sha256()
                with (root / row.relative_path).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                if digest.hexdigest() != row.sha256:
                    mismatches.append(row.relative_path)
            checks.append(
                _check(
                    "SHA-256 recalculado coincide con el manifiesto",
                    not mismatches,
                    f"{len(mismatches)} discrepancias" if mismatches else "ok",
                )
            )

        if check_images:
            unreadable: list[str] = []
            for row in manifest.rows:
                try:
                    with Image.open(root / row.relative_path) as image:
                        image.load()
                except (UnidentifiedImageError, OSError) as exc:
                    unreadable.append(f"{row.relative_path} ({type(exc).__name__})")
            checks.append(
                _check(
                    "Las 1.000 imágenes abren con Pillow",
                    not unreadable,
                    f"{len(unreadable)} ilegibles" if unreadable else "ok",
                )
            )

    meta_path = manifest_path.with_name(manifest_path.stem + ".meta.json")
    if meta_path.is_file():
        try:
            meta = read_json(meta_path)
            recorded = meta.get("fingerprint")
            actual = manifest_sha256(manifest_path)
            meta_ok = recorded == actual
            detail = "coincide" if meta_ok else f"{recorded!r} != {actual!r}"
        except (OSError, ValueError) as exc:
            meta_ok = False
            detail = f"{type(exc).__name__}: {exc}"
        checks.append(_check("Fingerprint del meta coincide con el CSV", meta_ok, detail))

    if compare_path is not None:
        try:
            other = load_manifest(compare_path)
            same_rows = other.rows == manifest.rows
            same_fingerprint = other.fingerprint == manifest.fingerprint
        except (ManifestError, OSError) as exc:
            same_rows = same_fingerprint = False
            checks.append(
                _check("Comparación con la segunda corrida", False, f"{type(exc).__name__}: {exc}")
            )
        else:
            checks.append(
                _check(
                    "Comparación con la segunda corrida (fila por fila)",
                    same_rows,
                    "idénticas" if same_rows else "difieren",
                )
            )
            checks.append(
                _check(
                    "Mismo SHA-256 de archivo en ambas corridas",
                    same_fingerprint,
                    f"{manifest.fingerprint}" if same_fingerprint else "fingerprints distintos",
                )
            )

    return VerificationReport(checks=tuple(checks), manifest=manifest)


def render_report(report: VerificationReport) -> str:
    lines: list[str] = []
    lines.append(f"{'RESULTADO':<8} COMPROBACIÓN")
    lines.append("-" * 78)
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"{status:<8} {check.name}")
        if check.detail:
            lines.append(f"{'':<8}   -> {check.detail}")
    lines.append("-" * 78)
    failed = sum(1 for check in report.checks if not check.passed)
    if failed:
        lines.append(f"RESULTADO GLOBAL: FAIL ({failed} comprobación(es) no superadas)")
    else:
        lines.append(f"RESULTADO GLOBAL: PASS ({len(report.checks)} comprobaciones)")
    return "\n".join(lines)
