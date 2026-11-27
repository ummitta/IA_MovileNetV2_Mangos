#!/usr/bin/env python3
"""Genera el manifiesto compartido de 1.000 imágenes (Issue #1) y sus metadatos.

Uso típico::

    python scripts/build_manifest.py \
        --dataset-dir data/raw/mangoleafbd \
        --seed 42 \
        --output data/manifests/mangoleaf_1000_manifest.csv

Produce, junto al CSV:

* ``<stem>.meta.json``     metadatos completos (seed, comando, duplicados, conteos)
* ``<stem>.summary.md``    resumen legible con las tablas y los hallazgos
* ``class_index.json``     mapeo fijo class <-> class_idx (0-7)
"""

from __future__ import annotations

import argparse
import platform
import shlex
import sys
import time
from collections import Counter
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

import PIL  # noqa: E402

from mangoleaf.build import BuildError, BuildResult, build_manifest  # noqa: E402
from mangoleaf.config import (  # noqa: E402
    CLASSES,
    DATASET_DOI,
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_DHASH_THRESHOLD,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIR,
    DEFAULT_SEED,
    EXPECTED_SOURCE_IMAGES_PER_CLASS,
    IMAGES_PER_CLASS,
    SPLITS,
)  # noqa: E402
from mangoleaf.dataset import DatasetError  # noqa: E402
from mangoleaf.manifest import (  # noqa: E402
    ManifestError,
    class_index_document,
    manifest_sha256,
    render_summary_markdown,
    write_json,
    write_manifest,
)

RESIDUAL_RISK = (
    "El dataset no publica un identificador de hoja. Se agruparon copias exactas (SHA-256), "
    "variantes re-guardadas con el mismo stem/timestamp y variantes rotadas en múltiplos de 90° "
    "o re-guardadas con dHash <= umbral. Rotaciones arbitrarias, recortes agresivos o cambios "
    "de escala que no conservan el nombre ni el dHash pueden escapar a la detección: el "
    "solapamiento entre particiones no puede demostrarse más allá de estas señales."
)


def _near_duplicate_stats(result: BuildResult, threshold: int) -> dict[str, object]:
    groups = result.groups
    members_by_id = {group.group_id: group.members for group in groups}
    class_by_path = {record.relative_path: record.class_name for record in result.dedup.records}

    multi = [group for group in groups if len(group.members) > 1]
    signal_counts: Counter[str] = Counter()
    for group in multi:
        for signal in group.signals:
            signal_counts[signal] += 1

    cross_class = 0
    for group in groups:
        classes = {class_by_path[path] for path in group.members}
        if len(classes) > 1:
            cross_class += 1

    return {
        "groups": len(groups),
        "multi_member_groups": len(multi),
        "singletons": len(groups) - len(multi),
        "max_members": max(len(members_by_id[group.group_id]) for group in groups) if groups else 0,
        "cross_class_groups": cross_class,
        "threshold": threshold,
        "signals": dict(signal_counts),
        "example_groups": [
            {
                "group_id": group.group_id,
                "members": list(group.members),
                "signals": list(group.signals),
            }
            for group in multi[:5]
        ],
    }


def _exact_stats(result: BuildResult) -> dict[str, object]:
    removed = result.dedup.removed
    kept_paths = Counter(kept.relative_path for kept, _ in removed)
    by_class: Counter[str] = Counter(dropped.class_name for _, dropped in removed)
    return {
        "removed": len(removed),
        "groups": len(kept_paths),
        "cross_class_conflicts": len(result.dedup.cross_class_conflicts),
        "removed_by_class": dict(by_class),
    }


def _leaf_identity(result: BuildResult) -> dict[str, object]:
    timestamps = Counter(
        record.capture_timestamp
        for record in result.dedup.records
        if record.capture_timestamp is not None
    )
    total_with_timestamp = sum(timestamps.values())
    return {
        "authors_leaf_id_published": False,
        "authors_declared_distinct_leaves": 1800,
        "authors_note": (
            "Los autores indican ~1.800 hojas distintas entre las 4.000 imágenes; el resto son "
            "variantes obtenidas ampliando o rotando la misma hoja."
        ),
        "filename_scheme": "timestamp de captura (YYYYMMDD_HHMMSS) + sufijos (Custom), - Copy, (n)",
        "files_with_capture_timestamp": total_with_timestamp,
        "timestamp_collisions": sum(1 for count in timestamps.values() if count > 1),
        "signals_used": [
            "SHA-256 exacto",
            "stem normalizado (sin sufijos ni prefijo IMG_)",
            "timestamp de captura",
            f"dHash 64 bits x 4 orientaciones (umbral {DEFAULT_DHASH_THRESHOLD})",
        ],
        "scope": (
            "Verificado que no existe ID de hoja en el dataset. Se agrupan copias exactas y "
            "variantes re-guardadas/rotadas en múltiplos de 90°; no es posible demostrar "
            "variantes con rotación arbitraria o recorte."
        ),
    }


def _print_report(result: BuildResult) -> None:
    header = f"{'idx':>3}  {'Clase':<18}{'train':>6}{'val':>5}{'test':>5}{'total':>6}"
    print(header)
    print("-" * len(header))
    for class_idx, selection in enumerate(result.selections):
        assert selection.class_name == CLASSES[class_idx]
        counts = selection.split_counts
        total = sum(counts[split] for split in SPLITS)
        print(
            f"{class_idx:>3}  "
            f"{selection.class_name:<18}{counts['train']:>6}{counts['val']:>5}"
            f"{counts['test']:>5}{total:>6}"
        )
    print("-" * len(header))
    print(
        f"{'':>3}  {'TOTAL':<18}{result.totals['train']:>6}{result.totals['val']:>5}"
        f"{result.totals['test']:>5}{result.totals['total']:>6}"
    )
    print()
    print(
        f"Fuente: {len(result.scan.records) + len(result.scan.invalid)} imágenes válidas "
        f"({len(result.scan.invalid)} inválidas, {len(result.scan.ignored)} archivos ignorados)"
    )
    print(
        f"Copias exactas eliminadas: {len(result.dedup.removed)} "
        f"(conflictos entre clases: {len(result.dedup.cross_class_conflicts)})"
    )
    extras = sum(selection.extra_representatives for selection in result.selections)
    print(f"Representantes extra (clase con <125 grupos): {extras}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="raíz del dataset con las 8 carpetas de clase",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--meta", type=Path, default=None, help="(por defecto: <stem>.meta.json)")
    parser.add_argument(
        "--summary", type=Path, default=None, help="(por defecto: <stem>.summary.md)"
    )
    parser.add_argument(
        "--index", type=Path, default=None, help="(por defecto: class_index.json junto al CSV)"
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=DEFAULT_DHASH_THRESHOLD,
        help="umbral Hamming del dHash; <0 desactiva la señal perceptual",
    )
    parser.add_argument(
        "--no-validate-images",
        action="store_true",
        help="no decodificar con Pillow (solo para pruebas; no usar en la entrega)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    meta_path = args.meta or output.with_name(output.stem + ".meta.json")
    summary_path = args.summary or output.with_name(output.stem + ".summary.md")
    index_path = args.index or output.with_name("class_index.json")

    command = "python " + shlex.join([str(Path(sys.argv[0]))] + list(sys.argv[1:]))

    try:
        started = time.perf_counter()
        result = build_manifest(
            args.dataset_dir,
            args.seed,
            dhash_threshold=args.dhash_threshold,
            validate_images=not args.no_validate_images,
        )
        elapsed = time.perf_counter() - started

        write_manifest(result.rows, output)
        fingerprint = manifest_sha256(output)

        exact = _exact_stats(result)
        near = _near_duplicate_stats(result, args.dhash_threshold)
        selection_rows = []
        for selection in result.selections:
            counts = selection.split_counts
            selection_rows.append(
                {
                    "class_idx": CLASSES.index(selection.class_name),
                    "class": selection.class_name,
                    "train": counts["train"],
                    "val": counts["val"],
                    "test": counts["test"],
                    "total": sum(counts[split] for split in SPLITS),
                    "source_files": selection.source_files,
                    "valid_unique": selection.valid_unique,
                    "eligible_groups": selection.eligible_groups,
                    "groups_used": selection.groups_used,
                    "extra_representatives": selection.extra_representatives,
                }
            )

        meta = {
            "dataset": {
                "name": DATASET_NAME,
                "version": DATASET_VERSION,
                "doi": DATASET_DOI,
                "page_url": f"https://doi.org/{DATASET_DOI}",
                "source_images": len(result.scan.records) + len(result.scan.invalid),
                "source_images_per_class": EXPECTED_SOURCE_IMAGES_PER_CLASS,
            },
            "generation": {
                "seed": args.seed,
                "command": command,
                "python": platform.python_version(),
                "pillow": PIL.__version__,
                "dhash_threshold": args.dhash_threshold,
                "elapsed_seconds": round(elapsed, 2),
                "script": "scripts/build_manifest.py",
            },
            "selection": {
                "images_per_class": IMAGES_PER_CLASS,
                "per_class": selection_rows,
            },
            "totals": {
                "by_split": {
                    "train": result.totals["train"],
                    "val": result.totals["val"],
                    "test": result.totals["test"],
                },
                "rows": result.totals["total"],
            },
            "duplicates": {"exact": exact, "near_duplicate": near},
            "validation": {
                "images_checked": len(result.scan.records) + len(result.scan.invalid),
                "invalid": [
                    {"path": path, "reason": reason} for path, reason in result.scan.invalid
                ],
                "ignored": list(result.scan.ignored),
            },
            "leaf_identity": _leaf_identity(result),
            "fingerprint": fingerprint,
            "manifest_path": output.as_posix(),
            "residual_risk": RESIDUAL_RISK,
        }

        write_json(meta, meta_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_summary_markdown(meta), encoding="utf-8")
        write_json(class_index_document(), index_path)
    except (BuildError, DatasetError, ManifestError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _print_report(result)
    print()
    print(f"CSV:      {output} ({len(result.rows)} filas)")
    print(f"Metadatos:{meta_path}")
    print(f"Resumen:  {summary_path}")
    print(f"Índice:   {index_path}")
    print(f"SHA-256 del manifiesto: {fingerprint}")
    print(f"Tiempo de generación: {elapsed:.1f} s (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
