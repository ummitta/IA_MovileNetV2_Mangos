#!/usr/bin/env python3
"""Verifica el manifiesto compartido contra los criterios de la Issue #1.

Comprobaciones: 1.000 filas, 125 por clase, 700/150/150, class_idx coherente,
rutas y SHA-256 únicos, intersecciones vacías entre train/val/test, cada grupo
near-duplicate en una sola partición, existencia de los archivos, rehash
opcional, decodificación con Pillow y comparación fila por fila con una segunda
corrida (``--compare``).

Uso::

    python scripts/verify_manifest.py data/manifests/mangoleaf_1000_manifest.csv \
        --dataset-dir data/raw/mangoleafbd --rehash --check-images \
        --compare data/manifests/repro_b.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

from mangoleaf.config import DEFAULT_RAW_DIR  # noqa: E402
from mangoleaf.verify import render_report, verify_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="raíz del dataset para comprobar existencia y rehash",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="segundo CSV: comprobación fila por fila (reproducibilidad)",
    )
    parser.add_argument("--rehash", action="store_true", help="recalcular SHA-256 de cada imagen")
    parser.add_argument("--check-images", action="store_true", help="abrir cada imagen con Pillow")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_manifest(
        args.manifest,
        dataset_dir=args.dataset_dir,
        compare_path=args.compare,
        rehash=args.rehash,
        check_images=args.check_images,
    )
    print(render_report(report))
    if report.passed:
        print("\nEl manifiesto puede consumirse tal cual por las tres corridas.")
        return 0
    print("\nCorrige los fallos antes de usar el manifiesto.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
