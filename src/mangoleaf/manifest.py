"""Manifest schema, I/O, fingerprinting and human-readable summary."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from mangoleaf.config import (
    CLASS_TO_IDX,
    CLASSES,
    DATASET_DOI,
    DATASET_NAME,
    DATASET_VERSION,
    MANIFEST_COLUMNS,
    SPLITS,
)


class ManifestError(RuntimeError):
    """Raised when a manifest file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ManifestRow:
    relative_path: str
    class_name: str
    class_idx: int
    split: str
    sha256: str
    near_dup_group: str

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "class": self.class_name,
            "class_idx": self.class_idx,
            "split": self.split,
            "sha256": self.sha256,
            "near_dup_group": self.near_dup_group,
        }


@dataclass(frozen=True)
class Manifest:
    path: Path
    rows: tuple[ManifestRow, ...]
    fingerprint: str

    def by_split(self, split: str) -> tuple[ManifestRow, ...]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        return tuple(row for row in self.rows if row.split == split)

    def by_class(self, class_name: str) -> tuple[ManifestRow, ...]:
        return tuple(row for row in self.rows if row.class_name == class_name)

    def class_counts(self) -> dict[str, dict[str, int]]:
        counts = {name: dict.fromkeys(SPLITS, 0) for name in CLASSES}
        for row in self.rows:
            counts[row.class_name][row.split] += 1
        return counts

    def split_counts(self) -> dict[str, int]:
        return {split: sum(1 for row in self.rows if row.split == split) for split in SPLITS}


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(rows: list[ManifestRow] | tuple[ManifestRow, ...], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def load_manifest(path: Path) -> Manifest:
    """Load the shared manifest and validate its schema before returning it."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != MANIFEST_COLUMNS:
            raise ManifestError(f"Unexpected CSV header {header}; expected {MANIFEST_COLUMNS}")
        rows: list[ManifestRow] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                class_name = raw["class"]
                class_idx = int(raw["class_idx"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ManifestError(f"Malformed row at line {line_number}: {raw}") from exc
            if class_name not in CLASS_TO_IDX:
                raise ManifestError(f"Line {line_number}: unknown class {class_name!r}")
            if CLASS_TO_IDX[class_name] != class_idx:
                raise ManifestError(
                    f"Line {line_number}: class_idx {class_idx} does not match {class_name}"
                )
            if raw["split"] not in SPLITS:
                raise ManifestError(f"Line {line_number}: unknown split {raw['split']!r}")
            if not raw["relative_path"] or not raw["sha256"] or not raw["near_dup_group"]:
                raise ManifestError(f"Line {line_number}: empty required field in {raw}")
            rows.append(
                ManifestRow(
                    relative_path=raw["relative_path"],
                    class_name=class_name,
                    class_idx=class_idx,
                    split=raw["split"],
                    sha256=raw["sha256"],
                    near_dup_group=raw["near_dup_group"],
                )
            )

    return Manifest(path=path, rows=tuple(rows), fingerprint=manifest_sha256(path))


def class_index_document() -> dict[str, object]:
    return {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "classes": [{"class_idx": CLASS_TO_IDX[name], "class": name} for name in CLASSES],
    }


def write_json(document: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_summary_markdown(meta: dict) -> str:
    generation = meta["generation"]
    duplicates = meta["duplicates"]
    totals = meta["totals"]

    allocation_rows = [
        [
            entry["class_idx"],
            entry["class"],
            entry["train"],
            entry["val"],
            entry["test"],
            entry["total"],
        ]
        for entry in meta["selection"]["per_class"]
    ]
    allocation_rows.append(
        [
            "",
            "**Total**",
            totals["by_split"]["train"],
            totals["by_split"]["val"],
            totals["by_split"]["test"],
            totals["rows"],
        ]
    )

    exact = duplicates["exact"]
    near = duplicates["near_duplicate"]
    dataset = meta["dataset"]
    validation = meta["validation"]
    name = dataset["name"]
    version = dataset["version"]
    signal_counts = near["signals"]
    signal_rows = [[signal, count] for signal, count in sorted(signal_counts.items())]

    return f"""# Manifiesto compartido — {name} {version} (1.000 imágenes)

Generado para la Issue #1: particiones únicas que consumirán las tres corridas
(MobileNetV2 base, MobileNetV2 ± Dropout y ResNet-18) sin recalcular nada.

## Datos de la generación

- **Dataset:** {name} {version} — DOI [{DATASET_DOI}](https://doi.org/{DATASET_DOI})
- **Imágenes fuente:** {dataset["source_images"]} ({dataset["source_images_per_class"]}/clase)
- **Semilla:** `{generation["seed"]}`
- **Comando exacto:** `{generation["command"]}`
- **SHA-256 del manifiesto:** `{meta["fingerprint"]}`
- **Filtro perceptual:** dHash 64 bits, 4 orientaciones, umbral {generation["dhash_threshold"]}
- **Versiones:** Python {generation["python"]}, Pillow {generation["pillow"]}

## Reparto por clase y partición

{_markdown_table(["class_idx", "Clase", "train", "val", "test", "total"], allocation_rows)}

## Duplicados y solapamiento

- Copias exactas eliminadas antes del muestreo: **{exact["removed"]}**
  ({exact["groups"]} hashes repetidos; {exact["cross_class_conflicts"]} conflictos entre clases).
- Grupos near-duplicate: **{near["groups"]}**;
  con más de una imagen: **{near["multi_member_groups"]}**;
  tamaño máximo: **{near["max_members"]}**;
  grupos que cruzan clases: **{near["cross_class_groups"]}**.
- Señales que unieron grupos:

{_markdown_table(["Señal", "Grupos"], signal_rows)}

## Identificadores de hoja (lo que se pudo comprobar)

- IDs de hoja publicados por los autores: **no disponibles**. Los nombres codifican el
  instante de captura (`YYYYMMDD_HHMMSS`) y el posprocesado (`(Custom)`, `- Copy`, `(n)`).
- Los autores declaran ~1.800 hojas distintas entre las 4.000 imágenes; el resto son
  variantes por zoom/rotación.
- Señales usadas: SHA-256 exacto, stem normalizado, timestamp de captura y dHash con
  rotaciones de 90°.
- **Alcance:** las copias exactas se eliminan por completo; las variantes re-guardadas,
  ampliadas y rotadas en múltiplos de 90° quedan agrupadas. Rotaciones arbitrarias o
  recortes agresivos **no son demostrables** con los metadatos disponibles y se declaran
  como riesgo residual.

## Integridad

- Imágenes válidas verificadas con Pillow: **{validation["images_checked"]}**.
- Archivos inválidos: **{len(validation["invalid"])}**.
- Filas del manifiesto: **{totals["rows"]}** (125 por clase; 700/150/150).

## Riesgo residual

{meta["residual_risk"]}
"""
