from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mangoleaf.dataset import sha256_file
from mangoleaf.manifest import write_manifest
from mangoleaf.verify import render_report, verify_manifest


def _failed_names(report) -> set[str]:
    return {check.name for check in report.checks if not check.passed}


def test_valid_manifest_passes(tmp_path: Path, valid_rows) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(valid_rows(), path)

    report = verify_manifest(path)

    assert report.passed, _failed_names(report)
    assert len(report.manifest.rows) == 1000  # type: ignore[union-attr]
    assert "RESULTADO GLOBAL: PASS" in render_report(report)


def test_comparison_against_second_run(tmp_path: Path, valid_rows) -> None:
    first = tmp_path / "run1.csv"
    second = tmp_path / "run2.csv"
    write_manifest(valid_rows(), first)
    write_manifest(valid_rows(), second)

    report = verify_manifest(first, compare_path=second)

    assert report.passed, _failed_names(report)
    names = {check.name for check in report.checks}
    assert "Comparación con la segunda corrida (fila por fila)" in names


def test_comparison_detects_different_rows(tmp_path: Path, valid_rows) -> None:
    first = tmp_path / "run1.csv"
    second = tmp_path / "run2.csv"
    rows = valid_rows()
    write_manifest(rows, first)
    write_manifest(rows[::-1], second)

    report = verify_manifest(first, compare_path=second)

    assert not report.passed
    assert "Comparación con la segunda corrida (fila por fila)" in _failed_names(report)


def test_leakage_is_detected(tmp_path: Path, valid_rows) -> None:
    rows = valid_rows()
    train_row = rows[0]
    val_row = next(row for row in rows if row.split == "val")
    rows[rows.index(val_row)] = replace(
        val_row, relative_path=train_row.relative_path, sha256=train_row.sha256
    )
    path = tmp_path / "leaky.csv"
    write_manifest(rows, path)

    report = verify_manifest(path)
    failed = _failed_names(report)

    assert not report.passed
    assert "Rutas relativas únicas" in failed
    assert "SHA-256 únicos en el manifiesto" in failed
    assert "Intersección de rutas entre particiones = ∅" in failed
    assert "Intersección de SHA-256 entre particiones = ∅" in failed


def test_missing_dataset_files_are_reported(tmp_path: Path, valid_rows) -> None:
    path = tmp_path / "manifest.csv"
    write_manifest(valid_rows(), path)

    report = verify_manifest(path, dataset_dir=tmp_path / "raw")

    assert not report.passed
    assert "Todas las rutas existen dentro de --dataset-dir" in _failed_names(report)


def test_rehash_and_image_checks(tmp_path: Path, valid_rows, write_image) -> None:
    dataset_dir = tmp_path / "raw"
    rows = []
    for serial, row in enumerate(valid_rows()):
        target = dataset_dir / row.relative_path
        write_image(target, size=8, seed=serial)
        rows.append(replace(row, sha256=sha256_file(target)))

    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    report = verify_manifest(path, dataset_dir=dataset_dir, rehash=True, check_images=True)

    assert report.passed, _failed_names(report)
    names = {check.name for check in report.checks}
    assert "SHA-256 recalculado coincide con el manifiesto" in names
    assert "Las 1.000 imágenes abren con Pillow" in names


def test_rehash_detects_modified_file(tmp_path: Path, valid_rows, write_image) -> None:
    dataset_dir = tmp_path / "raw"
    rows = []
    for serial, row in enumerate(valid_rows()):
        target = dataset_dir / row.relative_path
        write_image(target, size=8, seed=serial)
        rows.append(replace(row, sha256=sha256_file(target)))

    victim = dataset_dir / rows[0].relative_path
    write_image(victim, size=8, seed=99_999)

    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    report = verify_manifest(path, dataset_dir=dataset_dir, rehash=True)

    assert not report.passed
    assert "SHA-256 recalculado coincide con el manifiesto" in _failed_names(report)


def test_rehash_reports_missing_file_and_checks_next_row(tmp_path, valid_rows):
    missing, modified = valid_rows()[:2]
    image = tmp_path / modified.relative_path
    image.parent.mkdir(parents=True)
    image.write_bytes(b"modified")
    path = tmp_path / "manifest.csv"
    write_manifest([missing, modified], path)

    report = render_report(verify_manifest(path, dataset_dir=tmp_path, rehash=True))

    assert missing.relative_path in report
    assert modified.relative_path in report
    assert "RESULTADO GLOBAL: FAIL" in report
