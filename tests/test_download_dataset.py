import hashlib
import json
import zipfile
from pathlib import Path


def test_failed_overwrite_preserves_previous_dataset(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import download_dataset as download

    archive = tmp_path / "dataset.zip"
    destination = tmp_path / "raw"
    filename = "Healthy/sample.jpg"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(filename, b"new contents")

    monkeypatch.setattr(download, "CLASSES", ["Healthy"])
    monkeypatch.setattr(download, "EXPECTED_SOURCE_IMAGES_PER_CLASS", 1)
    monkeypatch.setattr(download, "EXPECTED_SOURCE_IMAGES", 1)
    monkeypatch.setattr(download, "EXPECTED_ARCHIVE_BYTES", archive.stat().st_size)
    metadata = tmp_path / "metadata.json"
    entry = {"filename": filename, "content_details": {"sha256_hash": "0" * 64}}
    metadata.write_text(json.dumps({"files": [entry]}))
    args = ["--dest", str(destination), "--archive", str(archive), "--metadata-json", str(metadata)]

    image = destination / filename
    image.parent.mkdir(parents=True)
    image.write_bytes(b"original")
    marker = destination / ".source.json"
    marker.write_text('{"version": 1}')
    assert download.main([*args, "--overwrite"]) == 1
    assert download.main(args) == 0
    assert image.read_bytes() == b"original"
    assert marker.read_text() == '{"version": 1}'

    entry["content_details"]["sha256_hash"] = hashlib.sha256(b"new contents").hexdigest()
    metadata.write_text(json.dumps({"files": [entry]}))
    assert download.main([*args, "--overwrite"]) == 0
    assert image.read_bytes() == b"new contents"
    assert json.loads(marker.read_text())["publisher_hashes_verified"] == 1
