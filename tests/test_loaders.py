from dataclasses import replace

import pytest
from PIL import Image

from mangoleaf.config import SPLITS
from mangoleaf.manifest import write_manifest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights  # noqa: E402

from mangoleaf.loaders import build_loaders  # noqa: E402


def test_shared_loaders_preserve_inputs_and_repeat_training(tmp_path, valid_rows, write_image):
    rows = []
    for idx, row in enumerate(valid_rows()[::110][:9]):
        row = replace(row, split=SPLITS[idx // 3])
        path = write_image(tmp_path / row.relative_path, size=24, seed=idx)
        if idx == 3:
            with Image.open(path) as source:
                grey = source.convert("L")
            grey.save(path)
        rows.append(row)
    csv_path = tmp_path / "manifest.csv"
    write_manifest(rows, csv_path)
    first = build_loaders(tmp_path, csv_path, batch_size=2)
    second = build_loaders(tmp_path, csv_path, batch_size=2)
    for split in SPLITS:
        seen = []
        for left, right in zip(first[split], second[split], strict=True):
            images, labels, paths = left
            assert images.shape == (len(paths), 3, 224, 224)
            assert images.dtype == torch.float32
            assert labels.dtype == torch.int64
            assert torch.equal(images, right[0]) and torch.equal(labels, right[1])
            assert paths == right[2]
            seen.extend(paths)
        expected = [row.relative_path for row in rows if row.split == split]
        assert sorted(seen) == sorted(expected)
        if split != "train":
            assert seen == expected
    images, labels, paths = next(iter(first["val"]))
    with Image.open(tmp_path / paths[0]) as source:
        for weights in (MobileNet_V2_Weights.IMAGENET1K_V1, ResNet18_Weights.IMAGENET1K_V1):
            assert torch.equal(images[0], weights.transforms()(source.convert("RGB")))
    assert labels[0].item() == rows[3].class_idx
