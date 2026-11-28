"""Cargadores compartidos para MobileNetV2 y ResNet-18 con pesos ImageNet V1."""

from pathlib import Path
from random import Random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from mangoleaf.config import DEFAULT_SEED, SPLITS
from mangoleaf.manifest import ManifestRow, load_manifest


class LeafDataset(Dataset):
    def __init__(self, root: Path, rows: tuple[ManifestRow, ...], rng: Random | None):
        self.root = root.resolve()
        self.rows = rows
        self.rng = rng
        # Both IMAGENET1K_V1 weights use the same resize, crop and normalization.
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row.relative_path).resolve()
        if Path(row.relative_path).is_absolute() or not path.is_relative_to(self.root):
            raise ValueError(f"Ruta fuera del dataset: {row.relative_path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        rng = self.rng
        if rng is not None:
            if rng.random() < 0.5:
                image = F.hflip(image)
            if rng.random() < 0.5:
                image = F.affine(
                    image,
                    angle=rng.uniform(-10, 10),
                    translate=[0, 0],
                    scale=rng.uniform(1, 1.1),
                    shear=0,
                    interpolation=InterpolationMode.BILINEAR,
                )
            if rng.random() < 0.5:
                image = F.adjust_brightness(image, rng.uniform(0.85, 1.15))
        return self.preprocess(image), row.class_idx, row.relative_path


def build_loaders(
    dataset_dir: Path, manifest_path: Path, *, batch_size: int = 32, seed: int = DEFAULT_SEED
) -> dict[str, DataLoader]:
    """Crear cargadores nuevos al inicio de cada corrida para repetir orden y aumentos."""
    manifest = load_manifest(manifest_path)
    loaders = {}
    for split in SPLITS:
        training = split == "train"
        dataset = LeafDataset(
            Path(dataset_dir), manifest.by_split(split), Random(seed) if training else None
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            generator=torch.Generator().manual_seed(seed),
            drop_last=False,
        )
    return loaders
