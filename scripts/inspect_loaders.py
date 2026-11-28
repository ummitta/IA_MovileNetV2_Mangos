"""Inspecciona un lote por partición o valida el recorrido completo con --full."""

import argparse
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

import torch  # noqa: E402

from mangoleaf.config import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIR,
    DEFAULT_SEED,
    IDX_TO_CLASS,
    SPLIT_TOTALS,
)
from mangoleaf.loaders import build_loaders  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    loaders = build_loaders(
        args.dataset_dir, args.manifest, batch_size=args.batch_size, seed=args.seed
    )
    print(f"Índice de clases: {IDX_TO_CLASS}")
    for split, loader in loaders.items():
        count = 0
        classes = set()
        for images, labels, paths in loader:
            if count == 0:
                print(
                    f"{split}: {len(loader.dataset)} ejemplos; "
                    f"lote {tuple(images.shape)}, {images.dtype}"
                )
                print(f"  {paths[0]} -> {labels[0].item()} ({IDX_TO_CLASS[labels[0].item()]})")
            if not args.full:
                break
            if (
                images.shape[1:] != (3, 224, 224)
                or images.dtype != torch.float32
                or not torch.isfinite(images).all()
            ):
                raise ValueError(f"{split}: dimensiones, tipo o valores inválidos")
            count += len(labels)
            classes.update(labels.tolist())
        if args.full:
            if count != SPLIT_TOTALS[split] or classes != set(IDX_TO_CLASS):
                raise ValueError(f"{split}: conteo {count}, etiquetas {classes}")
            print(f"  Verificados: {count} ejemplos, {len(classes)} clases")
    if args.full:
        val = loaders["val"].dataset
        first, repeated = val[0], val[0]
        if first[1:] != repeated[1:] or not torch.equal(first[0], repeated[0]):
            raise ValueError("Validación: dos lecturas distintas para la misma imagen")
        print("Validación: tensor, etiqueta y ruta idénticos en dos lecturas.")


if __name__ == "__main__":
    main()
