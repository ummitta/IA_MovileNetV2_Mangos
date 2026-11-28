"""Entrena variantes pareadas de MobileNetV2 y compara sus mejores checkpoints."""

import argparse
import csv
import hashlib
import json
import math
from copy import deepcopy
from itertools import islice
from pathlib import Path

from _bootstrap import setup_src_path

setup_src_path()

import torch  # noqa: E402
import torchvision  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2  # noqa: E402

from mangoleaf.config import (  # noqa: E402
    CLASS_TO_IDX,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIR,
    DEFAULT_SEED,
)
from mangoleaf.loaders import build_loaders  # noqa: E402


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def epoch(model, batches, device, optimizer=None):
    total = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels, _ in batches:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(labels)
            count += len(labels)
    mean = total / count
    if not math.isfinite(mean):
        raise ValueError("Pérdida no finita")
    return mean


def run_variant(model, args, dropout, probe, config):
    folder = args.output / f"dropout_{dropout:.1f}"
    folder.mkdir()
    model.classifier[0].p = dropout
    model.to(args.device, memory_format=torch.channels_last)
    torch.manual_seed(args.seed)
    loaders = build_loaders(args.dataset_dir, args.manifest, batch_size=32, seed=args.seed)
    save_json(folder / "config.json", {**config, "dropout": dropout})
    best = float("inf")
    best_record = {}
    with (folder / "history.csv").open("w", newline="") as file:
        columns = ("phase", "epoch", "lr", "train_loss", "val_loss")
        history = csv.DictWriter(file, fieldnames=columns)
        history.writeheader()
        for phase, epochs, lr in [("frozen", 10, 1e-4), ("fine_tune", 20, 1e-5)]:
            frozen = phase == "frozen"
            model.features.requires_grad_(not frozen)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
            phase_best, stale, plateau = float("inf"), 0, 0
            for step in range(1, (1 if args.smoke else epochs) + 1):
                model.train()
                model.features.train(not frozen)  # Congelar también estadísticas de BatchNorm.
                weight = model.features[0][0].weight
                before = weight.detach().clone() if step == 1 else None
                active_lr = optimizer.param_groups[0]["lr"]
                limit = 1 if args.smoke else None
                train = epoch(model, islice(loaders["train"], limit), args.device, optimizer)
                if before is not None and torch.equal(before, weight) != frozen:
                    raise RuntimeError(f"Actualización inesperada del extractor en {phase}")
                model.eval()
                val = epoch(model, islice(loaders["val"], limit), args.device)
                record = dict(phase=phase, epoch=step, lr=active_lr, train_loss=train, val_loss=val)
                history.writerow(record)
                file.flush()
                print(f"p={dropout} {phase} {step}: train={train:.5f} val={val:.5f}", flush=True)
                if val < best:
                    best, best_record = val, record
                    torch.save(model.state_dict(), folder / "best.pt")
                if val < phase_best:
                    phase_best, stale, plateau = val, 0, 0
                else:
                    stale += 1
                    plateau += 1
                    if plateau == 5:
                        for group in optimizer.param_groups:
                            group["lr"] *= 0.5
                        plateau = 0
                    if stale == 10:
                        break
    model.load_state_dict(
        torch.load(folder / "best.pt", map_location=args.device, weights_only=True)
    )
    with torch.no_grad():
        logits = model(probe)
        if logits.shape != (len(probe), len(CLASS_TO_IDX)) or not logits.isfinite().all():
            raise ValueError("Checkpoint inválido")
    return {"dropout": dropout, **best_record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Un lote por fase y por partición.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    class_file = args.manifest.with_name("class_index.json")
    classes = json.loads(class_file.read_text())
    if {r["class"]: r["class_idx"] for r in classes["classes"]} != CLASS_TO_IDX:
        raise ValueError("El índice no coincide con el contrato de clases")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    if not isinstance(model.classifier[0], nn.Dropout):
        raise ValueError("Se esperaba Dropout en la cabeza de torchvision")
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_TO_IDX))
    probe = next(iter(build_loaders(args.dataset_dir, args.manifest)["val"]))[0]
    model.eval()
    with torch.no_grad():
        if model(probe).shape != (len(probe), len(CLASS_TO_IDX)):
            raise ValueError("La cabeza debe generar ocho logits por imagen")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "class_index.json", classes)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "weights": "IMAGENET1K_V1",
        "batch_size": 32,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "phase_epochs": [1, 1] if args.smoke else [10, 20],
        "phase_lr": [1e-4, 1e-5],
        "lr_factor": 0.5,
        "lr_patience": 5,
        "stop_patience": 10,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    probe = probe.to(args.device)
    results = [run_variant(deepcopy(model), args, p, probe, config) for p in (0.5, 0.0)]
    save_json(args.output / "comparison.json", results)
    print(results)


if __name__ == "__main__":
    main()
