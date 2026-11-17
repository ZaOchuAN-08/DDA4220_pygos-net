#!/usr/bin/env python
"""Training entry point; imports the Model and Loss modules owned by B and C."""
from __future__ import annotations

import argparse
import importlib
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from pygosnet.data.loader import build_loaders
from pygosnet.engine import Trainer
from pygosnet.engine.device import select_device


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=None, help="Override for short debug runs")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline engineering checks only")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    device = select_device(args.device)
    config["device"] = str(device)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.no_pretrained:
        config["model"].setdefault("kwargs", {})["pretrained"] = False
    if device.type != "cuda":
        config["data"]["pin_memory"] = False
    print(f"Device: {device}")

    model_cls = load_class(config["model"]["class_path"])
    loss_cls = load_class(config["loss"]["class_path"])
    model = model_cls(**config["model"].get("kwargs", {})).to(device)
    criterion = loss_cls(**config["loss"].get("kwargs", {})).to(device)
    if getattr(criterion, "use_hard_negative", False):
        print("NOTE: current background term is mean-square suppression, not proposal Top-K hard-negative loss.")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]),
                                 weight_decay=float(config["training"]["weight_decay"]))
    train_loader, val_loader = build_loaders(config, root)
    run_dir = root / config["output_dir"]
    trainer = Trainer(model, criterion, optimizer, device, run_dir,
                      selection_start_epoch=int(config["training"]["selection_start_epoch"]),
                      grad_clip_norm=config["training"].get("grad_clip_norm"))
    if args.resume:
        trainer.resume(args.resume)
    epochs = args.epochs if args.epochs is not None else int(config["training"]["epochs"])
    trainer.fit(train_loader, val_loader, epochs, config)


if __name__ == "__main__":
    main()
