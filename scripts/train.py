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
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cls = load_class(config["model"]["class_path"])
    loss_cls = load_class(config["loss"]["class_path"])
    model = model_cls(**config["model"].get("kwargs", {})).to(device)
    criterion = loss_cls(**config["loss"].get("kwargs", {})).to(device)
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
