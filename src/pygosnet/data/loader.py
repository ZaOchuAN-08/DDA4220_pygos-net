"""DataLoader factory with deterministic worker seeding."""
from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PenguinDataset


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(config, project_root="."):
    data_cfg = config["data"]
    root = project_root
    train_set = PenguinDataset(
        data_cfg["train_manifest"], root, "train", data_cfg["crop_size"],
        data_cfg["sigma"], data_cfg["stride"], data_cfg["support_threshold"])
    val_set = PenguinDataset(
        data_cfg["val_manifest"], root, "val", data_cfg["crop_size"],
        data_cfg["sigma"], data_cfg["stride"], data_cfg["support_threshold"])
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    common = {
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": bool(data_cfg["pin_memory"]),
        "worker_init_fn": seed_worker,
        "persistent_workers": int(data_cfg["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_set, batch_size=int(config["training"]["batch_size"]),
                              shuffle=True, drop_last=False, generator=generator, **common)
    val_loader = DataLoader(val_set, batch_size=int(config["training"]["batch_size"]),
                            shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
