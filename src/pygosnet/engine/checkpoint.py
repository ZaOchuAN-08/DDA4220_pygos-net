"""Atomic checkpoint save/load helpers."""
from __future__ import annotations

import os
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, epoch, best_score=float("inf"), scheduler=None, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_score": float(best_score),
        "config": config or {},
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return {"epoch": int(payload.get("epoch", 0)), "best_score": float(payload.get("best_score", float("inf"))), "config": payload.get("config", {})}
