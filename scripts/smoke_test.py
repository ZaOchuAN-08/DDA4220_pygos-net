#!/usr/bin/env python
"""Smoke-test the A-owned Trainer without waiting for B/C implementations."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pygosnet.engine import Trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
    def forward(self, x):
        z = torch.nn.functional.adaptive_avg_pool2d(self.conv(x), (4, 4))
        return {"density": torch.relu(z[:, :1]), "segmentation": torch.sigmoid(z[:, 1:2])}


class TinyLoss(nn.Module):
    def forward(self, outputs, batch):
        loss = (outputs["density"] - batch["density"]).abs().mean()
        return {"total": loss, "structural": loss.detach(), "segmentation": loss.detach(), "background": loss.detach()}


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, n=8):
        self.images = torch.rand(n, 3, 32, 32)
        self.density = torch.rand(n, 1, 4, 4)
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return {"image": self.images[i], "density": self.density[i],
                "count": self.density[i].sum(), "seg_mask": (self.density[i] > 0).float(),
                "tier": "L1", "image_id": str(i)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/smoke_test")
    args = p.parse_args()
    model, criterion = TinyModel(), TinyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(DictDataset(), batch_size=4)
    trainer = Trainer(model, criterion, optimizer, "cpu", args.output, selection_start_epoch=1)
    trainer.fit(loader, loader, epochs=2, config={"smoke_test": True})
    state = trainer.resume(Path(args.output) / "last.pt")
    assert state["epoch"] == 2
    assert (Path(args.output) / "best.pt").exists()
    print("SMOKE TEST PASSED: forward/backward, validation, CSV logging, checkpoint save/resume")

if __name__ == "__main__": main()
