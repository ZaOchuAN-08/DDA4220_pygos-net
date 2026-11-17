"""Manifest-backed BAS penguin dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .density import make_density_targets
from .transforms import EvalTransform, TrainTransform, image_to_tensor


def density_tier(count: int) -> str:
    if count == 0: return "L0"
    if count <= 100: return "L1"
    if count <= 500: return "L2"
    if count <= 1000: return "L3"
    return "L4"


class PenguinDataset(Dataset):
    """Reads JSONL rows created by scripts/prepare_data.py.

    Each row contains image_path (relative to project root), points [[x,y],...],
    count, and image_id. Training uses 256x256 random crops; validation/test use
    complete 448x448 images.
    """
    def __init__(self, manifest_path, project_root=".", split="train", crop_size=256,
                 sigma=4.0, stride=8, support_threshold=1e-3):
        self.project_root = Path(project_root)
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = self.project_root / self.manifest_path
        self.rows = [json.loads(line) for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.split = split
        self.transform = TrainTransform(crop_size) if split == "train" else EvalTransform()
        self.sigma = float(sigma)
        self.stride = int(stride)
        self.support_threshold = float(support_threshold)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.project_root / row["image_path"]
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        points = row["points"]
        image, points = self.transform(image, points)
        w, h = image.size
        density, seg_mask = make_density_targets(
            points, height=h, width=w, sigma=self.sigma,
            stride=self.stride, support_threshold=self.support_threshold,
        )
        image_tensor = image_to_tensor(image)
        density_tensor = torch.from_numpy(density[None]).float()
        seg_tensor = torch.from_numpy(seg_mask[None]).float()
        count = torch.tensor(float(len(points)), dtype=torch.float32)
        return {
            "image": image_tensor,
            "density": density_tensor,
            "seg_mask": seg_tensor,
            # Both masks use the same GT density-support definition.
            "structural_mask": seg_tensor.clone(),
            "sample_key": f"{image_path.parent.name}/{image_path.name}",
            "count": count,
            "tier": density_tier(len(points)),
            "image_id": row["image_id"],
        }
