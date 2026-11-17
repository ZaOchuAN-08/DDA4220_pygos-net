"""Point-aware augmentations. Density maps are rebuilt after geometric transforms."""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from PIL import Image


class TrainTransform:
    def __init__(self, crop_size: int = 256, horizontal_flip_p: float = 0.5):
        self.crop_size = int(crop_size)
        self.horizontal_flip_p = float(horizontal_flip_p)

    def __call__(self, image: Image.Image, points: Sequence[Sequence[float]]):
        width, height = image.size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(f"Image {image.size} is smaller than crop {self.crop_size}")
        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))

        cropped = []
        for x, y in points:
            x, y = float(x) - left, float(y) - top
            if 0 <= x < self.crop_size and 0 <= y < self.crop_size:
                cropped.append((x, y))

        if random.random() < self.horizontal_flip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            cropped = [(self.crop_size - x, y) for x, y in cropped]
        return image, cropped


class EvalTransform:
    """Evaluation keeps the original full-resolution image and points."""
    def __call__(self, image: Image.Image, points: Sequence[Sequence[float]]):
        return image, list(points)


def image_to_tensor(image: Image.Image):
    """RGB PIL image -> ImageNet-normalized float tensor [3,H,W]."""
    import torch
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).float()
