"""Point annotations -> normalized Gaussian density maps.

The high-resolution density map is normalized so that each annotated point
contributes exactly one unit of mass. It is then sum-pooled to stride 8.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def points_to_density(
    points: Sequence[Sequence[float]],
    height: int = 448,
    width: int = 448,
    sigma: float = 4.0,
    truncate: float = 3.0,
) -> np.ndarray:
    """Render a [H,W] density map whose sum equals len(points).

    Points are (x, y) coordinates in pixel coordinates. Coordinates represent
    pixel-space centers; a point is retained when 0 <= x < W and 0 <= y < H.
    Each Gaussian is truncated at truncate*sigma and renormalized after
    clipping at image boundaries, preserving one count per valid point.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    density = np.zeros((height, width), dtype=np.float32)
    radius = max(1, int(math.ceil(truncate * sigma)))
    denom = 2.0 * sigma * sigma

    for point in points:
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x < width and 0.0 <= y < height):
            continue
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        kernel = np.exp(-((xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2) / denom)
        mass = float(kernel.sum())
        if mass > 0:
            density[y0:y1, x0:x1] += kernel / mass
    return density


def sum_pool_density(density: np.ndarray, stride: int = 8) -> np.ndarray:
    """Sum-pool [H,W] density to [H/stride,W/stride], preserving mass."""
    if density.ndim != 2:
        raise ValueError(f"Expected 2D density, got shape {density.shape}")
    h, w = density.shape
    if h % stride or w % stride:
        raise ValueError(f"H and W must be divisible by stride={stride}; got {(h, w)}")
    return density.reshape(h // stride, stride, w // stride, stride).sum(axis=(1, 3)).astype(np.float32)


def make_density_targets(
    points: Sequence[Sequence[float]],
    height: int = 448,
    width: int = 448,
    sigma: float = 4.0,
    stride: int = 8,
    support_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (stride-8 density [H/8,W/8], support mask [H/8,W/8])."""
    high_res = points_to_density(points, height, width, sigma)
    low_res = sum_pool_density(high_res, stride)
    support = (low_res > support_threshold).astype(np.float32)
    return low_res, support
