import numpy as np
import torch
from PIL import Image

from pygosnet.data.density import points_to_density, sum_pool_density, make_density_targets
from pygosnet.data.dataset import density_tier
from pygosnet.data.transforms import TrainTransform


def test_density_mass_preserved():
    points = [(10.5, 20.5), (223.0, 223.0), (447.0, 447.0)]
    high = points_to_density(points, 448, 448, sigma=4)
    low = sum_pool_density(high, stride=8)
    assert np.isclose(high.sum(), 3.0, atol=1e-5)
    assert np.isclose(low.sum(), 3.0, atol=1e-5)


def test_empty_density():
    density, mask = make_density_targets([], 448, 448)
    assert density.shape == (56, 56)
    assert mask.shape == (56, 56)
    assert density.sum() == 0
    assert mask.sum() == 0


def test_density_tiers():
    assert [density_tier(x) for x in [0, 1, 100, 101, 500, 501, 1000, 1001]] == [
        "L0", "L1", "L1", "L2", "L2", "L3", "L3", "L4"
    ]


def test_crop_rebuild_points_in_bounds():
    transform = TrainTransform(crop_size=256, horizontal_flip_p=0.0)
    image = Image.new("RGB", (448, 448))
    # Seed is intentionally not fixed here; only geometric invariants are tested.
    _, points = transform(image, [(10, 10), (220, 220), (440, 440)])
    assert all(0 <= x < 256 and 0 <= y < 256 for x, y in points)
