"""End-to-end integration tests for the Pygos-Net training pipeline.

Run from the repository root:
    python -m pytest tests/test_pipeline_integration.py -v

These tests use synthetic tensors only; they do not require the real penguin dataset.
They exercise the real PygosNet and CombinedLoss APIs, then perform one optimizer step.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pygosnet.models.pygosnet import PygosNet
from pygosnet.losses.combined import CombinedLoss


# Small input keeps the CPU test relatively quick. The model's encoder returns
# stride-8 features, so 32x32 inputs should produce 4x4 output maps.
IMAGE_SIZE = 32
BATCH_SIZE = 2


class SyntheticPenguinDataset(Dataset):
    """Mimics A's current Dataset contract: image, density, seg_mask, count, etc."""

    def __init__(self, n: int = 4, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)

        # Create nonnegative stride-8 density maps with a few positive locations.
        self.density = torch.zeros(n, 1, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
        for i in range(n):
            self.density[i, 0, (i + 1) % 4, (i + 2) % 4] = 1.0
            if i % 2 == 0:
                self.density[i, 0, 1, 1] = 0.5

        # A supplies one support mask; no structural_mask field is required.
        self.seg_mask = (self.density > 1e-3).float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        density = self.density[index]
        return {
            "image": self.images[index],
            "density": density,
            "seg_mask": self.seg_mask[index],
            "count": density.sum(),
            "tier": "L1",
            "image_id": f"synthetic_{index}",
        }


def _assert_finite_scalar(value: torch.Tensor | float, name: str) -> None:
    tensor = torch.as_tensor(value)
    assert tensor.numel() == 1, f"{name} should be scalar, got {tuple(tensor.shape)}"
    assert torch.isfinite(tensor).all(), f"{name} is NaN or Inf"


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def model(device: torch.device) -> PygosNet:
    # pretrained=False avoids downloading weights in CI or on a new machine.
    # DCB is enabled here so the integration test covers the proposed model path.
    torch.manual_seed(123)
    return PygosNet(use_dcb=True, pretrained=False).to(device)


@pytest.fixture()
def batch(device: torch.device) -> dict:
    loader = DataLoader(SyntheticPenguinDataset(), batch_size=BATCH_SIZE, shuffle=False)
    result = next(iter(loader))
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in result.items()
    }


def test_synthetic_dataset_contract(batch: dict) -> None:
    assert set(("image", "density", "seg_mask")).issubset(batch.keys())
    assert "structural_mask" not in batch, (
        "The agreed Dataset contract uses seg_mask as the shared support mask."
    )
    assert batch["image"].shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["density"].shape == (BATCH_SIZE, 1, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
    assert batch["seg_mask"].shape == batch["density"].shape
    assert torch.isfinite(batch["image"]).all()
    assert torch.isfinite(batch["density"]).all()
    assert set(torch.unique(batch["seg_mask"]).tolist()).issubset({0.0, 1.0})


def test_model_forward_contract(model: PygosNet, batch: dict) -> None:
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image"])

    assert {"density", "segmentation", "segmentation_logits"}.issubset(outputs)
    expected_shape = batch["density"].shape
    for key in ("density", "segmentation", "segmentation_logits"):
        assert outputs[key].shape == expected_shape, (
            f"{key}: expected {expected_shape}, got {outputs[key].shape}"
        )
        assert torch.isfinite(outputs[key]).all(), f"{key} contains NaN/Inf"

    assert (outputs["density"] >= 0).all(), "Density output must be nonnegative"
    assert ((outputs["segmentation"] >= 0) & (outputs["segmentation"] <= 1)).all()
    assert torch.allclose(
        outputs["segmentation"],
        outputs["segmentation_logits"].sigmoid(),
        atol=1e-6,
    )


def test_combined_loss_backward_and_optimizer_step(
    model: PygosNet, batch: dict
) -> None:
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["image"])
    losses = criterion(outputs, batch)

    assert isinstance(losses, dict), (
        "CombinedLoss should return a dict containing at least the total loss."
    )
    assert "total" in losses, f"Missing 'total' in loss keys: {list(losses)}"
    for key, value in losses.items():
        if torch.is_tensor(value) or isinstance(value, (int, float)):
            _assert_finite_scalar(value, f"loss[{key}]")

    total = losses["total"]
    assert torch.is_tensor(total) and total.requires_grad
    total.backward()

    grad_tensors = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert grad_tensors, "No model parameters received gradients"
    assert all(torch.isfinite(grad).all() for grad in grad_tensors), "Non-finite gradients"
    assert any(torch.count_nonzero(grad).item() > 0 for grad in grad_tensors), (
        "All gradients are zero; check loss wiring and model outputs."
    )

    before = next(model.parameters()).detach().clone()
    optimizer.step()
    after = next(model.parameters()).detach()
    assert not torch.equal(before, after), "Optimizer step did not update model parameters"


def test_predict_requires_eval_mode(model: PygosNet, batch: dict) -> None:
    model.train()
    with pytest.raises(RuntimeError, match=r"eval"):
        model.predict(batch["image"])


def test_predict_returns_masked_density_and_count(
    model: PygosNet, batch: dict
) -> None:
    model.eval()
    with torch.no_grad():
        result = model.predict(batch["image"])

    assert {"density", "segmentation", "segmentation_logits",
            "density_masked", "count"}.issubset(result)
    assert result["density_masked"].shape == result["density"].shape
    assert result["count"].shape == (BATCH_SIZE,)
    assert torch.isfinite(result["count"]).all()
    assert torch.allclose(
        result["count"],
        result["density_masked"].sum(dim=(1, 2, 3)),
        atol=1e-6,
    )


def test_empty_target_batch_loss_is_finite(device: torch.device) -> None:
    """Exercise the all-background edge case required by the proposal."""
    torch.manual_seed(321)
    model = PygosNet(use_dcb=False, pretrained=False).to(device)
    criterion = CombinedLoss()
    images = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    batch = {
        "image": images,
        "density": torch.zeros(2, 1, IMAGE_SIZE // 8, IMAGE_SIZE // 8, device=device),
        "seg_mask": torch.zeros(2, 1, IMAGE_SIZE // 8, IMAGE_SIZE // 8, device=device),
    }
    outputs = model(images)
    losses = criterion(outputs, batch)
    _assert_finite_scalar(losses["total"], "empty-target total loss")
    losses["total"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_loss_rejects_or_exposes_shape_mismatch(device: torch.device) -> None:
    """A deliberately wrong target shape must not silently pass as valid."""
    model = PygosNet(use_dcb=False, pretrained=False).to(device)
    criterion = CombinedLoss()
    images = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    outputs = model(images)
    bad_batch = {
        "density": torch.zeros(2, 1, 3, 3, device=device),
        "seg_mask": torch.zeros(2, 1, 3, 3, device=device),
    }
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        criterion(outputs, bad_batch)
