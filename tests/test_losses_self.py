
import torch

from pygosnet.losses import (
    MaskedStructuralLoss,
    WeightedSegmentationLoss,
    BackgroundSuppressionLoss,
    CombinedLoss,
)


def make_fake_batch(batch_size=2, height=32, width=32):
    """
    Generate synthetic density maps and masks.
    No real dataset or model is required.
    """
    torch.manual_seed(42)

    # Coordinates of the stride-8 output grid.
    y = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)

    density_gt = torch.zeros(
        batch_size, 1, height, width
    )

    # Add several synthetic Gaussian peaks to each image.
    centers = [
        [(8, 8), (12, 13), (24, 23)],
        [(5, 25), (16, 16)],
    ]

    sigma = 1.5

    for b in range(batch_size):
        for cy, cx in centers[b % len(centers)]:
            gaussian = torch.exp(
                -((x - cx) ** 2 + (y - cy) ** 2)
                / (2 * sigma**2)
            )
            density_gt[b, 0] += gaussian[0, 0]

    # Follow the proposed support threshold.
    # Both masks are the same label under this definition.
    structural_mask = (density_gt > 1e-3).float()
    seg_mask = structural_mask.clone()

    return {
        "density": density_gt,
        "structural_mask": structural_mask,
        "seg_mask": seg_mask,
    }


def make_fake_outputs(batch_size=2, height=32, width=32):
    """Generate synthetic model outputs."""

    density_pred = (
        torch.rand(batch_size, 1, height, width) * 0.2
    ).requires_grad_(True)

    segmentation_logits = torch.randn(
        batch_size, 1, height, width
    ).requires_grad_(True)

    # B's segmentation output is a probability, not logits.
    segmentation_prob = torch.sigmoid(segmentation_logits)

    return {
        "density": density_pred,
        "segmentation": segmentation_prob,
        "segmentation_logits": segmentation_logits,
    }


def test_individual_losses():
    batch = make_fake_batch()
    outputs = make_fake_outputs()

    sl = MaskedStructuralLoss()(
        outputs["density"],
        batch["density"],
        batch["structural_mask"],
    )

    seg = WeightedSegmentationLoss()(
        outputs["segmentation_logits"],
        batch["seg_mask"],
    )

    bg = BackgroundSuppressionLoss()(
        outputs["density"],
        batch["structural_mask"],
    )

    for name, value in [
        ("structural", sl),
        ("segmentation", seg),
        ("background", bg),
    ]:
        assert value.ndim == 0, f"{name} is not a scalar"
        assert torch.isfinite(value), f"{name} is NaN or Inf"
        print(f"{name} loss: {value.item():.6f}")


def test_combined_loss_and_backward():
    batch = make_fake_batch()
    outputs = make_fake_outputs()

    criterion = CombinedLoss(
        lambda_seg=0.1,
        lambda_bg=0.05,
        num_scales=3,
    )

    losses = criterion(outputs, batch)

    expected_total = (
        losses["structural"]
        + 0.1 * losses["segmentation"]
        + 0.05 * losses["background"]
    )

    assert torch.allclose(
        losses["total"], expected_total
    ), "Total loss does not match the formula"

    assert torch.isfinite(losses["total"])

    print("\nCombined losses:")
    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")

    # Simulate training: compute gradients.
    losses["total"].backward()

    assert outputs["density"].grad is not None
    assert outputs["segmentation_logits"].grad is not None

    assert torch.isfinite(outputs["density"].grad).all()
    assert torch.isfinite(
        outputs["segmentation_logits"].grad
    ).all()

    print("\nBackward pass: PASS")


def test_masks_are_identical():
    batch = make_fake_batch()

    assert torch.equal(
        batch["seg_mask"],
        batch["structural_mask"],
    )

    print("seg_mask == structural_mask: PASS")


def test_background_edge_cases():
    loss_fn = BackgroundSuppressionLoss()

    # Case 1: no background cells.
    pred1 = torch.rand(
        1, 1, 8, 8, requires_grad=True
    )
    mask1 = torch.ones_like(pred1)

    loss1 = loss_fn(pred1, mask1)
    assert loss1.item() == 0.0

    # Case 2: every cell is background.
    pred2 = torch.ones(
        1, 1, 8, 8, requires_grad=True
    )
    mask2 = torch.zeros_like(pred2)

    loss2 = loss_fn(pred2, mask2)
    assert torch.isclose(loss2, torch.tensor(1.0))

    # Confirm gradients also work in both cases.
    (loss1 + loss2).backward()

    print("Background edge cases: PASS")


if __name__ == "__main__":
    test_individual_losses()
    test_combined_loss_and_backward()
    test_masks_are_identical()
    test_background_edge_cases()

    print("\nAll mock-input tests passed!")
