
import torch

from pygosnet.losses import CombinedLoss


def test_combined_loss_forward_backward():
    torch.manual_seed(42)

    density_pred = torch.rand(
        2, 1, 32, 32, requires_grad=True
    )
    seg_logits = torch.randn(
        2, 1, 32, 32, requires_grad=True
    )

    density_gt = torch.rand(2, 1, 32, 32)
    seg_mask = torch.randint(
        0, 2, (2, 1, 32, 32)
    ).float()

    structural_mask = (density_gt > 0).float()

    outputs = {
        "density": density_pred,
        "segmentation_logits": seg_logits,
        "segmentation": torch.sigmoid(seg_logits),
    }

    batch = {
        "density": density_gt,
        "structural_mask": structural_mask,
        "seg_mask": seg_mask,
    }

    criterion = CombinedLoss(num_scales=3)
    losses = criterion(outputs, batch)

    assert set(losses.keys()) == {
        "total",
        "structural",
        "segmentation",
        "background",
    }

    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value)

    losses["total"].backward()

    assert density_pred.grad is not None
    assert seg_logits.grad is not None
    assert torch.isfinite(density_pred.grad).all()
    assert torch.isfinite(seg_logits.grad).all()


def test_background_loss_is_zero_without_background():
    from pygosnet.losses import BackgroundSuppressionLoss

    pred = torch.rand(1, 1, 8, 8, requires_grad=True)
    mask = torch.ones_like(pred)

    loss = BackgroundSuppressionLoss()(pred, mask)

    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None


def test_background_loss_matches_formula():
    from pygosnet.losses import BackgroundSuppressionLoss

    pred = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]],
        requires_grad=True,
    )
    mask = torch.tensor(
        [[[[1.0, 0.0], [1.0, 0.0]]]]
    )

    loss = BackgroundSuppressionLoss()(pred, mask)

    # Background values are 2 and 4.
    expected = (2.0**2 + 4.0**2) / 2.0

    assert torch.isclose(loss, torch.tensor(expected))
