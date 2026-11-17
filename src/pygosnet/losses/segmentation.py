
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSegmentationLoss(nn.Module):
    """Weighted pixel-wise binary cross-entropy."""

    def __init__(
        self,
        foreground_weight: float = 1.0,
        background_weight: float = 1.0,
    ):
        super().__init__()

        if foreground_weight <= 0 or background_weight <= 0:
            raise ValueError("Pixel weights must be positive.")

        self.foreground_weight = foreground_weight
        self.background_weight = background_weight

    def forward(
        self,
        segmentation_logits: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:

        if segmentation_logits.shape != seg_mask.shape:
            raise ValueError("Segmentation logits and target shapes differ.")

        target = seg_mask.to(
            device=segmentation_logits.device,
            dtype=segmentation_logits.dtype,
        )

        pixel_loss = F.binary_cross_entropy_with_logits(
            segmentation_logits,
            target,
            reduction="none",
        )

        pixel_weights = torch.where(
            target >= 0.5,
            torch.full_like(target, self.foreground_weight),
            torch.full_like(target, self.background_weight),
        )

        return (
            (pixel_loss * pixel_weights).sum()
            / pixel_weights.sum().clamp_min(1e-8)
        )
