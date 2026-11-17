
import torch
import torch.nn as nn

from .structural import MaskedStructuralLoss
from .segmentation import WeightedSegmentationLoss
from .background import BackgroundSuppressionLoss


class CombinedLoss(nn.Module):
    """
    L_total = SL* + lambda1 * L_seg + lambda2 * L_bg
    """

    def __init__(
        self,
        lambda_seg: float = 0.1,
        lambda_bg: float = 0.05,
        num_scales: int = 3,
        ssim_data_range: float = 1.0,
        foreground_weight: float = 1.0,
        background_weight: float = 1.0,
        use_hard_negative: bool = True,
    ):
        super().__init__()

        if lambda_seg < 0 or lambda_bg < 0:
            raise ValueError("Loss coefficients must be non-negative.")

        self.lambda_seg = lambda_seg
        self.lambda_bg = lambda_bg
        # Compatibility switch: current implementation remains background MSE,
        # not the proposal's Top-K hard-negative objective.
        self.use_hard_negative = bool(use_hard_negative)

        self.structural_loss = MaskedStructuralLoss(
            num_scales=num_scales,
            data_range=ssim_data_range,
        )

        self.segmentation_loss = WeightedSegmentationLoss(
            foreground_weight=foreground_weight,
            background_weight=background_weight,
        )

        self.background_loss = BackgroundSuppressionLoss()

    def forward(
        self,
        outputs: dict,
        batch: dict,
    ) -> dict:

        required_outputs = ("density", "segmentation_logits")
        required_batch = (
            "density",
            "structural_mask",
            "seg_mask",
        )

        for key in required_outputs:
            if key not in outputs:
                raise KeyError(f"Missing model output: {key}")

        for key in required_batch:
            if key not in batch:
                raise KeyError(f"Missing batch field: {key}")

        density_pred = outputs["density"]
        segmentation_logits = outputs["segmentation_logits"]

        density_gt = batch["density"]
        structural_mask = batch["structural_mask"]
        seg_mask = batch["seg_mask"]

        sl = self.structural_loss(
            density_pred,
            density_gt,
            structural_mask,
        )

        seg = self.segmentation_loss(
            segmentation_logits,
            seg_mask,
        )

        bg = (self.background_loss(density_pred, structural_mask)
              if self.use_hard_negative else density_pred.sum() * 0.0)

        total = (
            sl
            + self.lambda_seg * seg
            + self.lambda_bg * bg
        )

        return {
            "total": total,
            "structural": sl,
            "segmentation": seg,
            "background": bg,
        }
