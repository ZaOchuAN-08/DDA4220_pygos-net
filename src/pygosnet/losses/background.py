
import torch
import torch.nn as nn


class BackgroundSuppressionLoss(nn.Module):
    """Mean squared predicted density on ground-truth background cells."""

    def forward(
        self,
        density_pred: torch.Tensor,
        structural_mask: torch.Tensor,
    ) -> torch.Tensor:

        if density_pred.shape != structural_mask.shape:
            raise ValueError("Density and structural mask shapes differ.")

        # M_bg = {l_m | S_gt(l_m) = 0}
        bg_mask = structural_mask <= 0

        if not bg_mask.any():
            # Differentiable zero if there are no background cells.
            return density_pred.sum() * 0.0

        bg_density = density_pred[bg_mask]

        # L_bg = mean(D_pred(l_m) ** 2), over background cells only.
        return bg_density.square().mean()
