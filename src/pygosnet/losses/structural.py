
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 7,
) -> torch.Tensor:
    """Return 1 - mean SSIM, using differentiable local statistics."""

    if pred.shape != target.shape:
        raise ValueError("SSIM inputs must have identical shapes.")

    if pred.ndim != 4:
        raise ValueError("Expected tensors shaped [B, C, H, W].")

    h, w = pred.shape[-2:]
    smallest_dim = min(h, w)

    # Use an odd window that fits the current pyramid level.
    window = min(window_size, smallest_dim)
    if window % 2 == 0:
        window -= 1

    if window < 1:
        raise ValueError("SSIM input spatial dimensions cannot be empty.")

    padding = window // 2

    def local_mean(x):
        return F.avg_pool2d(
            x,
            kernel_size=window,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )

    mu_x = local_mean(pred)
    mu_y = local_mean(target)

    sigma_x2 = (
        local_mean(pred * pred) - mu_x * mu_x
    ).clamp_min(0.0)

    sigma_y2 = (
        local_mean(target * target) - mu_y * mu_y
    ).clamp_min(0.0)

    sigma_xy = local_mean(pred * target) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (
        (2.0 * mu_x * mu_y + c1)
        * (2.0 * sigma_xy + c2)
    )

    denominator = (
        (mu_x.square() + mu_y.square() + c1)
        * (sigma_x2 + sigma_y2 + c2)
    ).clamp_min(1e-12)

    ssim = numerator / denominator
    return 1.0 - ssim.mean()


class MaskedStructuralLoss(nn.Module):
    """
    SL* = (1/K) sum_k [1 - SSIM(
        Pool_k(D_pred * S_gt),
        Pool_k(D_target * S_gt)
    )]

    Pool_k uses average pooling with factor 2**(k-1).
    """

    def __init__(
        self,
        num_scales: int = 3,
        data_range: float = 1.0,
        window_size: int = 7,
    ):
        super().__init__()

        if num_scales < 1:
            raise ValueError("num_scales must be >= 1.")
        if data_range <= 0:
            raise ValueError("data_range must be positive.")

        self.num_scales = num_scales
        self.data_range = data_range
        self.window_size = window_size

    def forward(
        self,
        density_pred: torch.Tensor,
        density_gt: torch.Tensor,
        structural_mask: torch.Tensor,
    ) -> torch.Tensor:

        if density_pred.shape != density_gt.shape:
            raise ValueError("Predicted and target density shapes differ.")

        if structural_mask.shape != density_pred.shape:
            raise ValueError("Structural mask shape does not match density.")

        if density_pred.ndim != 4:
            raise ValueError("Expected density maps shaped [B, 1, H, W].")

        mask = structural_mask.to(
            device=density_pred.device,
            dtype=density_pred.dtype,
        )

        pred_masked = density_pred * mask
        target_masked = density_gt * mask

        scale_losses = []

        for scale_idx in range(self.num_scales):
            factor = 2 ** scale_idx

            if factor > 1:
                # Keep both maps at the same spatial resolution.
                pred_scale = F.avg_pool2d(
                    pred_masked,
                    kernel_size=factor,
                    stride=factor,
                )
                target_scale = F.avg_pool2d(
                    target_masked,
                    kernel_size=factor,
                    stride=factor,
                )
            else:
                pred_scale = pred_masked
                target_scale = target_masked

            scale_losses.append(
                _ssim_loss(
                    pred_scale,
                    target_scale,
                    data_range=self.data_range,
                    window_size=self.window_size,
                )
            )

        return torch.stack(scale_losses).mean()
