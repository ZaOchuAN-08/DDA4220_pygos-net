"""Shared two-head model; hard-negative loss belongs to the loss package."""
import torch
from torch import nn
from .backbone import VGG19Encoder
from .dcb import DilatedContextBlock


def feature_dampening(probability, threshold=0.5, background_factor=0.1):
    # Faithful hard gate: density gradients do not flow through the comparison.
    return torch.where(probability >= threshold, torch.ones_like(probability),
                       torch.full_like(probability, background_factor))


def _head(nonnegative=False):
    layers = [nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
              nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
              nn.Conv2d(128, 1, 1)]
    if nonnegative:
        layers.append(nn.ReLU())
    head = nn.Sequential(*layers)
    for layer in head.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
    return head


class PygosNet(nn.Module):
    def __init__(self, use_dcb: bool = True, pretrained: bool = True,
                 dcb_branch_channels: int = 128):
        super().__init__()
        self.encoder = VGG19Encoder(pretrained=pretrained)
        self.context = (DilatedContextBlock(branch_channels=dcb_branch_channels)
                        if use_dcb else nn.Identity())
        self.segmentation_head = _head()
        self.density_head = _head(nonnegative=True)
        self.use_dcb = use_dcb

    def forward(self, image):
        features = self.context(self.encoder(image))
        logits = self.segmentation_head(features)
        probability = logits.sigmoid()
        density = self.density_head(features * feature_dampening(probability))
        # Always return unmasked density, including in eval mode, for loss consistency.
        return {"density": density, "segmentation": probability,
                "segmentation_logits": logits}

    @torch.no_grad()
    def predict(self, image):
        if self.training:
            raise RuntimeError("Call model.eval() before model.predict().")
        result = self(image)
        masked = result["density"] * (result["segmentation"] >= 0.5)
        return {**result, "density_masked": masked,
                "count": masked.sum(dim=(1, 2, 3))}
