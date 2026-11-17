"""Parallel dilated context, preserving spatial shape."""
from torch import cat, nn


class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int = 512, branch_channels: int = 128):
        super().__init__()
        if channels <= 0 or branch_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, branch_channels, 3,
                                    padding=rate, dilation=rate), nn.ReLU())
            for rate in (1, 2, 4)
        ])
        self.project = nn.Sequential(nn.Conv2d(3 * branch_channels, channels, 1), nn.ReLU())
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        return self.project(cat([branch(features) for branch in self.branches], dim=1))
