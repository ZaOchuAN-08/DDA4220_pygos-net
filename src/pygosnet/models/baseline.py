"""Baseline shares every non-DCB module with PygosNet."""
from .pygosnet import PygosNet


class Baseline(PygosNet):
    def __init__(self, pretrained: bool = True):
        super().__init__(use_dcb=False, pretrained=pretrained)
