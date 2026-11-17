"""VGG-19 feature encoder, without pool5 or classifier."""
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights
from torchvision.models.vgg import cfgs, make_layers


class VGG19Encoder(nn.Module):
    output_channels = 512
    output_stride = 8

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Construct features only: avoid allocating VGG's large unused classifier.
        self.features = make_layers(cfgs["E"][:-1])
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
        if pretrained:
            state = VGG19_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            features = {key.removeprefix("features."): value
                        for key, value in state.items() if key.startswith("features.")}
            self.features.load_state_dict(features, strict=True)
        self.pretrained = pretrained

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB input [B, 3, H, W].")
        if image.shape[-2] < 16 or image.shape[-1] < 16:
            raise ValueError("Input height and width must be at least 16.")
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("Input H and W must be divisible by 16; resize in the data pipeline.")
        return F.interpolate(self.features(image), scale_factor=2,
                             mode="bilinear", align_corners=True)
