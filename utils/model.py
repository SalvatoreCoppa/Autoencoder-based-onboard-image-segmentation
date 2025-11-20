import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from utils.decoder import Decoder
from utils.aspp import ASPPLight

class DeepLabV3PlusWithMobileNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # === Backbone MobileNetV3 ===
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
        return_layers = {
            "0": "skip1",   # 16@112x112
            "4": "skip2",   # 40@28x28
            "7": "skip3",   # 80@14x14
            "16": "out"     # 960@7x7
        }
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # === Proiezioni 1x1 per le skip connections (tutte a 64 canali) ===
        self.skip1_proj = nn.Conv2d(16, 64, kernel_size=1)
        self.skip2_proj = nn.Conv2d(40, 64, kernel_size=1)
        self.skip3_proj = nn.Conv2d(80, 64, kernel_size=1)

        # === ASPP light ===
        self.aspp = ASPPLight(in_channels=960, out_channels=256)

        # === Decoder ===
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)

        # Estrai e proietta gli skip
        skip1 = self.skip1_proj(features["skip1"])  # 64@112x112
        skip2 = self.skip2_proj(features["skip2"])  # 64@28x28
        skip3 = self.skip3_proj(features["skip3"])  # 64@14x14
        x = features["out"]                         # 256@7x7

        x = self.aspp(x)

        # Decoder

        out = self.decoder(x, skips=[skip1, skip2, skip3])  # [B, num_classes, H, W]

        return out

