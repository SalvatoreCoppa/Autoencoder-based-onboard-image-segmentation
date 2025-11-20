import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.up1 = self._block(256, 128)          # ASPP → 14×14
        self.up2 = self._block(128 + 64, 64)      # + skip3 (64@14x14) → 28×28
        self.up3 = self._block(64 + 64, 64)       # + skip2 (64@28x28) → 56×56
        self.up4 = self._block(64 + 64, 32)       # + skip1 (64@112x112) → 112×112
        self.up5 = nn.Conv2d(32, num_classes, kernel_size=1)  # output finale

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skips):
        skip1, skip2, skip3 = skips  # Tutti a 64 canali

        x = self.up1(x)                        # 7×7 → 14×14
        x = torch.cat([x, skip3], dim=1)       # Concat skip3 → [B, 192, 14, 14]

        x = self.up2(x)                        # → 28×28
        x = torch.cat([x, skip2], dim=1)       # → [B, 128, 28, 28]

        x = self.up3(x)                        # → 56×56
        x = F.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)       # → [B, 128, 112, 112]

        x = self.up4(x)                        # → [B, 32, 112, 112]
        x = self.up5(x)                        # → [B, num_classes, 112, 112]

        return x
