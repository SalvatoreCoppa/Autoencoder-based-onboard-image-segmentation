import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPLight(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3_pool = nn.AdaptiveAvgPool2d(1)
        self.branch3_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.branch1(x)  # [B, 256, H, W]
        x2 = self.branch2(x)  # [B, 256, H, W]
        x3 = self.branch3_pool(x)  # [B, in_c, 1, 1]
        x3 = self.branch3_conv(x3)  # [B, 256, 1, 1]
        x3 = F.interpolate(x3, size=size, mode='bilinear', align_corners=False)  # [B, 256, H, W]

        x = torch.cat([x1, x2, x3], dim=1)  # [B, 256*3, H, W]
        return self.project(x)  # [B, 256, H, W]
