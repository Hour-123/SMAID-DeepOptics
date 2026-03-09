import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, low_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(low_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class MediumDualHeadNet(nn.Module):
    """Compact U-Net style network for RGB and depth reconstruction."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels)
        self.down2 = Down(base_channels, base_channels)
        self.down3 = Down(base_channels, base_channels)
        self.bottleneck = DoubleConv(base_channels, base_channels)

        self.up1 = Up(base_channels, base_channels, base_channels)
        self.up2 = Up(base_channels, base_channels, base_channels)
        self.up3 = Up(base_channels, base_channels, base_channels)

        self.rgb_head = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.depth_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        rgb = torch.sigmoid(self.rgb_head(x))
        depth = torch.sigmoid(self.depth_head(x))
        return rgb, depth


def my_medium_model(in_channels: int = 1, base_channels: int = 64) -> MediumDualHeadNet:
    return MediumDualHeadNet(in_channels=in_channels, base_channels=base_channels)
