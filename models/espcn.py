"""ESPCN Super-Resolution Architecture (Y-Channel)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPCN(nn.Module):
    def __init__(self, scale_factor: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, scale_factor ** 2, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.shuffle(self.conv3(x))
