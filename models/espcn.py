"""ESPCN Super-Resolution Architecture (Y-Channel).
Default: original 3-layer ESPCN (64/32). Options: wider channels, and `residual=True` which adds a
bicubic-upsampled copy of the input to the network output (global residual learning), so the network
only learns a correction and mild/clean inputs are not over-smoothed."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPCN(nn.Module):
    def __init__(self, scale_factor: int = 2, c1: int = 64, c2: int = 32, residual: bool = False) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.residual = residual
        self.register_buffer("res_flag", torch.tensor(1 if residual else 0))
        self.conv1 = nn.Conv2d(1, c1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, scale_factor ** 2, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.shuffle(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))
        if self.residual:
            out = out + F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        return out


def load_espcn(path, map_location="cpu") -> "ESPCN":
    """Build the right ESPCN from a checkpoint (widths, scale and residual flag inferred from the weights)."""
    sd = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = dict(sd)
    sd.setdefault("res_flag", torch.tensor(0))          # older checkpoints have no flag
    c1, c2 = sd["conv1.weight"].shape[0], sd["conv2.weight"].shape[0]
    scale = int(round(sd["conv3.weight"].shape[0] ** 0.5))
    model = ESPCN(scale, c1, c2, residual=bool(int(sd["res_flag"])))
    model.load_state_dict(sd)
    return model.eval()
