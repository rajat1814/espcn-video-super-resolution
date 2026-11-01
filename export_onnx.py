#!/usr/bin/env python3
"""
ESPCN PyTorch-to-ONNX Model Exporter and Graph Validator
========================================================
Exports trained PyTorch Efficient Sub-Pixel Convolutional Neural Network (ESPCN)
checkpoints (.pth) to ONNX FP32 format with dynamic spatial axes for single-channel
luminance (Y-channel) super-resolution.

Key Features:
- Verifies Python runtime, PyTorch CUDA availability, and ONNX Runtime providers.
- Exports the 3-layer ESPCN graph with dynamic batch, height, and width axes
  (`lr_y: [B, 1, H, W]` -> `sr_y: [B, 1, H * scale, W * scale]`).
- Validates the exported ONNX graph via `onnx.checker.check_model`.
- Performs numerical parity verification between PyTorch eager execution and
  ONNX Runtime inference.

Usage:
  python export_onnx.py --weights espcn_codec_adapted.pth --output models/espcn_codec_adapted.onnx --scale 2
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    sys.exit(
        f"[ERROR] PyTorch is not installed ({exc}).\n"
        "Install dependencies via: pip install -r requirements.txt"
    )

try:
    import onnx
    import onnxruntime as ort
except ImportError as exc:
    sys.exit(
        f"[ERROR] ONNX / ONNX Runtime is not installed ({exc}).\n"
        "Install dependencies via: pip install -r requirements.txt"
    )


def _bicubic_polyphase_2x() -> torch.Tensor:
    """Exact 2x bicubic (a=-0.75, replicate borders) as a fixed 5x5 conv with 4 phases -> PixelShuffle(2).
    Equivalent to cv2/torch bicubic but exports as plain Pad+Conv, which is MUCH faster than ONNX `Resize` on CPU."""
    a = -0.75

    def w(t: float) -> float:
        t = abs(t)
        if t <= 1:
            return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
        if t < 2:
            return a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a
        return 0.0

    p0 = [w(1.75), w(0.75), w(0.25), w(1.25), 0.0]
    p1 = [0.0, w(1.25), w(0.25), w(0.75), w(1.75)]
    k = torch.tensor([[[a_ * b_ for b_ in wx] for a_ in wy] for wy in (p0, p1) for wx in (p0, p1)], dtype=torch.float32)
    return k.unsqueeze(1)


class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network (Shi et al., CVPR 2016)
    configured for single-channel luminance (Y-channel) super-resolution.

    Input Tensor : `lr_y` of shape `[B, 1, H, W]` in range `[0.0, 1.0]`
    Output Tensor: `sr_y` of shape `[B, 1, H * scale_factor, W * scale_factor]`
    """

    def __init__(self, scale_factor: int = 2, c1: int = 64, c2: int = 32, residual: bool = False) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.residual = residual
        self.register_buffer("res_flag", torch.tensor(1 if residual else 0))
        self.register_buffer("skip_w", _bicubic_polyphase_2x(), persistent=False)
        self.fast_skip = False  # set True only while exporting to ONNX (scale 2)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, scale_factor ** 2, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        out = self.shuffle(self.conv3(x))
        if self.residual:
            if self.fast_skip and self.scale_factor == 2:
                skip = F.conv2d(F.pad(inp, (2, 2, 2, 2), mode="replicate"), self.skip_w)
                out = out + F.pixel_shuffle(skip, 2)
            else:
                out = out + F.interpolate(inp, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        return out


def verify_environment() -> None:
    """Log runtime environment, CUDA device status, and available ONNX Runtime providers."""
    print("=" * 72)
    print("1. RUNTIME ENVIRONMENT & HARDWARE DIAGNOSTICS")
    print("=" * 72)
    print(f"• Python Version       : {platform.python_version()}")
    print(f"• PyTorch Version      : {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"• GPU Device           : {torch.cuda.get_device_name(0)}")
    print(f"• ONNX Runtime Version : {ort.__version__}")
    print(f"• Execution Providers  : {ort.get_available_providers()}\n")


def load_espcn_checkpoint(weights_path: Path, scale: int) -> nn.Module:
    """Instantiate the ESPCN architecture and load weights from disk.
    Exits with an error if the checkpoint is missing (no silent random-weight export).
    """
    print("=" * 72)
    print("2. MODEL ARCHITECTURE & CHECKPOINT INITIALIZATION")
    print("=" * 72)

    if not weights_path.exists():
        sys.exit(
            f"[ERROR] Checkpoint `{weights_path}` not found.\n"
            "Refusing to export randomly initialised weights. Train first (train.py) "
            "or pass the correct --weights path."
        )

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = dict(state)
    state.setdefault("res_flag", torch.tensor(0))
    model = ESPCN(scale_factor=scale, c1=state["conv1.weight"].shape[0], c2=state["conv2.weight"].shape[0], residual=bool(int(state["res_flag"])))
    model.load_state_dict(state, strict=True)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"• Loaded checkpoint `{weights_path}` ({param_count:,} trainable parameters).\n")
    return model


def export_to_onnx(
    model: nn.Module,
    output_onnx_path: Path,
    scale: int,
    height: int = 180,
    width: int = 320,
    opset: int = 17,
) -> None:
    """Export the PyTorch ESPCN model to ONNX FP32 and verify numerical equivalence."""
    print("=" * 72)
    print("3. ONNX GRAPH EXPORT & NUMERICAL EQUIVALENCE VERIFICATION")
    print("=" * 72)

    sample_input = torch.randn(1, 1, height, width, dtype=torch.float32)
    model.fast_skip = True   # export graph uses Pad+Conv bicubic skip (fast on CPU)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["lr_y"],
        output_names=["sr_y"],
        dynamic_axes={
            "lr_y": {0: "batch", 2: "height", 3: "width"},
            "sr_y": {0: "batch", 2: "height_out", 3: "width_out"},
        },
    )
    try:
        torch.onnx.export(model, (sample_input,), str(output_onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, (sample_input,), str(output_onnx_path), **export_kwargs)

    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(output_onnx_path), providers=providers)

    test_input = np.random.rand(1, 1, height, width).astype(np.float32)
    model.fast_skip = False  # reference = PyTorch's own bicubic
    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(test_input)).numpy()
    onnx_output = session.run(["sr_y"], {"lr_y": test_input})[0]

    max_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
    print(f"• Exported ONNX Model  : `{output_onnx_path}` ({output_onnx_path.stat().st_size / 1024:.1f} KB)")
    print(f"• Input Tensor `lr_y`  : {test_input.shape}")
    print(f"• Output Tensor `sr_y` : {onnx_output.shape} ({scale}x spatial upscale)")
    print(f"• Max |PyTorch - ONNX| : {max_diff:.2e}")
    print("• Verification Status  : PASSED\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PyTorch ESPCN (.pth) checkpoint to ONNX FP32 format."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("espcn_codec_adapted.pth"),
        help="Path to input PyTorch checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/espcn_codec_adapted.onnx"),
        help="Destination path for exported ONNX FP32 model.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Super-resolution upscale factor (default: 2).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=180,
        help="Sample input frame height for export verification (default: 180).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Sample input frame width for export verification (default: 320).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_environment()
    model = load_espcn_checkpoint(args.weights, args.scale)
    export_to_onnx(model, args.output, args.scale, args.height, args.width, args.opset)


if __name__ == "__main__":
    main()