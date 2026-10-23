#!/usr/bin/env python3
"""
ONNX Mixed-Precision (FP16) Graph Converter and Latency Profiler
================================================================
Converts exported FP32 ONNX super-resolution models to half-precision (FP16)
using `onnxconverter-common` and benchmarks single-frame inference latency
and throughput across available ONNX Runtime execution providers.

Key Features:
- Converts internal graph weights and activations to IEEE 754 float16 while
  retaining float32 I/O interfaces (`keep_io_types=True`) for zero-copy
  integration with standard OpenCV processing pipelines.
- Validates numerical stability (checking for NaN/Inf divergence against FP32).
- Profiles mean, median, and 95th-percentile (p95) inference latency.

Usage:
  python onnx_fp16.py --input models/espcn_codec_adapted.onnx --output models/espcn_codec_adapted_fp16.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError as exc:
    sys.exit(
        f"[ERROR] ONNX / ONNX Runtime is not installed ({exc}).\n"
        "Install dependencies via: pip install -r requirements.txt"
    )

try:
    from onnxconverter_common import float16
except ImportError as exc:
    sys.exit(
        f"[ERROR] `onnxconverter-common` is not installed ({exc}).\n"
        "Install dependencies via: pip install onnxconverter-common"
    )


def convert_onnx_to_fp16(
    input_onnx_path: Path,
    output_fp16_path: Path,
    keep_io_types: bool = True,
) -> None:
    """Convert an ONNX FP32 model graph to FP16 precision."""
    print("=" * 72)
    print("1. ONNX GRAPH MIXED-PRECISION CONVERSION (FP32 -> FP16)")
    print("=" * 72)

    if not input_onnx_path.exists():
        sys.exit(
            f"[ERROR] Source ONNX model `{input_onnx_path}` not found.\n"
            "Ensure `models/espcn_codec_adapted.onnx` exists or run `export_onnx.py` first."
        )

    model_fp32 = onnx.load(str(input_onnx_path))
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=keep_io_types,
    )

    output_fp16_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(output_fp16_path))

    fp32_kb = input_onnx_path.stat().st_size / 1024.0
    fp16_kb = output_fp16_path.stat().st_size / 1024.0
    reduction_pct = (1.0 - (fp16_kb / fp32_kb)) * 100.0 if fp32_kb > 0 else 0.0

    print(f"• Source FP32 Model    : `{input_onnx_path}` ({fp32_kb:.1f} KB)")
    print(f"• Target FP16 Model    : `{output_fp16_path}` ({fp16_kb:.1f} KB, -{reduction_pct:.1f}% size)")
    print(f"• Retain float32 I/O   : {keep_io_types}\n")


def profile_fp16_inference(
    fp32_path: Path,
    fp16_path: Path,
    height: int = 180,
    width: int = 320,
    warmup_runs: int = 15,
    timed_runs: int = 50,
) -> None:
    """Validate numerical accuracy and benchmark inference throughput on hardware."""
    print("=" * 72)
    print("2. NUMERICAL VALIDATION & HARDWARE LATENCY PROFILING")
    print("=" * 72)

    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_fp32 = ort.InferenceSession(str(fp32_path), sess_options=sess_options, providers=providers)
    sess_fp16 = ort.InferenceSession(str(fp16_path), sess_options=sess_options, providers=providers)

    active_provider = sess_fp16.get_providers()[0]
    input_name = sess_fp16.get_inputs()[0].name
    output_name = sess_fp16.get_outputs()[0].name
    input_dtype = sess_fp16.get_inputs()[0].type

    np_dtype = np.float16 if "float16" in input_dtype else np.float32
    test_y_frame = np.random.rand(1, 1, height, width).astype(np_dtype)

    out_fp32 = sess_fp32.run([output_name], {input_name: test_y_frame.astype(np.float32)})[0]
    out_fp16 = sess_fp16.run([output_name], {input_name: test_y_frame})[0]

    if np.isnan(out_fp16).any() or np.isinf(out_fp16).any():
        sys.exit("[ERROR] FP16 inference produced NaN or Inf values.")

    max_abs_diff = float(np.max(np.abs(out_fp32.astype(np.float32) - out_fp16.astype(np.float32))))
    mean_abs_diff = float(np.mean(np.abs(out_fp32.astype(np.float32) - out_fp16.astype(np.float32))))

    for _ in range(warmup_runs):
        sess_fp16.run([output_name], {input_name: test_y_frame})

    latencies_ms: list[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        sess_fp16.run([output_name], {input_name: test_y_frame})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"• Active Execution Prov: {active_provider}")
    print(f"• Input Tensor Shape   : {test_y_frame.shape} ({test_y_frame.dtype})")
    print(f"• Output Tensor Shape  : {out_fp16.shape} ({out_fp16.dtype})")
    print(f"• FP32 vs FP16 Error   : mean={mean_abs_diff:.2e}, max={max_abs_diff:.2e}")
    print(f"• Latency ({timed_runs} cycles) : median={median_ms:.2f} ms, mean={avg_ms:.2f} ms, p95={p95_ms:.2f} ms")
    print(f"• Inference Throughput : {fps:.1f} FPS\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ONNX FP32 model to FP16 and profile inference latency."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("models/espcn_codec_adapted.onnx"),
        help="Path to source ONNX FP32 model (default: models/espcn_codec_adapted.onnx).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/espcn_codec_adapted_fp16.onnx"),
        help="Path to destination ONNX FP16 model (default: models/espcn_codec_adapted_fp16.onnx).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=180,
        help="Input frame height for latency profiling (default: 180).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Input frame width for latency profiling (default: 320).",
    )
    parser.add_argument(
        "--native-fp16-io",
        action="store_true",
        help="Convert model input/output nodes to float16 instead of keeping float32 I/O.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_onnx_to_fp16(
        input_onnx_path=args.input,
        output_fp16_path=args.output,
        keep_io_types=not args.native_fp16_io,
    )
    profile_fp16_inference(
        fp32_path=args.input,
        fp16_path=args.output,
        height=args.height,
        width=args.width,
    )


if __name__ == "__main__":
    main()