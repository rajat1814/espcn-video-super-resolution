#!/usr/bin/env python3
"""
Real-Time Video Super-Resolution & Artifact Suppression Engine (OpenCV + ONNX)
==============================================================================
Captures live webcam feeds or compressed video streams, performs single-channel
luminance (Y-channel) super-resolution in YCrCb color space using ONNX Runtime,
and renders a split-screen comparison against standard bicubic interpolation.

Architecture & Pipeline:
1. Color Space Decomposition:
   - Converts incoming BGR frames to YCrCb (`cv2.COLOR_BGR2YCrCb`).
   - Extracts the luminance channel (`Y`) normalized to `[0.0, 1.0]` with tensor
     dimensions `[1, 1, H, W]` for neural super-resolution.
   - Upscales chrominance channels (`Cr`, `Cb`) via bicubic interpolation
     (`cv2.INTER_CUBIC`) and reconstructs the high-resolution BGR output.
2. Hardware Execution & Live Provider Switching:
   - Supports both NVIDIA CUDA (`CUDAExecutionProvider`) and multi-threaded CPU
     (`CPUExecutionProvider`) execution with automatic hardware fallback.
   - Interactive hotkeys allow real-time switching between interpolation modes
     and hardware execution providers without restarting the stream.
3. Diagnostics & Recording:
   - Includes a real-time EMA-smoothed FPS and frame latency HUD overlay, MP4
     stream recording (`--record-video`), and automated stability testing
     (`--stress-seconds`).

Usage:
  # Launch live webcam stream (auto-detects GPU or CPU):
  python live_stream.py --model models/espcn_codec_adapted.onnx --source 0

  # Launch using synthetic test stream and record MP4 output:
  python live_stream.py --model models/espcn_codec_adapted.onnx --source synthetic --record-video results/demo_backup.mp4
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    sys.exit(
        f"[ERROR] `onnxruntime` is not installed ({exc}).\n"
        "Install dependencies via: pip install -r requirements.txt"
    )


class ONNXSuperResolver:
    """ONNX Runtime inference engine for Y-channel (`[1, 1, H, W]`) super-resolution."""

    def __init__(
        self,
        model_path: Path,
        scale: int = 2,
        device: str = "auto",
        num_threads: int = 4,
    ) -> None:
        self.model_path = model_path
        self.scale = scale
        self.num_threads = num_threads
        self.available_providers = ort.get_available_providers()
        self.has_cuda = "CUDAExecutionProvider" in self.available_providers

        self.session: ort.InferenceSession
        self.active_provider: str = "CPUExecutionProvider"
        self.input_name: str = "lr_y"
        self.output_name: str = "sr_y"
        self.expect_fp16: bool = False

        self.set_device(device)

    def set_device(self, device: str) -> str:
        """Configure the ONNX Runtime session for 'gpu', 'cpu', or 'auto' execution."""
        device = device.lower()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "gpu":
            if self.has_cuda:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print(
                    "• [INFO] CUDAExecutionProvider not detected on this system; "
                    "using CPUExecutionProvider."
                )
                providers = ["CPUExecutionProvider"]
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.has_cuda
                else ["CPUExecutionProvider"]
            )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self.num_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=sess_opts, providers=providers
            )
        except Exception as err:
            print(f"• [WARNING] GPU initialization failed ({err}); falling back to CPUExecutionProvider.")
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )

        self.active_provider = self.session.get_providers()[0]
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()[0]
        self.input_name = self.input_meta.name
        self.output_name = self.output_meta.name
        self.expect_fp16 = "float16" in self.input_meta.type
        return self.active_provider

    def toggle_gpu_cpu(self) -> str:
        """Switch active execution provider between CUDA GPU and CPU at runtime."""
        if self.active_provider == "CUDAExecutionProvider":
            return self.set_device("cpu")
        return self.set_device("gpu")

    def upscale_y_channel(self, y_channel_uint8: np.ndarray) -> np.ndarray:
        """Super-resolve a 2D `uint8` luminance array `[H, W]` -> `[H*scale, W*scale]`."""
        dtype = np.float16 if self.expect_fp16 else np.float32
        y_norm = (y_channel_uint8.astype(dtype)) * (1.0 / 255.0)
        input_tensor = np.expand_dims(np.expand_dims(y_norm, axis=0), axis=0)

        ort_out = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        y_sr = np.squeeze(ort_out, axis=(0, 1)).astype(np.float32)
        y_sr_uint8 = np.clip(y_sr * 255.0, 0.0, 255.0).astype(np.uint8)
        return y_sr_uint8


def process_frame_ycrcb(
    bgr_lr: np.ndarray,
    engine: Optional[ONNXSuperResolver],
    scale: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a bicubic baseline BGR frame and an ONNX super-resolved BGR frame."""
    h_lr, w_lr = bgr_lr.shape[:2]
    h_hr, w_hr = h_lr * scale, w_lr * scale

    bicubic_bgr = cv2.resize(bgr_lr, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    if engine is None:
        return bicubic_bgr, bicubic_bgr.copy()

    ycrcb_lr = cv2.cvtColor(bgr_lr, cv2.COLOR_BGR2YCrCb)
    y_lr, cr_lr, cb_lr = cv2.split(ycrcb_lr)

    y_hr = engine.upscale_y_channel(y_lr)
    h_out, w_out = y_hr.shape[:2]
    if (h_out, w_out) != (h_hr, w_hr):
        bicubic_bgr = cv2.resize(bgr_lr, (w_out, h_out), interpolation=cv2.INTER_CUBIC)

    cr_hr = cv2.resize(cr_lr, (w_out, h_out), interpolation=cv2.INTER_CUBIC)
    cb_hr = cv2.resize(cb_lr, (w_out, h_out), interpolation=cv2.INTER_CUBIC)

    ycrcb_hr = cv2.merge([y_hr, cr_hr, cb_hr])
    sr_bgr = cv2.cvtColor(ycrcb_hr, cv2.COLOR_YCrCb2BGR)
    return bicubic_bgr, sr_bgr


def generate_synthetic_test_frame(
    frame_idx: int, height: int = 180, width: int = 320
) -> np.ndarray:
    """Synthesize a low-resolution talking-head test frame with JPEG Q=30 compression
    artifacts for camera-free pipeline verification.
    """
    canvas = np.full((height, width, 3), (38, 44, 52), dtype=np.uint8)
    cx = width // 2 + int(10 * np.sin(frame_idx * 0.08))
    cy = height // 2 - 8 + int(4 * np.cos(frame_idx * 0.12))

    cv2.ellipse(canvas, (cx, height + 8), (75, 55), 0, 180, 360, (90, 110, 140), -1)
    cv2.circle(canvas, (cx, cy), 38, (180, 205, 235), -1)
    cv2.circle(canvas, (cx - 13, cy - 6), 4, (30, 30, 30), -1)
    cv2.circle(canvas, (cx + 13, cy - 6), 4, (30, 30, 30), -1)

    mouth_h = 3 + abs(int(6 * np.sin(frame_idx * 0.25)))
    cv2.ellipse(canvas, (cx, cy + 13), (11, mouth_h), 0, 0, 360, (60, 50, 140), -1)

    cv2.line(canvas, (15, 40), (95, height - 30), (0, 215, 255), 2)
    cv2.rectangle(canvas, (width - 95, 42), (width - 18, 98), (80, 220, 100), 2)
    cv2.putText(
        canvas,
        f"FRAME #{frame_idx:04d}",
        (14, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    ok, enc = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    if ok:
        canvas = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return canvas


def draw_hud_overlay(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    fps: float,
    latency_ms: float,
    mode_label: str,
    provider_label: str,
    is_recording: bool = False,
) -> np.ndarray:
    """Concatenate baseline and super-resolved frames horizontally and render telemetry HUD."""
    cv2.line(right_bgr, (0, 0), (0, right_bgr.shape[0]), (0, 255, 180), 2)
    combined = cv2.hconcat([left_bgr, right_bgr])
    h_comb, w_comb = combined.shape[:2]
    half_w = w_comb // 2

    overlay = combined.copy()
    cv2.rectangle(overlay, (0, 0), (w_comb, 64), (18, 18, 24), -1)
    cv2.rectangle(overlay, (0, h_comb - 30), (w_comb, h_comb), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.76, combined, 0.24, 0, combined)

    cv2.putText(
        combined,
        "RAW DEGRADED / BICUBIC BASELINE",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (110, 200, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        f"Output Res: {left_bgr.shape[1]}x{left_bgr.shape[0]}",
        (14, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    fps_color = (80, 245, 120) if fps >= 25.0 else (0, 190, 255)
    rec_badge = "  [REC]" if is_recording else ""
    cv2.putText(
        combined,
        f"LIVE: {mode_label}{rec_badge}",
        (half_w + 14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 80, 255) if is_recording else fps_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        f"FPS: {fps:5.1f} | Latency: {latency_ms:5.2f} ms | {provider_label}",
        (half_w + 14, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        combined,
        "Keys: [1] Bicubic  [2] Secondary  [3] Primary ESPCN  [G] GPU/CPU Toggle  [S] Snap  [Q] Quit",
        (14, h_comb - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return combined


def resolve_default_model(requested: Path) -> Path:
    """Locate the primary ONNX model file across standard project paths."""
    if requested.exists():
        return requested
    candidates = [
        Path("models/espcn_codec_adapted_fp16.onnx"),
        Path("models/espcn_codec_adapted.onnx"),
        Path("dummy_fp16.onnx"),
        Path("dummy_fp32.onnx"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time OpenCV + ONNX Super-Resolution Streaming Engine."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/espcn_codec_adapted.onnx"),
        help="Primary ONNX model path (default: models/espcn_codec_adapted.onnx).",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=None,
        help="Optional secondary/baseline ONNX model path for hotkey [2].",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Execution device: 'auto' (prefers CUDA GPU if available), 'gpu', or 'cpu'.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for default webcam, video file path, or 'synthetic'.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Spatial upscaling factor (default: 2).",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=320,
        help="Low-resolution input width before upscaling (default: 320).",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=180,
        help="Low-resolution input height before upscaling (default: 180).",
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        help="Optional path to record the side-by-side output stream as an MP4 file.",
    )
    parser.add_argument(
        "--stress-seconds",
        type=float,
        default=0.0,
        help="Run continuous stability benchmark for N seconds and export CSV telemetry.",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="If > 0, process N frames in headless mode without opening a GUI window.",
    )
    parser.add_argument(
        "--save-sample",
        type=Path,
        default=None,
        help="Optional path to save a rendered side-by-side PNG frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_default_model(args.model)

    print(f"• Available ORT Providers    : {ort.get_available_providers()}")
    primary_engine: Optional[ONNXSuperResolver] = None
    baseline_engine: Optional[ONNXSuperResolver] = None

    if model_path.exists():
        primary_engine = ONNXSuperResolver(model_path, scale=args.scale, device=args.device)
        print(f"• Loaded primary ONNX model  : `{model_path}` ({primary_engine.active_provider})")
    else:
        print(f"• [WARNING] Model `{model_path}` not found; running in Bicubic-only mode.")

    if args.baseline_model and args.baseline_model.exists():
        baseline_engine = ONNXSuperResolver(args.baseline_model, scale=args.scale, device=args.device)
        print(f"• Loaded secondary ONNX model: `{args.baseline_model}` ({baseline_engine.active_provider})")
    else:
        baseline_engine = primary_engine

    use_synthetic = args.source.lower() == "synthetic"
    cap: Optional[cv2.VideoCapture] = None
    if not use_synthetic:
        src: int | str = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"• [INFO] Video source `{args.source}` unavailable; using synthetic test stream.")
            use_synthetic = True

    writer: Optional[cv2.VideoWriter] = None
    active_mode = 3 if primary_engine is not None else 1
    ema_fps = 0.0
    frame_idx = 0
    dropped_frames = 0
    latencies_ms: list[float] = []
    last_combined: Optional[np.ndarray] = None
    stream_start = time.perf_counter()

    print("• Starting real-time super-resolution stream...")
    try:
        while True:
            if args.headless_frames > 0 and frame_idx >= args.headless_frames:
                break
            if args.stress_seconds > 0 and (time.perf_counter() - stream_start) >= args.stress_seconds:
                break

            t_start = time.perf_counter()

            if use_synthetic:
                frame_lr = generate_synthetic_test_frame(
                    frame_idx, height=args.input_height, width=args.input_width
                )
            else:
                assert cap is not None
                ret, raw_frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = cap.read()
                    if not ret:
                        dropped_frames += 1
                        break
                frame_lr = cv2.resize(
                    raw_frame,
                    (args.input_width, args.input_height),
                    interpolation=cv2.INTER_AREA,
                )

            if active_mode == 1 or primary_engine is None:
                engine_to_use = None
                mode_label = "BICUBIC BASELINE"
                provider_label = "OpenCV CPU"
            elif active_mode == 2 and baseline_engine is not None:
                engine_to_use = baseline_engine
                mode_label = f"BASELINE ({baseline_engine.model_path.name})"
                provider_label = baseline_engine.active_provider
            else:
                engine_to_use = primary_engine
                mode_label = f"CODEC-ADAPTED ({primary_engine.model_path.name})"
                provider_label = primary_engine.active_provider

            left_bgr, right_bgr = process_frame_ycrcb(
                frame_lr, engine=engine_to_use, scale=args.scale
            )

            elapsed = time.perf_counter() - t_start
            latency_ms = elapsed * 1000.0
            latencies_ms.append(latency_ms)
            inst_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            ema_fps = inst_fps if frame_idx == 0 else (0.9 * ema_fps + 0.1 * inst_fps)

            combined = draw_hud_overlay(
                left_bgr=left_bgr,
                right_bgr=right_bgr,
                fps=ema_fps,
                latency_ms=latency_ms,
                mode_label=mode_label,
                provider_label=provider_label,
                is_recording=(args.record_video is not None),
            )
            last_combined = combined
            frame_idx += 1

            if args.record_video is not None:
                if writer is None:
                    args.record_video.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h_c, w_c = combined.shape[:2]
                    writer = cv2.VideoWriter(str(args.record_video), fourcc, 30.0, (w_c, h_c))
                writer.write(combined)

            if args.headless_frames == 0 and args.stress_seconds == 0.0:
                cv2.imshow("ESPCN Real-Time Super-Resolution Engine (Bicubic vs. ONNX)", combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("1"):
                    active_mode = 1
                elif key == ord("2"):
                    active_mode = 2
                elif key == ord("3"):
                    active_mode = 3
                elif key == ord("g"):
                    if primary_engine is not None:
                        new_prov = primary_engine.toggle_gpu_cpu()
                        if baseline_engine is not None and baseline_engine is not primary_engine:
                            baseline_engine.set_device(
                                "gpu" if new_prov == "CUDAExecutionProvider" else "cpu"
                            )
                        ema_fps = 0.0
                        print(f"• Active execution provider switched to: {new_prov}")
                elif key == ord("s"):
                    snap_name = f"snapshot_frame_{frame_idx:04d}.png"
                    cv2.imwrite(snap_name, combined)
                    print(f"• Saved frame snapshot to `{snap_name}`.")
    finally:
        if writer is not None:
            writer.release()
            print(f"• Saved recorded stream to `{args.record_video}`.")
        if cap is not None:
            cap.release()
        if args.headless_frames == 0 and args.stress_seconds == 0.0:
            cv2.destroyAllWindows()

    if args.save_sample is not None and last_combined is not None:
        args.save_sample.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_sample), last_combined)
        print(f"• Saved verification frame to `{args.save_sample}`.")

    if latencies_ms:
        mean_ms = float(np.mean(latencies_ms))
        median_ms = float(np.median(latencies_ms))
        p95_ms = float(np.percentile(latencies_ms, 95))
        avg_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        print(
            f"• Telemetry Summary: {frame_idx} frames | Dropped: {dropped_frames} | "
            f"Median: {median_ms:.2f} ms | Mean: {mean_ms:.2f} ms (p95: {p95_ms:.2f} ms) | "
            f"Throughput: {avg_fps:.1f} FPS"
        )

        if args.stress_seconds > 0:
            stress_csv = Path("results/pipeline_stress_test.csv")
            stress_csv.parent.mkdir(parents=True, exist_ok=True)
            with stress_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Duration_s", "Frames_Processed", "Dropped_Frames", "Median_Latency_ms", "Mean_Latency_ms", "P95_Latency_ms", "Mean_FPS"])
                w.writerow([round(args.stress_seconds, 1), frame_idx, dropped_frames, round(median_ms, 2), round(mean_ms, 2), round(p95_ms, 2), round(avg_fps, 2)])
            print(f"• Exported stability telemetry to `{stress_csv}`.")


if __name__ == "__main__":
    main()