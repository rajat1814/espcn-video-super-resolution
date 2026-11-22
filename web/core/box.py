"""
The Box: The Pure Low-Res In -> High-Res Out Super-Resolution Engine.
=====================================================================
Single responsibility:
Takes a BGR image frame [H, W, 3], upscales luminance (Y-channel) using
the pre-trained ONNX ESPCN model, upscales chroma (Cr, Cb) bicubically,
and returns the reconstructed high-resolution BGR frame [2H, 2W, 3].
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    ort = None


class SuperResolutionBox:
    """The central inference box for Y-channel ESPCN super-resolution."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        scale: int = 2,
        prefer_gpu: bool = True,
        num_threads: int = 4,
    ) -> None:
        self.scale = scale
        self.num_threads = num_threads
        self.model_path = self._resolve_model_path(model_path)
        
        self.available_providers: List[str] = (
            ort.get_available_providers() if ort else ["CPUExecutionProvider"]
        )
        # Probe genuine hardware support instead of relying on provider list
        self.has_cuda: bool = self._probe_provider("CUDAExecutionProvider")
        self.has_dml: bool = self._probe_provider("DmlExecutionProvider")

        self.session: Optional[ort.InferenceSession] = None
        self.active_provider: str = "CPUExecutionProvider"
        self.input_name: str = "lr_y"
        self.output_name: str = "sr_y"
        self.expect_fp16: bool = False
        self.status_message: str = "Initializing..."

        self._init_session(prefer_gpu=prefer_gpu)

    def _probe_provider(self, provider_name: str) -> bool:
        """Verify if a provider is genuinely functional on this hardware."""
        if ort is None or provider_name not in self.available_providers:
            return False
        if provider_name == "CUDAExecutionProvider":
            # If no NVIDIA driver or CUDA toolkit is installed, skip without throwing DLL warnings
            if shutil.which("nvidia-smi") is None and "CUDA_PATH" not in os.environ:
                return False
        if provider_name == "DmlExecutionProvider":
            if os.name != "nt":
                return False
        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            s = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=[provider_name])
            return s.get_providers()[0] == provider_name
        except Exception:
            return False

    def _resolve_model_path(self, model_path: Optional[str | Path]) -> Path:
        """Find the ONNX model file on disk across common relative directories."""
        if model_path and Path(model_path).is_file():
            return Path(model_path)

        candidates = [
            Path("models/espcn_codec_res_v3_fp16.onnx"),
            Path("models/espcn_codec_res_v3.onnx"),
            Path("../models/espcn_codec_res_v3_fp16.onnx"),
            Path("../models/espcn_codec_res_v3.onnx"),
            Path("models/espcn_codec_adapted.onnx"),
            Path("../models/espcn_codec_adapted.onnx"),
        ]

        for p in candidates:
            if p.is_file():
                return p.resolve()

        # Search recursively in workspace if not directly found
        search_roots = [Path("."), Path("..")]
        for root in search_roots:
            found = list(root.glob("**/espcn_*.onnx"))
            if found:
                # Prefer res_v3 models
                res_models = [f for f in found if "res_v3" in f.name]
                return (res_models[0] if res_models else found[0]).resolve()

        raise FileNotFoundError(
            "Could not locate an ESPCN ONNX model file. Looked in models/."
        )

    def _init_session(self, prefer_gpu: bool = True) -> None:
        """Initialize the ONNX Runtime InferenceSession with best available hardware."""
        if ort is None:
            self.status_message = "Error: onnxruntime is not installed."
            return

        providers = []
        if prefer_gpu and self.has_cuda:
            providers.append("CUDAExecutionProvider")
        if prefer_gpu and self.has_dml:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        # Deduplicate
        providers = list(dict.fromkeys(providers))

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self.num_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=sess_opts, providers=providers
            )
            self.active_provider = self.session.get_providers()[0]
            self.status_message = f"Active Provider: {self.active_provider}"
        except Exception as err:
            # Fall back to CPU if GPU provider fails (e.g. missing cuDNN)
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self.active_provider = "CPUExecutionProvider"
            self.status_message = f"Fell back to CPU ({err})"

        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()[0]
        self.input_name = self.input_meta.name
        self.output_name = self.output_meta.name
        self.expect_fp16 = "float16" in self.input_meta.type

    def switch_provider(self, provider: str) -> Tuple[bool, str]:
        """Switch hardware provider between 'gpu' / 'cpu' at runtime."""
        if not self.session:
            return False, "Session not initialized"

        target_providers = []
        if provider.lower() in ("gpu", "cuda"):
            if not self.has_cuda and not self.has_dml:
                return False, "No GPU provider (CUDA/DirectML) detected on this machine"
            if self.has_cuda:
                target_providers.append("CUDAExecutionProvider")
            if self.has_dml:
                target_providers.append("DmlExecutionProvider")
            target_providers.append("CPUExecutionProvider")
        else:
            target_providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self.num_threads
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=sess_opts, providers=target_providers
            )
            self.active_provider = self.session.get_providers()[0]
            self.status_message = f"Switched to {self.active_provider}"
            return True, self.active_provider
        except Exception as e:
            return False, str(e)

    def upscale_frame(self, bgr_lr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Pure Box Operation: Low-Res BGR In -> High-Res BGR Out.
        Returns: (sr_bgr, latency_ms)
        """
        if self.session is None:
            # Fallback to bicubic if session failed
            t0 = time.perf_counter()
            sr = self.bicubic_upscale(bgr_lr)
            return sr, (time.perf_counter() - t0) * 1000.0

        h_lr, w_lr = bgr_lr.shape[:2]
        h_hr, w_hr = h_lr * self.scale, w_lr * self.scale

        t0 = time.perf_counter()

        # 1. Color decomposition to YCrCb
        ycrcb_lr = cv2.cvtColor(bgr_lr, cv2.COLOR_BGR2YCrCb)
        y_lr, cr_lr, cb_lr = cv2.split(ycrcb_lr)

        # 2. Normalize Y channel
        dtype = np.float16 if self.expect_fp16 else np.float32
        y_norm = (y_lr.astype(dtype)) * (1.0 / 255.0)
        input_tensor = np.expand_dims(np.expand_dims(y_norm, axis=0), axis=0)

        # 3. ONNX Model Inference on Y
        ort_out = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        y_sr = np.squeeze(ort_out, axis=(0, 1)).astype(np.float32)
        y_hr = np.clip(y_sr * 255.0, 0.0, 255.0).astype(np.uint8)

        # 4. Bicubic resize for Cr & Cb
        h_out, w_out = y_hr.shape[:2]
        cr_hr = cv2.resize(cr_lr, (w_out, h_out), interpolation=cv2.INTER_CUBIC)
        cb_hr = cv2.resize(cb_lr, (w_out, h_out), interpolation=cv2.INTER_CUBIC)

        # 5. Recombine to BGR
        ycrcb_hr = cv2.merge([y_hr, cr_hr, cb_hr])
        sr_bgr = cv2.cvtColor(ycrcb_hr, cv2.COLOR_YCrCb2BGR)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return sr_bgr, latency_ms

    def bicubic_upscale(self, bgr_lr: np.ndarray) -> np.ndarray:
        """Standard 2x bicubic baseline for comparison."""
        h_lr, w_lr = bgr_lr.shape[:2]
        return cv2.resize(
            bgr_lr, (w_lr * self.scale, h_lr * self.scale), interpolation=cv2.INTER_CUBIC
        )

    @staticmethod
    def apply_network_compression(
        bgr_image: np.ndarray, quality: int = 30
    ) -> np.ndarray:
        """
        Simulate lossy network/codec streaming compression.
        Compacts the frame using JPEG compression (quality 1-100, default 30)
        to reproduce DCT blocking and ringing artifacts.
        """
        quality = max(5, min(95, quality))
        ok, encoded = cv2.imencode(
            ".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return bgr_image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded if decoded is not None else bgr_image

    def get_info(self) -> Dict[str, Any]:
        """Return operational metadata about the active model and hardware."""
        return {
            "model_name": self.model_path.name,
            "model_path": str(self.model_path),
            "model_size_kb": round(os.path.getsize(self.model_path) / 1024, 2)
            if self.model_path.exists()
            else 0,
            "scale": self.scale,
            "active_provider": self.active_provider,
            "available_providers": self.available_providers,
            "precision": "FP16" if self.expect_fp16 else "FP32",
            "has_cuda": self.has_cuda,
            "has_dml": self.has_dml,
            "status": self.status_message,
        }
