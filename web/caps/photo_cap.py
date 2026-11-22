"""
Photo Cap: Adapter for single image processing.
Handles decoding, optional network compression injection, Box execution,
and base64 encoding for browser display.
"""

from __future__ import annotations

import base64
from typing import Dict, Any, Tuple
import cv2
import numpy as np

from ..core.box import SuperResolutionBox


class PhotoCap:
    """Adapter for single photo / static image super-resolution."""

    def __init__(self, box: SuperResolutionBox) -> None:
        self.box = box

    @staticmethod
    def _bgr_to_base64_jpeg(bgr: np.ndarray, quality: int = 95) -> str:
        """Encode a BGR numpy array to a base64 JPEG data URI."""
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ValueError("Failed to encode image to JPEG")
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def process(
        self,
        image_bytes: bytes,
        add_compression: bool = False,
        compression_quality: int = 30,
        return_bicubic: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an uploaded photo.
        Returns JSON-serializable dictionary with raw and modified data URIs + stats.
        """
        # Decode raw image bytes
        np_arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode the provided image file.")

        h_orig, w_orig = bgr.shape[:2]

        # 1. Raw image (apply compression if enabled)
        if add_compression:
            raw_display = self.box.apply_network_compression(bgr, quality=compression_quality)
            compression_applied = True
        else:
            raw_display = bgr
            compression_applied = False

        # 2. Super-Resolve via Core Box
        sr_bgr, latency_ms = self.box.upscale_frame(raw_display)
        h_out, w_out = sr_bgr.shape[:2]

        # 3. Bicubic Baseline
        bicubic_b64 = None
        if return_bicubic:
            bicubic_bgr = self.box.bicubic_upscale(raw_display)
            bicubic_b64 = self._bgr_to_base64_jpeg(bicubic_bgr)

        # 4. Package response
        raw_b64 = self._bgr_to_base64_jpeg(raw_display)
        sr_b64 = self._bgr_to_base64_jpeg(sr_bgr)

        return {
            "success": True,
            "raw_image": raw_b64,
            "modified_image": sr_b64,
            "bicubic_image": bicubic_b64,
            "input_resolution": f"{w_orig}×{h_orig}",
            "output_resolution": f"{w_out}×{h_out}",
            "latency_ms": round(latency_ms, 2),
            "fps": round(1000.0 / max(latency_ms, 0.01), 1),
            "compression_applied": compression_applied,
            "compression_quality": compression_quality if compression_applied else None,
            "active_provider": self.box.active_provider,
        }
