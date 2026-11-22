"""
Live Cap: Adapter for real-time WebSocket webcam streaming.
Handles incoming webcam frames, injects network compression if requested,
invokes the Core Box, and returns the super-resolved frame.
"""

from __future__ import annotations

import base64
from typing import Dict, Any, Tuple
import cv2
import numpy as np

from ..core.box import SuperResolutionBox


class LiveCap:
    """Adapter for interactive webcam streaming."""

    def __init__(self, box: SuperResolutionBox) -> None:
        self.box = box

    def process_frame(
        self,
        frame_bytes: bytes,
        add_compression: bool = False,
        compression_quality: int = 30,
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """Process a single frame from the live stream."""
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "Invalid frame data"}

        # 1. Apply network compression if toggled
        if add_compression:
            bgr_processed = self.box.apply_network_compression(
                bgr, quality=compression_quality
            )
        else:
            bgr_processed = bgr

        # 2. Core Box inference
        sr_bgr, latency_ms = self.box.upscale_frame(bgr_processed)

        # 3. Encode modified frame
        ok, buffer_sr = cv2.imencode(
            ".jpg", sr_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
        )
        sr_b64 = (
            f"data:image/jpeg;base64,{base64.b64encode(buffer_sr).decode('utf-8')}"
            if ok
            else None
        )

        raw_b64 = None
        if return_raw:
            ok_raw, buffer_raw = cv2.imencode(
                ".jpg", bgr_processed, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if ok_raw:
                raw_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_raw).decode('utf-8')}"

        return {
            "sr_image": sr_b64,
            "raw_image": raw_b64,
            "latency_ms": round(latency_ms, 2),
            "fps": round(1000.0 / max(latency_ms, 0.01), 1),
            "in_res": f"{bgr.shape[1]}×{bgr.shape[0]}",
            "out_res": f"{sr_bgr.shape[1]}×{sr_bgr.shape[0]}",
            "compression_applied": add_compression,
        }
