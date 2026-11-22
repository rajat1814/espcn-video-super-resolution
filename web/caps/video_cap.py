"""
Video Cap: Adapter for video file processing.
Extracts frames sequentially, optionally applies network compression,
passes each frame through the Core Box, and re-encodes to MP4.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import cv2

from ..core.box import SuperResolutionBox


class VideoCap:
    """Adapter for processing and upscaling video files."""

    def __init__(self, box: SuperResolutionBox, storage_dir: Optional[Path] = None) -> None:
        self.box = box
        self.storage_dir = storage_dir or Path("web/temp_videos")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.active_tasks.get(task_id)

    def start_video_processing(
        self,
        input_file_path: Path,
        add_compression: bool = False,
        compression_quality: int = 30,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """Launch background video processing thread and return a task ID."""
        task_id = str(uuid.uuid4())[:8]
        output_filename = f"sr_{task_id}.mp4"
        output_file_path = self.storage_dir / output_filename

        self.active_tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "output_path": str(output_file_path),
            "output_url": f"/api/video/download/{output_filename}",
            "error": None,
        }

        # Run in dedicated OS background thread for non-blocking CPU execution
        t = threading.Thread(
            target=self._process_video_job,
            args=(
                task_id,
                input_file_path,
                output_file_path,
                add_compression,
                compression_quality,
                event_callback,
            ),
            daemon=True,
        )
        t.start()
        return task_id

    def _process_video_job(
        self,
        task_id: str,
        input_path: Path,
        output_path: Path,
        add_compression: bool,
        compression_quality: int,
        event_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        """Worker executing frame-by-frame super-resolution."""
        task = self.active_tasks[task_id]
        task["status"] = "processing"

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            task["status"] = "failed"
            task["error"] = "Failed to open video file"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_out = w_in * self.box.scale
        h_out = h_in * self.box.scale

        task["total_frames"] = total_frames

        # FourCC for MP4 (avc1 or mp4v)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (w_out, h_out))

        if not writer.isOpened():
            # Try alternate fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (w_out, h_out))

        frame_idx = 0
        total_time = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if add_compression:
                    frame = self.box.apply_network_compression(frame, quality=compression_quality)

                sr_frame, latency = self.box.upscale_frame(frame)
                writer.write(sr_frame)

                frame_idx += 1
                total_time += latency
                avg_fps = (frame_idx / (total_time / 1000.0)) if total_time > 0 else 0.0
                pct = int((frame_idx / max(total_frames, 1)) * 100)

                task["current_frame"] = frame_idx
                task["progress"] = min(pct, 100)
                task["fps"] = round(avg_fps, 1)
                task["latency_ms"] = round(latency, 2)

                if event_callback and frame_idx % 15 == 0:
                    event_callback(
                        {
                            "type": "video_progress",
                            "task_id": task_id,
                            "progress": pct,
                            "frame": frame_idx,
                            "total": total_frames,
                        }
                    )

            task["status"] = "completed"
            task["progress"] = 100
            if event_callback:
                event_callback(
                    {
                        "type": "video_completed",
                        "task_id": task_id,
                        "output_url": task["output_url"],
                    }
                )
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = str(exc)
        finally:
            cap.release()
            writer.release()
            # Clean up source temporary file
            try:
                if input_path.exists():
                    os.remove(input_path)
            except Exception:
                pass
