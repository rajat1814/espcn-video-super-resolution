"""
FastAPI Server for Codec-Adapted ESPCN Super-Resolution.
Integrates the Core Box and Caps, exposes REST endpoints, Server-Sent Events,
and WebSockets for real-time webcam feeds.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .core.box import SuperResolutionBox
from .caps.photo_cap import PhotoCap
from .caps.video_cap import VideoCap
from .caps.live_cap import LiveCap
from .caps.benchmark_cap import BenchmarkCap

# Initialize Application
app = FastAPI(title="Codec-Adapted Video Super-Resolution Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate Core Box and Adapters
box = SuperResolutionBox()
photo_cap = PhotoCap(box)
video_cap = VideoCap(box, storage_dir=BASE_DIR / "temp_videos")
live_cap = LiveCap(box)
benchmark_cap = BenchmarkCap(box)

# Event Bus for Server-Sent Events (SSE)
event_listeners: List[asyncio.Queue] = []


def emit_event(event_type: str, message: str, level: str = "info", data: dict = None):
    """Broadcast an event to all connected UI clients."""
    payload = {
        "type": event_type,
        "message": message,
        "level": level,  # info, success, warning, error
        "timestamp": time.strftime("%H:%M:%S"),
        "data": data or {},
    }
    for q in list(event_listeners):
        try:
            q.put_nowait(payload)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Retrieve current server, model, and hardware status."""
    info = box.get_info()
    return JSONResponse(info)


@app.post("/api/provider")
async def switch_hardware(provider: str = Form(...)):
    """Switch hardware execution between CPU and GPU/CUDA."""
    success, result = box.switch_provider(provider)
    if success:
        emit_event("provider_change", f"Switched execution provider to {result}", "success")
        return {"success": True, "active_provider": result}
    emit_event("provider_error", f"Provider switch failed: {result}", "warning")
    return {"success": False, "error": result}


@app.post("/api/photo/upscale")
async def upscale_photo(
    file: UploadFile = File(...),
    add_compression: bool = Form(False),
    compression_quality: int = Form(30),
):
    """Upscale an uploaded photo and return raw, bicubic, and super-resolved images."""
    try:
        content = await file.read()
        res = photo_cap.process(
            image_bytes=content,
            add_compression=add_compression,
            compression_quality=compression_quality,
        )
        emit_event(
            "photo_processed",
            f"Photo upscaled {res['input_resolution']} -> {res['output_resolution']} in {res['latency_ms']} ms ({res['active_provider']})",
            "success",
        )
        return JSONResponse(res)
    except Exception as exc:
        emit_event("photo_error", f"Photo processing failed: {exc}", "error")
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/video/upload")
async def upload_video(
    file: UploadFile = File(...),
    add_compression: bool = Form(False),
    compression_quality: int = Form(30),
):
    """Upload a video and start background processing."""
    try:
        temp_input = TEMP_DIR / f"upload_{int(time.time())}_{file.filename}"
        with open(temp_input, "wb") as f:
            shutil.copyfileobj(file.file, f)

        task_id = video_cap.start_video_processing(
            input_file_path=temp_input,
            add_compression=add_compression,
            compression_quality=compression_quality,
            event_callback=lambda evt: emit_event(
                evt["type"],
                f"Video {evt.get('task_id')}: {evt.get('progress', 0)}% complete",
                "info",
                evt,
            ),
        )

        emit_event("video_started", f"Video processing started (Task ID: {task_id})", "info")
        return {"task_id": task_id, "status": "processing"}
    except Exception as exc:
        emit_event("video_error", f"Video upload error: {exc}", "error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/video/task/{task_id}")
async def get_video_task(task_id: str):
    """Check progress of a running video task."""
    task = video_cap.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(task)


@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    """Download an upscaled video file."""
    path = video_cap.storage_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.post("/api/benchmark")
async def run_benchmark():
    """Run hardware benchmark across CPU and GPU/NPU providers."""
    emit_event("benchmark_start", "Starting hardware inference benchmark...", "info")
    loop = asyncio.get_event_loop()
    # Run CPU-bound benchmark in threadpool
    results = await loop.run_in_executor(None, benchmark_cap.run_benchmark)
    emit_event("benchmark_complete", "Hardware benchmark completed", "success", results)
    return JSONResponse(results)


# ---------------------------------------------------------------------------
# Server-Sent Events (SSE) for Real-Time Event & Status Updates
# ---------------------------------------------------------------------------

@app.get("/api/events")
async def event_stream():
    """Real-time event stream providing server telemetry and status notifications."""
    queue = asyncio.Queue()
    event_listeners.append(queue)

    async def event_generator():
        try:
            # Initial hello heartbeat
            info = box.get_info()
            init_msg = {
                "type": "connected",
                "message": f"Server connected ({info['active_provider']}, {info['model_name']})",
                "level": "success",
                "timestamp": time.strftime("%H:%M:%S"),
                "data": info,
            }
            yield f"data: {json.dumps(init_msg)}\n\n"

            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in event_listeners:
                event_listeners.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# WebSocket for Live Webcam Feed
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def live_feed_websocket(websocket: WebSocket):
    """Duplex WebSocket connection for real-time webcam frames."""
    await websocket.accept()
    emit_event("webcam_connect", "Live webcam client connected", "info")
    try:
        while True:
            # Receive frame payload (bytes or JSON with settings)
            message = await websocket.receive()
            if "bytes" in message:
                frame_bytes = message["bytes"]
                add_compression = False
                quality = 30
            elif "text" in message:
                try:
                    payload = json.loads(message["text"])
                    b64_data = payload.get("frame", "")
                    if "," in b64_data:
                        b64_data = b64_data.split(",")[1]
                    import base64
                    frame_bytes = base64.b64decode(b64_data)
                    add_compression = bool(payload.get("add_compression", False))
                    quality = int(payload.get("quality", 30))
                except Exception:
                    continue
            else:
                continue

            # Process frame through LiveCap
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                live_cap.process_frame,
                frame_bytes,
                add_compression,
                quality,
                True,  # return raw frame for comparison
            )

            await websocket.send_text(json.dumps(res))
    except WebSocketDisconnect:
        emit_event("webcam_disconnect", "Live webcam client disconnected", "info")
    except Exception as exc:
        emit_event("webcam_error", f"Webcam stream error: {exc}", "error")


# ---------------------------------------------------------------------------
# Static Assets & Fallback
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_index():
    """Serve the single-page application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Web UI is being configured...</h1>")
