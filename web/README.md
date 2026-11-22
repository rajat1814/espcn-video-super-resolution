# ESPCN Web Dashboard & Backend Service

An interactive local web application for demonstrating **Real-Time Video Super-Resolution & Codec Artifact Suppression** to teachers, peers, and evaluators.

---

## Architecture

This web application follows a clean **Single Responsibility & Adapter Pattern**:

* **The Core Box (`web/core/box.py`):**
  A pure, minimal neural inference engine. Takes a raw low-resolution frame `[H, W, 3]`, runs luminance super-resolution via the 21K-parameter ONNX model, upscales chrominance bicubically, and outputs the super-resolved frame `[2H, 2W, 3]`.
* **The Caps / Adapters (`web/caps/`):**
  * `photo_cap.py`: Image decoding, optional network compression injection, base64 response generation.
  * `video_cap.py`: Background video decoding, frame-by-frame super-resolution, progress tracking, and MP4 re-encoding.
  * `live_cap.py`: Low-latency WebSocket streaming for real-time webcam processing.
  * `benchmark_cap.py`: Hardware benchmarking across CPU threads and GPU (CUDA/DirectML).
* **The Frontend (`web/static/`):**
  A pure vanilla HTML5/CSS3/ES6 web dashboard matching the custom wireframe layout with zero Node.js / npm dependencies.

---

## Quick Launch

### Windows (One-Click)
Double-click `web_launch.bat` or run:
```cmd
web_launch.bat
```

### Linux / macOS
```bash
chmod +x web_launch.sh
./web_launch.sh
```

The launcher automatically:
1. Detects Python.
2. Checks whether dependencies are installed; if running on a fresh clone, it automatically installs `web/requirements.txt`.
3. Starts the server at `http://127.0.0.1:8000`.
4. Automatically opens your default web browser.

---

## Features

1. **Top Floating Navigation Bar:** Toggle between **Photo**, **Video**, **Live Feed**, and **Benchmark**.
2. **Side-by-Side Viewport:**
   * **Left Box (Raw Input):** Displays the original low-resolution image, video, or webcam stream.
   * **Right Box (Modified Output):** Displays the 2× super-resolved output with real-time latency and FPS tags.
3. **"Add Network Compression" Checkbox (Below Raw Box):**
   * Injects realistic lossy JPEG DCT blocking and ringing artifacts into the raw input.
   * Demonstrates how the codec-adapted ESPCN model removes compression noise where standard bicubic interpolation fails.
4. **Live Webcam Stream:**
   * Handles browser webcam permissions.
   * Streams frames to the backend over WebSockets with real-time FPS overlay.
   * Includes a **"Capture Frame"** button to download full-resolution snapshots.
5. **Hardware Benchmark:**
   * On-demand benchmark comparing single-thread CPU, multi-thread CPU, and GPU (CUDA / DirectML).
6. **Live Telemetry & Event Drawer:**
   * Real-time Server-Sent Events (SSE) showing backend status, hardware, and error notifications.
