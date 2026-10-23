# Real-Time Video Super-Resolution & Codec Artifact Suppression (ESPCN + ONNX)

Real-time single-channel (Y-luminance) video super-resolution system built with **PyTorch**, **ONNX Runtime**, and **OpenCV**. This project trains and deploys a lightweight Sub-Pixel Convolutional Neural Network (**ESPCN**, $2\times$ spatial upscale) specifically adapted to suppress H.264 and JPEG compression artifacts on low-bitrate talking-head video streams.

## System Architecture

1. **Color Space Decomposition (`BGR` $\rightarrow$ `YCrCb`)**: Incoming low-resolution frames are converted to `YCrCb` color space. Because human visual perception is primarily sensitive to high-frequency detail in luminance, neural super-resolution is applied exclusively to the single-channel luminance plane (`Y` of shape `[1, 1, H, W]`), reducing computational cost by roughly $3\times$ compared to full RGB inference.
2. **Sub-Pixel Convolution (`ESPCN`)**: A 3-layer convolutional backbone extracts feature maps in low-resolution space (`1 -> 64 -> 32 -> 4`) followed by a parameter-free `PixelShuffle(2)` rearrangement operator that reconstructs the high-resolution luminance plane (`sr_y`).
3. **Hardware-Accelerated ONNX Inference**: Models are exported to ONNX (opset 17) with dynamic spatial axes and executed via **ONNX Runtime** across NVIDIA CUDA GPUs (`CUDAExecutionProvider`) or multi-threaded CPUs (`CPUExecutionProvider`) with automatic hardware fallback.

## Repository Structure

| Path | Description |
| :--- | :--- |
| `models/espcn.py` | PyTorch 3-layer ESPCN architecture definition (`conv1` $\rightarrow$ `conv2` $\rightarrow$ `conv3` + `PixelShuffle(2)`) |
| `models/espcn_codec_adapted.onnx` | Exported FP32 ONNX model trained on codec-degraded frames (`lr_y` $\rightarrow$ `sr_y`) |
| `models/espcn_codec_adapted_fp16.onnx` | Half-precision FP16 ONNX model generated via `onnx_fp16.py` |
| `models/espcn_codec_adapted_int8.onnx` | Dynamically quantized INT8 ONNX model evaluated during CPU benchmarking |
| `export_onnx.py` | Exports PyTorch `.pth` checkpoints to ONNX FP32 format and validates numerical parity |
| `onnx_fp16.py` | Converts ONNX FP32 models to mixed-precision FP16 and profiles inference latency |
| `live_stream.py` | Real-time OpenCV side-by-side streaming engine, telemetry HUD, stress tester, and MP4 recorder |
| `notebooks/minor_project.ipynb` | End-to-end notebook for dataset extraction, synthetic codec degradation, training, and evaluation |
| `results/` | Quantitative PSNR/SSIM CSV logs, ONNX latency benchmarks, and visual comparison figures |

## Environment Setup

Create a Python 3.9+ virtual environment and install all required dependencies from `requirements.txt` (which automatically installs the unified GPU+CPU `onnxruntime-gpu` package on Windows/Linux and `onnxruntime` on macOS):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage Guide

### 1. Real-Time Side-by-Side Streaming (`live_stream.py`)
Launch the live side-by-side comparison window (**Raw Degraded / Bicubic Baseline** on the left vs. **Codec-Adapted ESPCN** on the right) with an EMA-smoothed FPS and frame latency HUD:

```bash
# Run live on default webcam (auto-detects GPU or CPU):
python live_stream.py --model models/espcn_codec_adapted.onnx --source 0 --scale 2

# Run on a local video file (.mp4 / .avi):
python live_stream.py --model models/espcn_codec_adapted.onnx --source path/to/video.mp4 --scale 2

# Run using the built-in synthetic test stream (no camera required):
python live_stream.py --model models/espcn_codec_adapted.onnx --source synthetic --scale 2

# Explicitly force GPU or CPU execution:
python live_stream.py --model models/espcn_codec_adapted.onnx --source 0 --device gpu
python live_stream.py --model models/espcn_codec_adapted.onnx --source 0 --device cpu
```

**Interactive Keyboard Controls:**
* **`1`**: Switch right pane to **Bicubic Interpolation Baseline**
* **`2`**: Switch right pane to **Secondary / Baseline ESPCN Model**
* **`3`**: Switch right pane to **Primary Codec-Adapted ESPCN Model**
* **`G`**: Toggle live between **GPU (`CUDAExecutionProvider`)** and **CPU (`CPUExecutionProvider`)**
* **`S`**: Save a side-by-side PNG snapshot to the working directory
* **`Q` / `ESC`**: Exit the application

### 2. Record Live Demo Video & Run Stability Stress Tests
Record a side-by-side `.mp4` demonstration video or run a continuous stability test logging latency telemetry to `results/pipeline_stress_test.csv`:

```bash
# Record live side-by-side stream to MP4:
python live_stream.py --model models/espcn_codec_adapted.onnx --source 0 --record-video results/demo_backup.mp4

# Run a 60-second headless stability stress test:
python live_stream.py --model models/espcn_codec_adapted.onnx --source synthetic --stress-seconds 60
```

### 3. ONNX Export & Mixed-Precision (FP16) Conversion
Export a PyTorch `.pth` checkpoint to ONNX FP32 and convert it to half-precision FP16:

```bash
python export_onnx.py --weights espcn_codec_adapted.pth --output models/espcn_codec_adapted.onnx --scale 2
python onnx_fp16.py --input models/espcn_codec_adapted.onnx --output models/espcn_codec_adapted_fp16.onnx
```

## Experimental Results Summary

| Evaluation Metric / Runtime | Clean-Trained Baseline | Codec-Adapted ESPCN (Ours) | Observations |
| :--- | :---: | :---: | :--- |
| **Average PSNR (dB)** | 35.42 dB | **36.72 dB** | **+1.30 dB** gain on compressed test frames (JPEG Q=30) |
| **Average SSIM** | 0.9276 | **0.9508** | **+0.0232** structural similarity improvement |
| **PyTorch FP32 (CPU)** | — | 74.83 ms (13.36 FPS) | Baseline eager execution (`180x320` $\rightarrow$ `360x640`) |
| **ONNX Runtime FP32 (CPU)** | — | **38.72 ms (25.83 FPS)** | **1.93x speedup** via ONNX graph optimization |
| **ONNX Runtime INT8 (CPU)** | — | 550.46 ms (1.82 FPS) | Dynamic quantization overhead on non-VNNI CPU |
