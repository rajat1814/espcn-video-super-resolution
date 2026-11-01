# Results

Final results (residual codec-adapted ESPCN, "v3") are in `csv/`, `figures/` and `tables/`.
All metrics are Y-channel, 2x upscale (320x180 -> 640x360), on 480 frames from two held-out clips (talk07, talk08).
Latency was measured on an RTX 3050 Laptop GPU (see `csv/hardware_info.json`).

| Path | Contents |
|---|---|
| `csv/quality_summary.csv` | Mean/std PSNR and SSIM per method and degradation level, with paired differences and 95% CIs |
| `csv/latency.csv` | Latency and FPS for PyTorch vs ONNX Runtime, FP32 vs FP16, CPU vs GPU |
| `csv/pipeline_stress_test.csv` | 15-minute end-to-end test of `live_stream.py` (frames, dropped frames, latency, wall-clock FPS) |
| `csv/train_log_*.csv` | Training loss per epoch for every model |
| `figures/` | PSNR/SSIM bar charts, zoomed crops (`patches_*`), same-frame live comparisons (`live_compare_*`), latency and training-loss charts |
| `tables/` | The same results as markdown tables |

Earlier runs, kept for the version history:
- `legacy/`: the team's original first run (single clip, JPEG Q30), preliminary and superseded.
- `v1_backup/`: first full evaluation (60-epoch models).
- `../results_v2/`: second round (12 degradation levels, 200 epochs, wide-model ablation).
- `../results_v3/`: third round (residual models), the source of the final files in `csv/` and `figures/`.

Regenerate with `evaluate_all.py`, `benchmark_latency.py`, `make_report_assets.py` and `live_stream.py` (see the main README).
The demo video is not stored in the repository: [Google Drive](https://drive.google.com/file/d/11pM-2tb0tJ6gZHVJxKAJb1DoDb6QG1mg/view?usp=sharing). 