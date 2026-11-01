# Notebooks

`minor_project_original.ipynb` is the team's original Google Colab prototype, kept for reference and history.
It covers the first end-to-end experiment: frame extraction from a single video, synthetic JPEG (Q30) and H.264
degradation, baseline ESPCN training, codec adaptation, PSNR/SSIM evaluation, and an ONNX Runtime CPU benchmark.

It is **superseded** by the scripts in the repository root, which fix the limitations of this first run:

| Prototype (this notebook) | Final pipeline (repository scripts) |
|---|---|
| One source video, about 100 frames, last 20 used as test frames | 8 clips, split by clip; 480 test frames from 2 held-out clips |
| JPEG Q30 (plus one H.264 level) | 12 input levels: JPEG Q10-Q90, H.264 CRF 24-45, clean |
| Baseline 5 epochs; codec model continued for 5 more | Both models trained from scratch with identical settings, 200 epochs |
| CPU benchmark only | CPU and GPU, FP32 and FP16, plus a 15-minute end-to-end test |

Results from this notebook are in `results/legacy/`.
To run it, open it in Google Colab; it expects its data in Google Drive, so paths will need adjusting.
To reproduce the final results, use the scripts described in the main README instead.