# Checkpoints

Trained ESPCN weights (PyTorch state dicts). Standard size is about 86 KB per file and the wide models about 314 KB.
The network size and options are read from the file, so any of them loads with
`models.espcn.load_espcn(path)`. Export to ONNX with `export_onnx.py`.

## Naming
- `baseline`: trained on clean bicubic-downscaled frames only (conventional ESPCN setup)
- `codec` / `codec_adapted`: trained on JPEG and H.264 compressed frames
- `wide`: 128/64 channels instead of 64/32 (capacity experiment)
- `res`: bicubic residual skip (the network learns a correction on top of a bicubic upscale)

## Files
| File | Round | Description |
|---|---|---|
| `espcn_baseline.pth` | v1 | Clean-trained baseline, 60 epochs |
| `espcn_codec_adapted.pth` | v1 | Codec model, 60 epochs, strong compression only |
| `espcn_baseline_v2.pth` | v2 | Clean-trained baseline, 200 epochs (the standard ESPCN in the comparisons and live demo) |
| `espcn_codec_adapted_v2.pth` | v2 | Codec model, 200 epochs, 12 degradation levels |
| `espcn_baseline_wide_v2.pth` | v2 | Wide baseline (capacity ablation) |
| `espcn_codec_wide_v2.pth` | v2 | Wide codec model (capacity ablation) |
| `espcn_baseline_res_v3.pth` | v3 | Residual baseline, trained on clean data (controlled baseline) |
| `espcn_codec_res_v3.pth` | v3 | Residual codec model (**final model**) |

Retrain with `run_v2.bat` and `run_v3.bat` (see the main README).