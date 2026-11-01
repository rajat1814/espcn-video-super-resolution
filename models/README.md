# Models

`espcn.py` defines the ESPCN architecture (configurable width, optional bicubic residual skip).
`load_espcn(path)` rebuilds the right model from any checkpoint in `checkpoints/`.

The `.onnx` files are exports of those checkpoints (input `lr_y` [1,1,H,W] in 0..1, output `sr_y` [1,1,2H,2W], dynamic H and W).
Create them with `export_onnx.py`, then `onnx_fp16.py` for the FP16 versions.

| File | Round | Description |
|---|---|---|
| `espcn_baseline.onnx` | v1 | Clean-trained baseline |
| `espcn_codec_adapted.onnx` / `_fp16.onnx` | v1 | Codec-adapted model, FP32 / FP16 |
| `espcn_codec_adapted_int8.onnx` | v1 | Dynamic INT8 quantization (slower than FP32 on the test hardware) |
| `espcn_baseline_v2.onnx` | v2 | Standard ESPCN baseline (used as the baseline view in `live_stream.py`) |
| `espcn_codec_v2.onnx` / `_fp16.onnx` | v2 | Codec model trained on 12 degradation levels |
| `espcn_codec_wide_v2.onnx` / `_fp16.onnx` | v2 | Wide (128/64) codec model, capacity ablation |
| `espcn_codec_res_v3.onnx` | v3 | Final model (residual), FP32 |
| `espcn_codec_res_v3_fp16.onnx` | v3 | **Final model, FP16 (used by the live demo)** |

Run the demo with `python live_stream.py --model models/espcn_codec_res_v3_fp16.onnx --source 0 --scale 2`.