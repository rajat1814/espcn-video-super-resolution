@echo off
REM ============================================================================
REM  RESTART-SAFE v3 pipeline (residual ESPCN = bicubic skip + learned correction).
REM  Close the window any time; run this file again and finished stages are skipped.
REM  Needs data_v2 from run_v2.bat and the v2 checkpoints (kept for comparison).
REM ============================================================================
cd /d "%~dp0"
call venv\Scripts\activate.bat
set EPOCHS=200
set STEPS=300
set DEGS=clean jpeg_q10 jpeg_q20 jpeg_q30 jpeg_q50 jpeg_q70 jpeg_q90 h264_crf24 h264_crf30 h264_crf36 h264_crf42 h264_crf45
if not exist data_v2\.done (
  echo data_v2 not found - run run_v2.bat first.
  goto fail
)

if not exist checkpoints\espcn_baseline_res_v3.pth (
  echo [1/5] Training residual baseline ...
  python train.py --mode clean --residual --data data_v2 --epochs %EPOCHS% --steps-per-epoch %STEPS% --out checkpoints\espcn_baseline_res_v3.pth
  if errorlevel 1 goto fail
) else echo [1/5] residual baseline exists - skipping

if not exist checkpoints\espcn_codec_res_v3.pth (
  echo [2/5] Training residual codec-adapted model ...
  python train.py --mode codec --residual --data data_v2 --degradations %DEGS% --epochs %EPOCHS% --steps-per-epoch %STEPS% --out checkpoints\espcn_codec_res_v3.pth
  if errorlevel 1 goto fail
) else echo [2/5] residual codec model exists - skipping

if not exist models\espcn_codec_res_v3_fp16.onnx (
  echo [3/5] Exporting ONNX FP32 and FP16 ...
  python export_onnx.py --weights checkpoints\espcn_codec_res_v3.pth --output models\espcn_codec_res_v3.onnx --scale 2
  if errorlevel 1 goto fail
  python onnx_fp16.py --input models\espcn_codec_res_v3.onnx --output models\espcn_codec_res_v3_fp16.onnx
  if errorlevel 1 goto fail
) else echo [3/5] ONNX exists - skipping

if not exist results_v3\csv\quality_summary.csv (
  echo [4/5] Evaluating quality - several minutes ...
  python evaluate_all.py --data data_v2 --pth baseline=checkpoints\espcn_baseline_v2.pth codec=checkpoints\espcn_codec_adapted_v2.pth baseline_res=checkpoints\espcn_baseline_res_v3.pth codec_res=checkpoints\espcn_codec_res_v3.pth --onnx codec_res_fp16=models\espcn_codec_res_v3_fp16.onnx --ref baseline_res --out results_v3
  if errorlevel 1 goto fail
) else echo [4/5] evaluation exists - skipping

if not exist results_v3\csv\latency_res\latency.csv (
  echo [5/5] Benchmarking latency ...
  python benchmark_latency.py --pth checkpoints\espcn_codec_res_v3.pth --onnx-fp32 models\espcn_codec_res_v3.onnx --onnx-fp16 models\espcn_codec_res_v3_fp16.onnx --cpu-threads 8 --out results_v3\csv\latency_res
  if errorlevel 1 goto fail
)
echo.
echo ALL DONE. Results are in the results_v3 folder.
pause
exit /b 0

:fail
echo.
echo Stopped (see the message above). Finished stages are kept: fix the problem or send me the error, then run run_v3.bat again.
pause
exit /b 1
