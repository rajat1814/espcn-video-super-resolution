@echo off
REM ============================================================================
REM  RESTART-SAFE v2 pipeline. You may close the window at any time and run this
REM  file again: every finished stage is skipped, unfinished ones start over.
REM  Edit WIDE=0 below to skip the wider-model experiment (saves ~30-40 minutes).
REM ============================================================================
cd /d "%~dp0"
call venv\Scripts\activate.bat
set EPOCHS=200
set STEPS=300
set WIDE=1
set DEGS=clean jpeg_q10 jpeg_q20 jpeg_q30 jpeg_q50 jpeg_q70 jpeg_q90 h264_crf24 h264_crf30 h264_crf36 h264_crf42 h264_crf45
if not exist checkpoints mkdir checkpoints
if not exist models mkdir models

if not exist data_v2\.done (
  echo [1/7] Building dataset data_v2 ...
  python make_datasets.py --videos data\videos --out data_v2 --frames-per-seq 60 --segments 4 --test-clips talk07 talk08 --jpeg-q 10 20 30 50 70 90 --h264-crf 24 30 36 42 45 --clean-out
  if errorlevel 1 goto fail
  echo done> data_v2\.done
) else echo [1/7] dataset exists - skipping

if not exist checkpoints\espcn_baseline_v2.pth (
  echo [2/7] Training baseline ...
  python train.py --mode clean --data data_v2 --epochs %EPOCHS% --steps-per-epoch %STEPS% --out checkpoints\espcn_baseline_v2.pth
  if errorlevel 1 goto fail
) else echo [2/7] baseline exists - skipping

if not exist checkpoints\espcn_codec_adapted_v2.pth (
  echo [3/7] Training codec-adapted model ...
  python train.py --mode codec --data data_v2 --degradations %DEGS% --epochs %EPOCHS% --steps-per-epoch %STEPS% --out checkpoints\espcn_codec_adapted_v2.pth
  if errorlevel 1 goto fail
) else echo [3/7] codec model exists - skipping

if "%WIDE%"=="1" if not exist checkpoints\espcn_baseline_wide_v2.pth (
  echo [4/7] Training WIDE baseline ...
  python train.py --mode clean --data data_v2 --c1 128 --c2 64 --epochs %EPOCHS% --steps-per-epoch %STEPS% --out checkpoints\espcn_baseline_wide_v2.pth
  if errorlevel 1 goto fail
)
if "%WIDE%"=="1" if not exist checkpoints\espcn_codec_wide_v2.pth (
  echo [4/7] Training WIDE codec model ...
  python train.py --mode codec --data data_v2 --degradations %DEGS% --c1 128 --c2 64 --epochs %EPOCHS% --steps-per-epoch %STEPS% --out checkpoints\espcn_codec_wide_v2.pth
  if errorlevel 1 goto fail
)

if not exist models\espcn_codec_v2_fp16.onnx (
  echo [5/7] Exporting ONNX FP32 and FP16 ...
  python export_onnx.py --weights checkpoints\espcn_codec_adapted_v2.pth --output models\espcn_codec_v2.onnx --scale 2
  if errorlevel 1 goto fail
  python onnx_fp16.py --input models\espcn_codec_v2.onnx --output models\espcn_codec_v2_fp16.onnx
  if errorlevel 1 goto fail
)
if "%WIDE%"=="1" if not exist models\espcn_codec_wide_v2_fp16.onnx (
  python export_onnx.py --weights checkpoints\espcn_codec_wide_v2.pth --output models\espcn_codec_wide_v2.onnx --scale 2
  if errorlevel 1 goto fail
  python onnx_fp16.py --input models\espcn_codec_wide_v2.onnx --output models\espcn_codec_wide_v2_fp16.onnx
  if errorlevel 1 goto fail
)

set PTH=baseline=checkpoints\espcn_baseline_v2.pth codec=checkpoints\espcn_codec_adapted_v2.pth
set ONNX=codec_fp16=models\espcn_codec_v2_fp16.onnx
if "%WIDE%"=="1" set PTH=%PTH% baseline_wide=checkpoints\espcn_baseline_wide_v2.pth codec_wide=checkpoints\espcn_codec_wide_v2.pth
if "%WIDE%"=="1" set ONNX=%ONNX% codec_wide_fp16=models\espcn_codec_wide_v2_fp16.onnx
if not exist results_v2\csv\quality_summary.csv (
  echo [6/7] Evaluating quality - this takes several minutes ...
  python evaluate_all.py --data data_v2 --pth %PTH% --onnx %ONNX% --ref baseline --out results_v2
  if errorlevel 1 goto fail
) else echo [6/7] evaluation exists - skipping

if not exist results_v2\csv\latency_narrow\latency.csv (
  echo [7/7] Benchmarking latency ...
  python benchmark_latency.py --pth checkpoints\espcn_codec_adapted_v2.pth --onnx-fp32 models\espcn_codec_v2.onnx --onnx-fp16 models\espcn_codec_v2_fp16.onnx --cpu-threads 8 --out results_v2\csv\latency_narrow
  if errorlevel 1 goto fail
)
if "%WIDE%"=="1" if not exist results_v2\csv\latency_wide\latency.csv (
  python benchmark_latency.py --pth checkpoints\espcn_codec_wide_v2.pth --onnx-fp32 models\espcn_codec_wide_v2.onnx --onnx-fp16 models\espcn_codec_wide_v2_fp16.onnx --cpu-threads 8 --out results_v2\csv\latency_wide
  if errorlevel 1 goto fail
)
echo.
echo ALL DONE. Results are in the results_v2 folder.
pause
exit /b 0

:fail
echo.
echo A stage FAILED (see the red text above). Finished stages are kept: fix the problem or send the error, then run run_v2.bat again.
pause
exit /b 1
