#!/usr/bin/env python3
"""
Latency / throughput benchmark: PyTorch vs ONNX Runtime, FP32 vs FP16, CPU vs GPU
=================================================================================
Run on the SAME machine you will demo on (plugged in, high-performance power mode).
50 warm-up runs, 200 timed runs, GPU synchronised. ONNX timings include host<->device
copies (end-to-end per frame, as in live_stream.py).
Writes results/csv/latency.csv and results/csv/hardware_info.json

  python benchmark_latency.py --pth checkpoints/espcn_codec_adapted.pth \
      --onnx-fp32 models/espcn_codec_adapted.onnx --onnx-fp16 models/espcn_codec_adapted_fp16.onnx \
      [--onnx-int8 models/espcn_codec_adapted_int8.onnx]
"""
from __future__ import annotations
import argparse, json, platform, time
from pathlib import Path
import numpy as np, pandas as pd, torch

from models.espcn import ESPCN, load_espcn


def stats(times):
    t = np.asarray(times)
    return dict(median_ms=np.median(t), mean_ms=t.mean(), std_ms=t.std(), p95_ms=np.percentile(t, 95),
                fps_from_median=1000.0 / np.median(t), meets_30fps=bool(np.median(t) <= 1000 / 30))


def bench(fn, warmup, runs, sync=None):
    for _ in range(warmup):
        fn()
    if sync: sync()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn()
        if sync: sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return stats(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", type=Path); ap.add_argument("--onnx-fp32", type=Path)
    ap.add_argument("--onnx-fp16", type=Path); ap.add_argument("--onnx-int8", type=Path)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--height", type=int, default=180); ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--cpu-threads", type=int, nargs="*", default=[0], help="ONNX CPU intra-op threads (0=auto); e.g. 1 2 4 8")
    ap.add_argument("--out", type=Path, default=Path("results/csv"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort

    shape = (1, 1, a.height, a.width)
    x32 = np.random.rand(*shape).astype(np.float32)
    has_gpu = torch.cuda.is_available()
    avail = ort.get_available_providers()
    hw = dict(platform=platform.platform(), cpu=platform.processor(), python=platform.python_version(),
              torch=torch.__version__, onnxruntime=ort.__version__, ort_providers=avail,
              gpu=torch.cuda.get_device_name(0) if has_gpu else None, input_shape=shape)
    rows = []

    def add(name, device, prec, st, extra=None):
        r = dict(model=name, device=device, precision=prec, input=f"{a.height}x{a.width}", runs=a.runs, **st)
        r.update(extra or {}); rows.append(r)
        print(f"{name:<22}{device:<10}{prec:<6} median={st['median_ms']:.2f} ms  p95={st['p95_ms']:.2f}  FPS={st['fps_from_median']:.1f}")

    if a.pth:
        m = load_espcn(a.pth)
        xt = torch.from_numpy(x32)
        with torch.no_grad():
            add("PyTorch", "cpu", "fp32", bench(lambda: m(xt), a.warmup, a.runs))
            if has_gpu:
                mg = m.to("cuda"); xg = xt.to("cuda")
                add("PyTorch", "cuda", "fp32", bench(lambda: mg(xg), a.warmup, a.runs, torch.cuda.synchronize))
                mh = mg.half(); xh = xg.half()
                add("PyTorch", "cuda", "fp16", bench(lambda: mh(xh), a.warmup, a.runs, torch.cuda.synchronize))

    def run_onnx(label, path, prec):
        if not path: return
        sizes = dict(size_kb=round(path.stat().st_size / 1024, 1))
        configs = []
        for th in a.cpu_threads:
            configs.append(("cpu", ["CPUExecutionProvider"], th))
        if "CUDAExecutionProvider" in avail:
            configs.append(("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], 0))
        for dev, prov, th in configs:
            so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if th: so.intra_op_num_threads = th
            s = ort.InferenceSession(str(path), sess_options=so, providers=prov)
            if dev == "cuda" and s.get_providers()[0] != "CUDAExecutionProvider":
                print(f"[skip] CUDA EP failed to load for {path.name}"); continue
            i = s.get_inputs()[0]; x = x32.astype(np.float16) if "float16" in i.type else x32
            st = bench(lambda: s.run(None, {i.name: x}), a.warmup, a.runs)
            add(label, dev if dev == "cuda" else f"cpu(t={th or 'auto'})", prec, st, sizes)

    run_onnx("ONNX Runtime", a.onnx_fp32, "fp32")
    run_onnx("ONNX Runtime", a.onnx_fp16, "fp16")
    run_onnx("ONNX Runtime INT8", a.onnx_int8, "int8")

    df = pd.DataFrame(rows)
    base = df[(df.model == "PyTorch") & (df.device == "cpu")]
    if len(base):
        df["speedup_vs_pytorch_cpu"] = base.median_ms.iloc[0] / df.median_ms
    df.to_csv(a.out / "latency.csv", index=False)
    (a.out / "hardware_info.json").write_text(json.dumps(hw, indent=2, default=str))
    print(f"\nSaved {a.out/'latency.csv'} and hardware_info.json")


if __name__ == "__main__":
    main()
