#!/usr/bin/env python3
"""
Unified quality evaluation: one protocol for every number in the paper
======================================================================
Y-channel PSNR / SSIM (uint8 outputs, data_range=255) on the held-out TEST clips,
for Bicubic + any number of PyTorch (.pth) / ONNX (.onnx) models, across every
degradation level present in data/lr. Writes per-frame CSV, summary CSV (mean, std,
paired difference vs a reference with bootstrap 95% CI), a bar chart and patch crops.

  python evaluate_all.py \
     --pth baseline=checkpoints/espcn_baseline.pth codec=checkpoints/espcn_codec_adapted.pth \
     --onnx codec_fp16=models/espcn_codec_adapted_fp16.onnx --ref baseline
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from models.espcn import ESPCN, load_espcn

def sort_degs(names):
    """clean, then JPEG (high->low quality), then H.264 (low->high CRF)."""
    def key(n):
        if n == "clean": return (0, 0)
        if n.startswith("jpeg_q"): return (1, -int(n[6:]))
        if n.startswith("h264_crf"): return (2, int(n[8:]))
        return (3, 0)
    return sorted(names, key=key)


def read_y(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2YCrCb)[:, :, 0]


def to_u8(a):
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def parse_pairs(items):
    out = []
    for it in items or []:
        name, path = it.split("=", 1)
        out.append((name, Path(path)))
    return out


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pth", nargs="*", default=[], help="name=path.pth")
    ap.add_argument("--onnx", nargs="*", default=[], help="name=path.onnx")
    ap.add_argument("--ref", default="baseline", help="reference method for paired deltas")
    ap.add_argument("--max-frames", type=int, default=0, help="limit frames per degradation (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_models = {}
    for name, p in parse_pairs(a.pth):
        m = load_espcn(p)
        torch_models[name] = m.to(dev).eval()
    onnx_models = {}
    if a.onnx:
        import onnxruntime as ort
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        for name, p in parse_pairs(a.onnx):
            onnx_models[name] = ort.InferenceSession(str(p), providers=prov)

    def infer(name, y):
        if name == "bicubic":
            return cv2.resize(y, (y.shape[1] * a.scale, y.shape[0] * a.scale), interpolation=cv2.INTER_CUBIC)
        if name in torch_models:
            x = torch.from_numpy(y).float().div(255)[None, None].to(dev)
            with torch.no_grad():
                return to_u8(torch_models[name](x).squeeze().cpu().numpy())
        s = onnx_models[name]; i = s.get_inputs()[0]
        x = (y.astype(np.float16 if "float16" in i.type else np.float32) / 255.0)[None, None]
        return to_u8(s.run(None, {i.name: x})[0].squeeze().astype(np.float32))

    methods = ["bicubic"] + list(torch_models) + list(onnx_models)
    degs = sort_degs([p.name for p in (a.data / "lr").iterdir() if (p / a.split).exists()])
    if not degs:
        raise SystemExit("[ERROR] no degradation folders found under data/lr")
    print("Methods:", methods, "| degradations:", degs)

    rows, crops = [], {}
    for d in degs:
        files = sorted((a.data / "hr" / a.split).glob("*/*.png"))
        if a.max_frames:
            files = files[:a.max_frames]
        for k, hp in enumerate(files):
            rel = hp.relative_to(a.data / "hr")
            hr = read_y(hp); lr = read_y(a.data / "lr" / d / rel)
            for m in methods:
                sr = infer(m, lr)
                rows.append(dict(degradation=d, frame=str(rel), method=m,
                                 psnr=psnr(hr, sr, data_range=255), ssim=ssim(hr, sr, data_range=255)))
                if k == len(files) // 2:
                    crops.setdefault(d, {"hr": hr})[m] = sr
        print(f"  evaluated {d}: {len(files)} frames")

    df = pd.DataFrame(rows)
    (a.out / "csv").mkdir(parents=True, exist_ok=True); (a.out / "figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "csv" / "quality_per_frame.csv", index=False)

    summ = []
    for (d, m), g in df.groupby(["degradation", "method"]):
        r = dict(degradation=d, method=m, n_frames=len(g), psnr_mean=g.psnr.mean(), psnr_std=g.psnr.std(),
                 ssim_mean=g.ssim.mean(), ssim_std=g.ssim.std())
        if a.ref in methods and m != a.ref:
            ref = df[(df.degradation == d) & (df.method == a.ref)].set_index("frame")
            cur = g.set_index("frame"); dp = (cur.psnr - ref.psnr).dropna().values; ds = (cur.ssim - ref.ssim).dropna().values
            lo, hi = boot_ci(dp)
            r.update(dpsnr_vs_ref=dp.mean(), dpsnr_ci_lo=lo, dpsnr_ci_hi=hi, dssim_vs_ref=ds.mean())
        summ.append(r)
    sm = pd.DataFrame(summ)
    sm["degradation"] = pd.Categorical(sm.degradation, degs, ordered=True)
    sm = sm.sort_values(["degradation", "method"]).reset_index(drop=True)
    sm.to_csv(a.out / "csv" / "quality_summary.csv", index=False)
    print("\n" + sm.round(4).to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for metric, ylab in (("psnr", "PSNR (dB)"), ("ssim", "SSIM")):
        pv = sm.pivot(index="degradation", columns="method", values=f"{metric}_mean")[methods]
        er = sm.pivot(index="degradation", columns="method", values=f"{metric}_std")[methods]
        ax = pv.plot.bar(yerr=er, figsize=(11, 4.5), capsize=2); ax.set_ylabel(ylab)
        ax.set_ylim(pv.min().min() - (1.0 if metric == "psnr" else 0.02), pv.max().max() + (0.5 if metric == "psnr" else 0.01))
        plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(a.out / "figures" / f"{metric}_bar.png", dpi=200); plt.close()

    jp = [d for d in degs if d.startswith("jpeg")]; hh = [d for d in degs if d.startswith("h264")]
    targets = {"jpeg_q30", "h264_crf36"} | ({jp[-1]} if jp else set()) | ({hh[-1]} if hh else set())
    for d in sorted(targets):
        if d not in crops:
            continue
        c = crops[d]; H, W = c["hr"].shape; ch, cw = H // 4, W // 4; t, l = H // 3, W // 3
        fig, axs = plt.subplots(1, len(methods) + 1, figsize=(3 * (len(methods) + 1), 3))
        for ax, (n, im) in zip(axs, [("ground truth", c["hr"])] + [(m, c[m]) for m in methods]):
            ax.imshow(im[t:t + ch, l:l + cw], cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_title(n, fontsize=9); ax.axis("off")
        plt.tight_layout(); plt.savefig(a.out / "figures" / f"patches_{d}.png", dpi=200); plt.close()
    print("\nSaved CSVs to results/csv and figures to results/figures")


if __name__ == "__main__":
    main()
