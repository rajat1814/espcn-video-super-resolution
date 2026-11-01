#!/usr/bin/env python3
"""
Same-frame comparison figure for the report/slides (ground truth | bicubic | baseline | codec-adapted)
=======================================================================================================
Uses the SAME frame index from the original video (ground truth, resized to 640x360) and from the
low-bitrate test clip (made with ffmpeg from that video), runs the live-pipeline logic
(BGR->YCrCb, model on Y, chroma bicubic) and saves full-frame + zoomed-crop panels with Y-channel PSNR.

  python make_comparison_frame.py --hr-video data\\videos\\talk07.mp4 --lr-video results\\lowbit_test.mp4 ^
      --baseline models\\espcn_baseline_v2.onnx --codec models\\espcn_codec_res_v3_fp16.onnx --frames 60 150 250
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2, numpy as np


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read(); cap.release()
    if not ok:
        raise SystemExit(f"[ERROR] cannot read frame {idx} from {path}")
    return f


def upscale(sess, lr_bgr, size):
    ycc = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)
    i = sess.get_inputs()[0]
    y = (ycc[:, :, 0].astype(np.float16 if "float16" in i.type else np.float32) / 255.0)[None, None]
    out = sess.run(None, {i.name: y})[0].squeeze().astype(np.float32)
    y_sr = np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)
    cr = cv2.resize(ycc[:, :, 1], size, interpolation=cv2.INTER_CUBIC)
    cb = cv2.resize(ycc[:, :, 2], size, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cv2.merge([y_sr, cr, cb]), cv2.COLOR_YCrCb2BGR)


def y_psnr(a, b):
    ya = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    yb = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
    mse = np.mean((ya - yb) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-video", type=Path, required=True)
    ap.add_argument("--lr-video", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--codec", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[60, 150, 250])
    ap.add_argument("--crop", type=int, nargs=4, default=[200, 30, 240, 135], metavar=("X", "Y", "W", "H"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    a = ap.parse_args()

    import onnxruntime as ort
    try:
        ort.preload_dlls()
    except Exception:
        pass
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sb = ort.InferenceSession(str(a.baseline), providers=prov)
    sc = ort.InferenceSession(str(a.codec), providers=prov)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    a.out.mkdir(parents=True, exist_ok=True)
    size = (640, 360); x, y, w, h = a.crop

    for idx in a.frames:
        gt = cv2.resize(read_frame(a.hr_video, idx), size, interpolation=cv2.INTER_AREA)
        lr = read_frame(a.lr_video, idx)
        lr = cv2.resize(lr, (320, 180), interpolation=cv2.INTER_AREA) if lr.shape[1] != 320 else lr
        outs = {"ground truth": gt,
                "bicubic": cv2.resize(lr, size, interpolation=cv2.INTER_CUBIC),
                "baseline ESPCN": upscale(sb, lr, size),
                "codec-adapted": upscale(sc, lr, size)}
        fig, axs = plt.subplots(2, 4, figsize=(16, 5.6))
        for k, (name, im) in enumerate(outs.items()):
            t = name if name == "ground truth" else f"{name} | Y-PSNR {y_psnr(gt, im):.2f} dB"
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            axs[0, k].imshow(rgb); axs[0, k].set_title(t, fontsize=10); axs[0, k].axis("off")
            axs[1, k].imshow(rgb[y:y + h, x:x + w], interpolation="nearest"); axs[1, k].axis("off")
            print(f"frame {idx:>4} {name:<15} Y-PSNR {y_psnr(gt, im):.2f} dB" if name != "ground truth" else "", end="\n" if name != "ground truth" else "")
        plt.tight_layout(); p = a.out / f"live_compare_frame{idx}.png"; plt.savefig(p, dpi=200); plt.close()
        print("saved", p)


if __name__ == "__main__":
    main()
