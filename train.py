#!/usr/bin/env python3
"""
ESPCN trainer for BOTH models with identical settings (fair comparison)
=======================================================================
  --mode clean : LR = clean bicubic downscale            (baseline)
  --mode codec : LR = random JPEG / H.264 degradation   (codec-adapted)
Both are trained FROM SCRATCH with the same seed, epochs, optimizer, patch size,
batch size and number of steps; the ONLY difference is the degradation of the input.

  python train.py --mode clean --out checkpoints/espcn_baseline.pth
  python train.py --mode codec --out checkpoints/espcn_codec_adapted.pth
  # ablation: JPEG-only training
  python train.py --mode codec --degradations jpeg_q20 jpeg_q30 jpeg_q50 --out checkpoints/espcn_jpeg_only.pth
"""
from __future__ import annotations
import argparse, csv, random, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn

from models.espcn import ESPCN

DEFAULT_CODEC = ["jpeg_q20", "jpeg_q30", "jpeg_q50", "h264_crf30", "h264_crf36", "h264_crf42"]


def read_y(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2YCrCb)[:, :, 0]


def load_split(root: Path, degs: list[str], split: str = "train"):
    """Preload Y planes as uint8: list of (hr, {deg: lr})."""
    items = []
    for seq_dir in sorted((root / "hr" / split).glob("*")):
        for hr_path in sorted(seq_dir.glob("*.png")):
            rel = hr_path.relative_to(root / "hr")
            lr = {d: read_y(root / "lr" / d / rel) for d in degs}
            items.append((read_y(hr_path), lr))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clean", "codec"], required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--degradations", nargs="*", default=None)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--residual", action="store_true", help="add bicubic skip connection (global residual learning)")
    ap.add_argument("--c1", type=int, default=64, help="conv1 width")
    ap.add_argument("--c2", type=int, default=32, help="conv2 width")
    ap.add_argument("--patch", type=int, default=48, help="LR patch size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["l1", "mse"], default="l1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    degs = ["clean"] if a.mode == "clean" else (a.degradations or [p.name for p in sorted((a.data / "lr").iterdir()) if p.name != "clean"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"mode={a.mode} degradations={degs} device={dev}")
    items = load_split(a.data, degs, "train")
    if not items:
        raise SystemExit("[ERROR] no training frames; run make_datasets.py first")
    print(f"Training frames: {len(items)}")

    model = ESPCN(a.scale, a.c1, a.c2, residual=a.residual).to(dev)
    print(f"Residual={a.residual} | widths {a.c1}/{a.c2}, parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.L1Loss() if a.loss == "l1" else nn.MSELoss()
    s, p = a.scale, a.patch
    log_path = a.log or Path("results/csv") / f"train_log_{a.out.stem}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; t0 = time.time()
        for _ in range(a.steps_per_epoch):
            xb, yb = [], []
            for _ in range(a.batch_size):
                hr, lrs = random.choice(items)
                lr = lrs[random.choice(degs)]
                h, w = lr.shape
                t, l = random.randint(0, h - p), random.randint(0, w - p)
                x = lr[t:t + p, l:l + p]; y = hr[t * s:(t + p) * s, l * s:(l + p) * s]
                if random.random() < 0.5:
                    x, y = x[:, ::-1], y[:, ::-1]
                xb.append(x.copy()); yb.append(y.copy())
            x = torch.from_numpy(np.stack(xb)).float().div(255).unsqueeze(1).to(dev)
            y = torch.from_numpy(np.stack(yb)).float().div(255).unsqueeze(1).to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        mean = tot / a.steps_per_epoch
        rows.append({"epoch": ep, "loss": mean, "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)})
        print(f"Epoch {ep:02d}/{a.epochs} | {a.loss.upper()} {mean:.6f} | {rows[-1]['seconds']}s")

    torch.save(model.state_dict(), a.out)
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Saved {a.out} and {log_path}")


if __name__ == "__main__":
    main()
