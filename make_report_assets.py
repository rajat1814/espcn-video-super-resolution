#!/usr/bin/env python3
"""Turn result CSVs into paper-ready markdown tables and figures (no hand-typed numbers)."""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = Path("results"); (R / "figures").mkdir(parents=True, exist_ok=True); (R / "tables").mkdir(exist_ok=True)

if (R / "csv/quality_summary.csv").exists():
    q = pd.read_csv(R / "csv/quality_summary.csv").round(4)
    q.to_markdown(R / "tables/quality.md", index=False)
    print("quality table ->", R / "tables/quality.md")

if (R / "csv/latency.csv").exists():
    l = pd.read_csv(R / "csv/latency.csv").round(2)
    l.to_markdown(R / "tables/latency.md", index=False)
    lab = l.model + " " + l.precision + " " + l.device
    ax = l.set_index(lab).median_ms.plot.barh(figsize=(9, 0.5 * len(l) + 2), logx=True)
    ax.axvline(1000 / 30, color="r", ls="--", label="33.3 ms (30 FPS)"); ax.set_xlabel("median latency (ms, log)"); ax.legend()
    plt.tight_layout(); plt.savefig(R / "figures/latency_bar.png", dpi=200); plt.close()
    print("latency table + chart done")

for f in sorted((R / "csv").glob("train_log_*.csv")):
    d = pd.read_csv(f); plt.plot(d.epoch, d.loss, label=f.stem.replace("train_log_", ""))
if plt.get_fignums():
    plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend(); plt.tight_layout()
    plt.savefig(R / "figures/training_loss.png", dpi=200); print("training loss figure done")

s = R / "csv/pipeline_stress_test.csv"
if s.exists():
    pd.read_csv(s).to_markdown(R / "tables/stress_test.md", index=False); print("stress table done")
