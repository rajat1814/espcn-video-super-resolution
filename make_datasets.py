#!/usr/bin/env python3
"""
Synthetic codec-degradation dataset builder (JPEG + H.264) with clip-level splits
=================================================================================
Takes source videos, extracts CONSECUTIVE frames (so H.264 behaves like a real
stream), builds clean HR targets and degraded LR inputs, and splits train/test BY
CLIP so the model is never tested on a scene it was trained on.

Output layout
  data/hr/<split>/<seq>/00000.png                  clean HR target (W x H)
  data/lr/<degradation>/<split>/<seq>/00000.png     LR input (W/scale x H/scale)
degradations: clean, jpeg_q20 jpeg_q30 jpeg_q50, h264_crf30 h264_crf36 h264_crf42

Usage
  python make_datasets.py --videos data/videos --frames-per-seq 60 --segments 2 --test-clips clipE
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys
from pathlib import Path
import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_sequences(video: Path, n: int, segments: int, size: tuple[int, int]):
    """Return `segments` lists of `n` consecutive resized frames (evenly spaced starts)."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n:
        print(f"[WARN] {video.name}: only {total} frames, skipping")
        return []
    starts = [int(i * (total - n) / max(segments - 1, 1)) for i in range(segments)] if segments > 1 else [(total - n) // 2]
    seqs = []
    for st in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st)
        frames = []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
        if len(frames) == n:
            seqs.append(frames)
    cap.release()
    return seqs


def save_seq(frames, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:05d}.png"), f)


def jpeg_degrade(frames, q):
    out = []
    for f in frames:
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
        out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))
    return out


def h264_degrade(frames, crf, work: Path):
    if shutil.which("ffmpeg") is None:
        sys.exit("[ERROR] ffmpeg not found on PATH (needed for H.264 degradation).")
    if work.exists():
        shutil.rmtree(work)
    (work / "in").mkdir(parents=True); (work / "out").mkdir()
    for i, f in enumerate(frames):
        cv2.imwrite(str(work / "in" / f"{i:05d}.png"), f)
    mp4 = work / "c.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", "25",
                    "-i", str(work / "in" / "%05d.png"), "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp4), "-fps_mode", "passthrough",
                    "-start_number", "0", str(work / "out" / "%05d.png")], check=True)
    dec = [cv2.imread(str(p)) for p in sorted((work / "out").glob("*.png"))]
    shutil.rmtree(work)
    if len(dec) != len(frames):
        raise RuntimeError(f"H.264 frame count mismatch ({len(dec)} vs {len(frames)}); alignment broken")
    return dec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=Path("data/videos"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--scale", type=int, default=2, choices=[2, 4])
    ap.add_argument("--hr-size", type=int, nargs=2, default=[640, 360], metavar=("W", "H"))
    ap.add_argument("--frames-per-seq", type=int, default=60)
    ap.add_argument("--segments", type=int, default=2, help="consecutive-frame segments per clip")
    ap.add_argument("--test-clips", nargs="*", default=None, help="clip file stems used ONLY for testing")
    ap.add_argument("--jpeg-q", type=int, nargs="*", default=[20, 30, 50])
    ap.add_argument("--h264-crf", type=int, nargs="*", default=[30, 36, 42])
    ap.add_argument("--clean-out", action="store_true", help="delete existing data/hr and data/lr first")
    a = ap.parse_args()

    W, H = a.hr_size
    assert W % a.scale == 0 and H % a.scale == 0, "HR size must be divisible by scale"
    vids = sorted(p for p in a.videos.glob("*") if p.suffix.lower() in VIDEO_EXT)
    if not vids:
        sys.exit(f"[ERROR] no videos in {a.videos}")
    stems = [v.stem for v in vids]
    if a.test_clips is None:
        k = max(1, round(len(vids) * 0.25))
        a.test_clips = stems[-k:]
    if len(vids) < 2:
        print("[WARN] Only one clip: using first segments for train and last for test (weak generalisation evidence).")
        if a.segments < 2:
            sys.exit("[ERROR] need --segments >= 2 with a single clip")
    print("Clips:", stems, "| test clips:", a.test_clips)

    if a.clean_out:
        for d in ("hr", "lr"):
            shutil.rmtree(a.out / d, ignore_errors=True)

    counts = {"train": 0, "test": 0}
    for v in vids:
        seqs = extract_sequences(v, a.frames_per_seq, a.segments, (W, H))
        for j, frames in enumerate(seqs):
            if len(vids) < 2:
                split = "test" if j == len(seqs) - 1 else "train"
            else:
                split = "test" if v.stem in a.test_clips else "train"
            seq = f"{v.stem}_s{j}"
            save_seq(frames, a.out / "hr" / split / seq)
            lr = [cv2.resize(f, (W // a.scale, H // a.scale), interpolation=cv2.INTER_CUBIC) for f in frames]
            save_seq(lr, a.out / "lr" / "clean" / split / seq)
            for q in a.jpeg_q:
                save_seq(jpeg_degrade(lr, q), a.out / "lr" / f"jpeg_q{q}" / split / seq)
            for crf in a.h264_crf:
                save_seq(h264_degrade(lr, crf, a.out / "_tmp"), a.out / "lr" / f"h264_crf{crf}" / split / seq)
            counts[split] += len(frames)
            print(f"  {seq:<28} -> {split} ({len(frames)} frames)")
    print("Done. Frames per split:", counts)
    if counts["test"] == 0 or counts["train"] == 0:
        sys.exit("[ERROR] empty train or test split; check --test-clips")


if __name__ == "__main__":
    main()
