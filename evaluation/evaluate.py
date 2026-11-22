"""Evaluate per-pixel predictions on the NRG-Twins test split.

Each prediction file <pred_dir>/<source_id>.npz holds an array of shape (81, 294, 518): a ranking
score where higher means less rigid (`--score-key`, default `raw`, the NRG-PD regression output),
and optionally the predicted angle in degrees (`--angle-key`, default `angle_deg`) for MAE.
Labels are resized to the 294x518 model grid by nearest neighbour; keyframes are excluded.
The benchmark uses the 83 test twins listed in evaluation/test_benchmark.txt.

Example:
    python -m evaluation.evaluate --data NRG-Twins --pred-dir preds/seed20261108 --out metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .metrics import evaluate

GRID = (294, 518)
HERE = Path(__file__).resolve().parent


def load_labels(root: Path, split: str, sid: str):
    with np.load(root / split / sid / "labels.npz", allow_pickle=False) as z:
        theta_c = z["theta_c_deg"].astype(np.float32)
        valid = np.unpackbits(z["valid_mask_packed"], axis=-1, count=theta_c.shape[-1], bitorder="little").astype(bool)
        keyframe = z["is_keyframe"].astype(bool)
    return theta_c, valid, keyframe


def nearest(a: np.ndarray) -> np.ndarray:
    return cv2.resize(a, (GRID[1], GRID[0]), interpolation=cv2.INTER_NEAREST)


def clip_arrays(root, split, sid, sigma_hat, score, angle):
    theta_c, valid, keyframe = load_labels(root, split, sid)
    ys, ss, gs, an = [], [], [], []
    for t in range(theta_c.shape[0]):
        if keyframe[t]:
            continue
        y = nearest(theta_c[t])
        v = nearest(valid[t].astype(np.uint8)).astype(bool) & np.isfinite(y) & np.isfinite(score[t])
        if not v.any():
            continue
        ys.append(y[v]); ss.append(score[t][v]); gs.append(np.full(int(v.sum()), sigma_hat[t], np.float32))
        if angle is not None:
            an.append(angle[t][v])
    clip = {"id": sid, "theta_c": np.concatenate(ys), "score": np.concatenate(ss).astype(np.float64),
            "sigma_hat": np.concatenate(gs)}
    if angle is not None:
        clip["angle_deg"] = np.concatenate(an).astype(np.float64)
    return clip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="local copy of the NRG-Twins dataset")
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", type=Path, default=HERE / "test_benchmark.txt")
    ap.add_argument("--score-key", default="raw")
    ap.add_argument("--angle-key", default="angle_deg", help="set to '' to skip MAE")
    ap.add_argument("--raw-score", action="store_true", default=True,
                    help="the score is the NRG-PD regression output; derive degrees for MAE if no angle is stored")
    ap.add_argument("--not-raw", dest="raw_score", action="store_false", help="score is not NRG-PD output (baselines)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ids = [x.strip() for x in args.ids.read_text().splitlines() if x.strip()]
    sigma = {}
    with open(args.data / "metadata" / "sources.csv", newline="") as f:
        for row in csv.DictReader(f):
            sigma[row["source_id"]] = np.asarray(json.loads(row["sigma_hat_deg"]), np.float32)
    clips = []
    for sid in ids:
        with np.load(args.pred_dir / f"{sid}.npz", allow_pickle=False) as z:
            score = z[args.score_key].astype(np.float32)
            angle = z[args.angle_key].astype(np.float32) if args.angle_key and args.angle_key in z.files else None
        if angle is None and args.raw_score and args.angle_key:
            angle = (0.6 * np.sqrt(np.maximum(score, 0))).astype(np.float32)
        clips.append(clip_arrays(args.data, args.split, sid, sigma[sid], score, angle))
        print(f"loaded {sid} ({len(clips)}/{len(ids)})", flush=True)
    result = evaluate(clips)
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
