"""Training and validation samples from a local copy of the NRG-Twins dataset.

Target (paper Sec. 4): T = (min(theta, 3)^2 - sigma_hat^2) / 0.6^2 on labelled non-keyframe pixels,
where theta is the raw NRG angle and sigma_hat^2 the per-frame keyframe noise. The ordinal labels
are theta_c >= edge for the edges 0.03, 0.075, 0.15, 0.3, 0.6 and 1.2 deg, with
theta_c = sqrt(max(min(theta, 3)^2 - sigma_hat^2, 0)). Labels are resized to the 294x518 model grid
by nearest neighbour.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .model import ORDINAL_EDGES_DEG, UNIT_DEG

GRID = (294, 518)
WINDOW = 24
CAP_DEG = 3.0


def read_sources(root: Path) -> dict[str, dict]:
    out = {}
    with open(root / "metadata" / "sources.csv", newline="") as f:
        for row in csv.DictReader(f):
            out[row["source_id"]] = {
                "split": row["split"],
                "sigma2": np.asarray(json.loads(row["sigma_hat_squared_deg2"]), np.float32),
                "keyframes": set(int(k) for k in json.loads(row["keyframe_indices"])),
            }
    return out


def split_ids(root: Path, split: str) -> list[str]:
    return [x.strip() for x in (root / "splits" / f"{split}.txt").read_text().splitlines() if x.strip()]


@lru_cache(maxsize=8)
def _labels(path: str):
    with np.load(path, allow_pickle=False) as z:
        theta = z["theta_deg"].astype(np.float32)
        valid = np.unpackbits(z["valid_mask_packed"], axis=-1, count=theta.shape[-1], bitorder="little").astype(bool)
        keyframe = z["is_keyframe"].astype(bool)
    return theta, valid, keyframe


def read_frames(video: Path, indices) -> np.ndarray:
    """Decode the requested frames of an 848x480 twin video and resize to 518x294 (INTER_AREA)."""
    wanted = sorted(set(int(i) for i in indices))
    cap = cv2.VideoCapture(str(video))
    got, i = {}, 0
    while i <= wanted[-1]:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            got[i] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (GRID[1], GRID[0]), interpolation=cv2.INTER_AREA)
        i += 1
    cap.release()
    if len(got) != len(wanted):
        raise RuntimeError(f"could not decode all frames of {video}")
    return np.stack([got[int(i)] for i in indices])


def frame_targets(root: Path, split: str, sid: str, frames, sigma2: np.ndarray, keyframes: set[int]):
    theta_all, valid_all, _ = _labels(str(root / split / sid / "labels.npz"))
    target, ordinal, mask, theta_c_out = [], [], [], []
    for f in frames:
        f = int(f)
        theta = cv2.resize(np.nan_to_num(theta_all[f], nan=0.0), (GRID[1], GRID[0]), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid_all[f].astype(np.uint8), (GRID[1], GRID[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        if f in keyframes:
            valid[:] = False
        clipped = np.minimum(theta, CAP_DEG)
        t = (clipped * clipped - float(sigma2[f])) / (UNIT_DEG * UNIT_DEG)
        tc = np.sqrt(np.maximum(clipped * clipped - float(sigma2[f]), 0.0))
        target.append(t.astype(np.float32))
        ordinal.append(np.stack([tc >= e for e in ORDINAL_EDGES_DEG]).astype(np.float32))
        mask.append(valid)
        theta_c_out.append(tc.astype(np.float32))
    return np.stack(target), np.stack(ordinal), np.stack(mask), np.stack(theta_c_out)


def sample_window(n_frames: int, seed: int, step: int, attempt: int = 0):
    """24 frames with random start, temporal stride 1-3, reversed in time with probability 1/2."""
    parts = [seed, step] if attempt == 0 else [seed, step, attempt]
    rng = np.random.default_rng(np.random.SeedSequence(parts))
    stride = int(rng.integers(1, 4))
    start = int(rng.integers(0, n_frames - 1 - (WINDOW - 1) * stride + 1))
    idx = start + stride * np.arange(WINDOW, dtype=np.int64)
    if rng.random() < 0.5:
        idx = idx[::-1].copy()
    return idx


def training_sample(root: Path, ids: list[str], sources: dict, seed: int, step: int, attempt: int = 0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(step), 35, int(attempt)]))
    sid = ids[int(rng.integers(0, len(ids)))]
    idx = sample_window(81, seed, step, attempt)
    src = sources[sid]
    target, ordinal, mask, _ = frame_targets(root, "train", sid, idx, src["sigma2"], src["keyframes"])
    if not mask.any():
        return None
    frames = read_frames(root / "train" / sid / "twin.mp4", idx)
    return {"id": sid, "frames": frames, "target": target, "ordinal": ordinal, "mask": mask}
