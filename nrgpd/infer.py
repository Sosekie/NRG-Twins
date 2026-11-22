"""Run NRG-PD on a video and save the per-pixel predicted NRG angle.

Frames are centre-cropped to the 848:480 aspect ratio, resized to 848x480 and then to the
model resolution 518x294 (INTER_AREA), as in the paper. A clip of T >= 24 frames is covered by
24-frame windows whose starts are evenly spaced (stride about 19 frames); each frame takes its
prediction from exactly one window, split at the middle of each overlap. For T = 81 this gives the
windows starting at 0, 19, 38, 57 and the frame ranges 0-20, 21-39, 40-58, 59-80 used for all
numbers in the paper.

Example:
    python -m nrgpd.infer video.mp4 --weights weights/nrgpd_seed20261108.safetensors --out pred.npz
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import NRGPD, ORDINAL_EDGES_DEG, raw_to_degrees

WINDOW = 24
SOURCE_HW = (480, 848)
MODEL_HW = (294, 518)


def read_video(path: str | Path, max_frames: int | None = 81) -> np.ndarray:
    """Return uint8 RGB frames (T, 294, 518, 3)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    h, w = frames[0].shape[:2]
    H, W = SOURCE_HW
    if w / h > W / H + 1e-9:
        cw, ch = int(round(h * W / H)), h
    elif w / h < W / H - 1e-9:
        cw, ch = w, int(round(w * H / W))
    else:
        cw, ch = w, h
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    out = np.empty((len(frames), *MODEL_HW, 3), np.uint8)
    for i, bgr in enumerate(frames):
        crop = bgr[y0:y0 + ch, x0:x0 + cw]
        if (ch, cw) != (H, W):
            crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA if (cw > W or ch > H) else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out[i] = cv2.resize(rgb, (MODEL_HW[1], MODEL_HW[0]), interpolation=cv2.INTER_AREA)
    return out


def windows(n_frames: int) -> list[tuple[int, int, int]]:
    """(start, first_scored_frame, last_scored_frame) for each 24-frame window."""
    if n_frames < WINDOW:
        raise ValueError(f"NRG-PD needs at least {WINDOW} frames, got {n_frames}")
    n = math.ceil((n_frames - WINDOW) / 19) + 1
    starts = [int(round(x)) for x in np.linspace(0, n_frames - WINDOW, n)] if n > 1 else [0]
    out = []
    for k, s in enumerate(starts):
        first = 0 if k == 0 else (s + starts[k - 1] + WINDOW - 1) // 2
        last = n_frames - 1 if k == len(starts) - 1 else (starts[k + 1] + s + WINDOW - 1) // 2 - 1
        out.append((s, first, last))
    return out


@torch.inference_mode()
def predict(model: NRGPD, frames: np.ndarray, device: str = "cuda") -> dict[str, np.ndarray]:
    T = len(frames)
    raw = np.empty((T, *MODEL_HW), np.float32)
    prob = np.empty((T, *MODEL_HW, len(ORDINAL_EDGES_DEG)), np.float32)
    for start, first, last in windows(T):
        x = torch.from_numpy(frames[start:start + WINDOW]).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            out = model(x)[0].float().cpu().numpy()
        a, b = first - start, last - start + 1
        raw[first:last + 1] = out[a:b, 0]
        p = 1.0 / (1.0 + np.exp(-np.clip(out[a:b, 1:], -40, 40)))
        prob[first:last + 1] = np.moveaxis(p, 1, -1)
    return {"raw": raw, "angle_deg": raw_to_degrees(raw).astype(np.float32), "ordinal_prob": prob}


def load_model(weights: str | Path, vggt_weights: str | Path | None = None, device: str = "cuda") -> NRGPD:
    model = NRGPD(vggt_weights=vggt_weights, pretrained_init=False)
    model.load_trainable(weights)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--weights", required=True, help="NRG-PD weight file (.safetensors)")
    ap.add_argument("--vggt-weights", default=None, help="local VGGT-1B model.safetensors; downloaded if omitted")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--max-frames", type=int, default=81)
    ap.add_argument("--save-ordinal", action="store_true", help="also save the ordinal probabilities")
    args = ap.parse_args()
    frames = read_video(args.video, args.max_frames)
    model = load_model(args.weights, args.vggt_weights)
    pred = predict(model, frames)
    keep = {"angle_deg": pred["angle_deg"].astype(np.float16), "raw": pred["raw"]}
    if args.save_ordinal:
        keep["ordinal_prob"] = pred["ordinal_prob"].astype(np.float16)
    np.savez_compressed(args.out, **keep, ordinal_edges_deg=np.asarray(ORDINAL_EDGES_DEG, np.float32))
    print(f"saved {args.out}: {len(frames)} frames, median angle {np.median(pred['angle_deg']):.3f} deg")


if __name__ == "__main__":
    main()
