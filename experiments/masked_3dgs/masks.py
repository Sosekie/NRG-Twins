"""NRG-PD masks for masked 3D Gaussian Splatting (paper Sec. 5.2, Supp. G).

Mask rule. With theta(t, p) the NRG angle predicted by NRG-PD for frame t and pixel p (degrees, on the
518x294 model grid), a pixel is masked when its angle exceeds its own median over the clip by at least
delta = 0.06 deg:

    masked(t, p)  <=>  theta(t, p) - median_s theta(s, p) >= delta      (tested as >= delta - 1e-6)

No temporal max, dilation or frame gating. The binary mask is brought to the 848x480 video by
nearest-neighbour resizing.

Random control: in every frame, the same number of pixels as the NRG-PD mask, drawn uniformly at random
(numpy.random.default_rng(seed).permutation over the 848*480 pixels, frames in order). In the paper one
generator (seed 2026111606) was shared by all clips and both injection conditions, in manifest order;
here each call takes its own generator so a single clip can be reproduced only when called in that order.

How masks enter RobustSplat. Masked pixels are left out of the photometric loss. For each training frame
t the fit reads a keep map clutter_<t:03d>.png = 1 - mask (uint8, 480x848) from one .npz archive
(write_keep_archive below); the trainer multiplies both terms of the 3DGS loss by it,
L1 = mean(keep * |I - I_gt|) and SSIM computed on (keep * I, keep * I_gt), and RobustSplat's own learned
transient mask is switched off (--disable_mask). Fits run 30,000 iterations at 848x480; held-out frames
are rendered from their cameras. See README.md in this folder for the command.

Input: an NRG-PD prediction .npz from `python -m nrgpd.infer` (key 'angle_deg', (T, 294, 518)); files
holding 'degree_u8' and 'degree_u8_scale' (the 0.02-deg quantised maps used for the paper) are also read.

Example:
    python -m experiments.masked_3dgs.masks pred.npz --out masks.npz --keep-out external_keep.npz
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

DELTA_DEG = 0.06
OUT_HW = (480, 848)
RANDOM_SEED = 2026111606


def load_angle(path: str) -> np.ndarray:
    """Predicted NRG angle in degrees, float32 (T, H, W)."""
    with np.load(path, allow_pickle=False) as z:
        if "degree_u8" in z.files:
            return z["degree_u8"].astype(np.float32) * float(z["degree_u8_scale"])
        return z["angle_deg"].astype(np.float32)


def nrg_mask(angle_deg: np.ndarray, delta: float = DELTA_DEG, out_hw: tuple[int, int] = OUT_HW) -> np.ndarray:
    """Boolean mask (T, H_out, W_out): angle minus its per-pixel median over the clip >= delta."""
    rel = angle_deg - np.median(angle_deg, axis=0, keepdims=True)
    m = (rel >= delta - 1e-6).astype(np.uint8)
    h, w = out_hw
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_NEAREST) for f in m]).astype(bool)


def random_control(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per frame, the same number of pixels as `mask`, drawn uniformly at random."""
    t, h, w = mask.shape
    n_t = mask.reshape(t, -1).sum(1)
    out = np.zeros(mask.shape, dtype=bool)
    for i in range(t):
        n = int(n_t[i])
        if n:
            out[i].reshape(-1)[rng.permutation(h * w)[:n]] = True
    return out


def write_keep_archive(mask: np.ndarray, frames: list[int], path: str) -> None:
    """RobustSplat external keep maps: clutter_<t:03d>.png = 1 - mask[t] for each training frame t."""
    np.savez_compressed(path, **{f"clutter_{int(t):03d}.png": (~mask[t]).astype(np.uint8) for t in frames})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prediction", help="NRG-PD prediction .npz")
    ap.add_argument("--out", required=True, help="output .npz with key 'mask' (T, 480, 848) uint8")
    ap.add_argument("--delta", type=float, default=DELTA_DEG)
    ap.add_argument("--random-out", default=None, help="also write the equal-area random control here")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--keep-out", default=None, help="also write the RobustSplat keep archive here")
    ap.add_argument("--train-frames", type=int, nargs="*", default=None,
                    help="frames that the 3DGS fit trains on (default: all)")
    args = ap.parse_args()
    mask = nrg_mask(load_angle(args.prediction), args.delta)
    t = mask.shape[0]
    np.savez_compressed(args.out, mask=mask.astype(np.uint8), frame_indices=np.arange(t, dtype=np.int16))
    if args.random_out:
        rnd = random_control(mask, np.random.default_rng(args.seed))
        np.savez_compressed(args.random_out, mask=rnd.astype(np.uint8), frame_indices=np.arange(t, dtype=np.int16))
    if args.keep_out:
        write_keep_archive(mask, args.train_frames if args.train_frames is not None else list(range(t)), args.keep_out)
    print(json.dumps({"frames": t, "masked_area_share": float(mask.mean()), "delta_deg": args.delta}))


if __name__ == "__main__":
    main()
