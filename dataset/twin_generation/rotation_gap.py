"""Rotation gap between a real clip and its twin, and the keyframe-insertion rule (paper Sec. 3.1, Supp. A.1).

After each generation round, VGGT estimates one camera per frame for the real clip and for the twin.
With R_t the camera-to-world rotation of frame t, the twin is aligned to the real clip by the single
rotation Q that best maps twin to real orientations on the current keyframes K (Wahba's problem, closed
form by SVD):

    Q = argmin_{Q in SO(3)} sum_{k in K} || R^real_k - Q R^twin_k ||_F^2,

and the rotation gap of frame t is the geodesic angle

    delta_t = angle( (R^real_t)^T Q R^twin_t )   (degrees).

The loop stops when max_t delta_t <= 0.8 deg. Otherwise one real frame is inserted as a new keyframe:
among the interior frames whose index is a multiple of 4 (4, 8, ..., 76) and that are at least 8 frames
from every current keyframe, the one closest to the frame with the largest gap (ties: lower index).
At most 6 insertions are made.

Pure NumPy; no GPU needed. Example:
    python -m dataset.twin_generation.rotation_gap real_cameras.npz twin_cameras.npz --keyframes 0 80
"""
from __future__ import annotations

import argparse
import json

import numpy as np

ROTATION_THRESHOLD_DEG = 0.8
MAX_INSERTIONS = 6
MIN_KEYFRAME_GAP = 8
GRID = 4
FRAME_COUNT = 81
INITIAL_KEYFRAMES = (0, 80)


def camera_to_world(w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotations (N, 3, 3) and centres (N, 3) of camera-to-world poses from world-to-camera (N, 3|4, 4)."""
    r_w2c = np.asarray(w2c[:, :3, :3], dtype=np.float64)
    t_w2c = np.asarray(w2c[:, :3, 3], dtype=np.float64)
    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    centres = -np.einsum("nij,nj->ni", r_c2w, t_w2c)
    return r_c2w, centres


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame rotation angle of a_t^T b_t in degrees."""
    rel = np.einsum("nji,njk->nik", a, b)
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def fit_rotation(r_real: np.ndarray, r_twin: np.ndarray, keyframes: list[int]) -> np.ndarray:
    """Closed-form Wahba solution Q in SO(3) maximising sum_k tr(R^real_k^T Q R^twin_k)."""
    m = np.zeros((3, 3), dtype=np.float64)
    for i in keyframes:
        m += r_real[i] @ r_twin[i].T
    u, _, vt = np.linalg.svd(m)
    d = np.eye(3)
    d[2, 2] = np.linalg.det(u @ vt)
    return u @ d @ vt


def rotation_gap(real_w2c: np.ndarray, twin_w2c: np.ndarray, keyframes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_t in degrees for every frame, Q)."""
    r_real, _ = camera_to_world(real_w2c)
    r_twin, _ = camera_to_world(twin_w2c)
    q = fit_rotation(r_real, r_twin, keyframes)
    aligned = np.einsum("ij,njk->nik", q, r_twin)
    return geodesic_deg(r_real, aligned), q


def choose_keyframe(max_frame: int, keyframes: list[int], frame_count: int = FRAME_COUNT) -> int | None:
    """Interior multiple of 4 nearest to `max_frame`, at least 8 frames from every keyframe; ties -> lower index."""
    legal = [i for i in range(GRID, frame_count - GRID, GRID)
             if min(abs(i - k) for k in keyframes) >= MIN_KEYFRAME_GAP]
    if not legal:
        return None
    return min(legal, key=lambda i: (abs(i - max_frame), i))


def decide(delta: np.ndarray, keyframes: list[int], round_index: int) -> dict:
    """Stop/insert decision after generation round `round_index` (0 = initial two-keyframe twin)."""
    max_frame = int(np.argmax(delta))
    out = {"round": round_index, "keyframes": list(keyframes), "max_gap_deg": float(np.max(delta)),
           "median_gap_deg": float(np.median(delta)), "max_gap_frame": max_frame,
           "next_keyframe": None, "stop_reason": None}
    if out["max_gap_deg"] <= ROTATION_THRESHOLD_DEG:
        out["stop_reason"] = "rotation_threshold_met"
    elif round_index >= MAX_INSERTIONS:
        out["stop_reason"] = "max_insertions_reached"
    else:
        nxt = choose_keyframe(max_frame, keyframes, len(delta))
        if nxt is None:
            out["stop_reason"] = "no_admissible_keyframe"
        else:
            out["next_keyframe"] = nxt
    return out


def load_w2c(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        key = "w2c_vggt" if "w2c_vggt" in z.files else "w2c"
        return np.array(z[key], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("real_cameras", help="npz with world-to-camera poses 'w2c_vggt' (N,4,4) or 'w2c' (N,3|4,4)")
    ap.add_argument("twin_cameras", help="same format, twin clip")
    ap.add_argument("--keyframes", type=int, nargs="+", default=list(INITIAL_KEYFRAMES))
    ap.add_argument("--round", type=int, default=0, help="generation round index (0 = initial twin)")
    args = ap.parse_args()
    delta, q = rotation_gap(load_w2c(args.real_cameras), load_w2c(args.twin_cameras), sorted(args.keyframes))
    out = decide(delta, sorted(args.keyframes), args.round)
    out["Q"] = q.tolist()
    out["gap_deg"] = [float(x) for x in delta]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
