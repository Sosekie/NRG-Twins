"""Keyframe noise estimate, noise-subtracted angle and twin exclusion (paper Sec. 3.6).

Keyframes of a twin are generated to match the real frame, so their NRG
angles measure the noise of the label pipeline itself.

* On each keyframe k with labelled pixels, σ̂_k² is the mean of
  min(|θ|, 3°)² over its labelled pixels.
* σ̂²(t) for every frame is the linear interpolation between the nearest
  keyframes before and after t (a keyframe keeps its own value; frames
  outside the keyframe range take the nearest keyframe's value).
* θ_c = sqrt(max(min(θ, 3°)² − σ̂²(t), 0)) on labelled pixels.
* A twin is excluded if its pooled keyframe RMS angle exceeds 0.6°, or its
  noise-subtracted keyframe RMS exceeds 0.3°.  For the latter each keyframe
  is compared against σ̂² interpolated from the *other* keyframes (its own
  value would subtract to zero by construction).
"""
from __future__ import annotations

import math

import numpy as np

ANGLE_CAP_DEG = 3.0
#: Lower-inclusive ordinal bin edges in degrees (bins 0..6; 255 = no label).
ORDINAL_EDGES = np.array([0.03, 0.075, 0.15, 0.3, 0.6, 1.2], dtype=np.float64)
NO_LABEL = 255
MAX_RAW_RMS_DEG = 0.6
MAX_SUBTRACTED_RMS_DEG = 0.3


def capped_squares(theta: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """min(|θ|, 3°)² of the labelled pixels, float64."""
    values = np.asarray(theta)[np.asarray(valid, bool)].astype(np.float64)
    return np.minimum(np.abs(values), ANGLE_CAP_DEG) ** 2


def keyframe_noise(theta_maps, valid_masks, keyframes) -> dict[int, float]:
    """σ̂_k² for every keyframe with at least one labelled pixel."""
    out = {}
    for k in sorted(set(int(i) for i in keyframes)):
        sq = capped_squares(theta_maps[k], valid_masks[k])
        if sq.size:
            out[k] = float(np.mean(sq))
    return out


def interpolate_sigma2(keyframe_sigma2: dict, n_frames: int) -> list:
    """Per-frame σ̂² by linear interpolation in frame index between keyframes.

    A keyframe keeps its own value; frames before the first or after the last
    keyframe take the nearest keyframe's value.  Frames are None only if there
    is no keyframe value at all.
    """
    keys = {int(k): float(v) for k, v in keyframe_sigma2.items()}
    if any(k < 0 or k >= n_frames for k in keys):
        raise ValueError("keyframe index outside frame range")
    result = []
    for frame in range(n_frames):
        if frame in keys:
            result.append(keys[frame])
            continue
        left = [k for k in keys if k < frame]
        right = [k for k in keys if k > frame]
        if left and right:
            a, b = max(left), min(right)
            t = (frame - a) / (b - a)
            result.append((1.0 - t) * keys[a] + t * keys[b])
        elif left or right:
            nearest = min(left or right, key=lambda k: (abs(k - frame), k))
            result.append(keys[nearest])
        else:
            result.append(None)
    return result


def corrected_angle(theta: np.ndarray, valid: np.ndarray, sigma2: float):
    """θ_c (float16, NaN where unlabelled) and its ordinal bin (uint8, 255 where unlabelled).

    The bin is taken from θ_c before float16 rounding.
    """
    valid = np.asarray(valid, bool)
    c = np.sqrt(np.maximum(np.minimum(np.asarray(theta)[valid].astype(np.float64), ANGLE_CAP_DEG) ** 2
                           - float(sigma2), 0.0))
    corrected = np.full(theta.shape, np.nan, dtype=np.float16)
    ordinal = np.full(theta.shape, NO_LABEL, dtype=np.uint8)
    corrected[valid] = c.astype(np.float16)
    ordinal[valid] = np.searchsorted(ORDINAL_EDGES, c, side="right").astype(np.uint8)
    return corrected, ordinal


def _leave_one_out(stats: list[dict]) -> None:
    for x in stats:
        others = [o for o in stats if o is not x]
        if not others:
            x["sigma2_others"] = None
            continue
        left = [o for o in others if o["frame"] < x["frame"]]
        right = [o for o in others if o["frame"] > x["frame"]]
        if left and right:
            a = max(left, key=lambda o: o["frame"])
            b = min(right, key=lambda o: o["frame"])
            t = (x["frame"] - a["frame"]) / (b["frame"] - a["frame"])
            x["sigma2_others"] = (1 - t) * a["mean_sq"] + t * b["mean_sq"]
        else:
            x["sigma2_others"] = min(others, key=lambda o: (abs(o["frame"] - x["frame"]), o["frame"]))["mean_sq"]


def keyframe_rms(theta_maps, valid_masks, keyframes) -> dict:
    """Pooled raw and noise-subtracted keyframe RMS angles of one twin (degrees).

    Either value is None when it cannot be computed (no labelled keyframe
    pixels, or only one labelled keyframe for the subtracted value).
    """
    stats = []
    for k in sorted(set(int(i) for i in keyframes)):
        sq = capped_squares(theta_maps[k], valid_masks[k])
        if sq.size:
            stats.append({"frame": k, "n": int(sq.size), "sum_sq": float(sq.sum()), "mean_sq": float(sq.mean())})
    _leave_one_out(stats)
    raw = subtracted = None
    if stats:
        n = sum(x["n"] for x in stats)
        raw = math.sqrt(sum(x["sum_sq"] for x in stats) / n)
        if all(x["sigma2_others"] is not None for x in stats):
            subtracted = math.sqrt(max(sum(x["sum_sq"] - x["n"] * x["sigma2_others"] for x in stats) / n, 0))
    return {"raw_rms_deg": raw, "subtracted_rms_deg": subtracted, "keyframes": stats}


def twin_exclusion(raw_rms, subtracted_rms) -> str | None:
    """Exclusion reason for a twin, or None if it is kept."""
    if raw_rms is None:
        return "raw_rms_uncheckable"
    if raw_rms > MAX_RAW_RMS_DEG:
        return "raw_rms_gt_0.6_deg"
    if subtracted_rms is None:
        return "subtracted_rms_uncheckable"
    if subtracted_rms > MAX_SUBTRACTED_RMS_DEG:
        return "subtracted_rms_gt_0.3_deg"
    return None
