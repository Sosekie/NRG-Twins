"""NRG angle and label validity (paper Sec. 3.5).

The NRG angle at a twin pixel û is the angle between the viewing rays
K⁻¹[û, 1] and K⁻¹[û + Δu, 1], where Δu is the consensus flow from the
projected reference to the twin frame.  It is stored at û.

A pixel carries a label only if it is inside the warp support, has an
accepted flow consensus, a finite angle, and a finite lateral-baseline-to-depth
ratio t_perp / Z of at most 0.01.  ``exclusion_reason`` records the first
failing check in that order (code 0 = labelled).
"""
from __future__ import annotations

import numpy as np

REASON_VALID = 0
REASON_OUTSIDE_WARP = 1        # no reprojected reference point at this pixel
REASON_NO_CONSENSUS = 2        # flow consensus rejected
REASON_NONFINITE_ANGLE = 3
REASON_NONFINITE_BASELINE = 4  # t_perp / Z not finite
REASON_BASELINE_TOO_LARGE = 5  # t_perp / Z > 0.01

#: Upper bound on t_perp / Z for a labelled pixel.
MAX_BASELINE_OVER_DEPTH = 0.01


def ray_angle_deg(flow: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Angle in degrees between the rays of each pixel and of its flow endpoint (float32)."""
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    p0 = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3).astype(np.float64)
    p1 = p0.copy()
    p1[:, 0] += flow[..., 0].reshape(-1)
    p1[:, 1] += flow[..., 1].reshape(-1)
    inv = np.linalg.inv(K)
    r0, r1 = p0 @ inv.T, p1 @ inv.T
    r0 /= np.maximum(np.linalg.norm(r0, axis=1, keepdims=True), 1e-12)
    r1 /= np.maximum(np.linalg.norm(r1, axis=1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(r0 * r1, axis=1), -1, 1))).reshape(h, w).astype(np.float32)


def nrg_angle(consensus_flow: np.ndarray, accepted: np.ndarray, support: np.ndarray, K: np.ndarray):
    """Angle map (NaN where not accepted) and the final acceptance mask.

    The acceptance mask is the flow consensus restricted to the warp support
    and to finite angles.
    """
    theta = np.full(accepted.shape, np.nan, np.float32)
    if accepted.any():
        theta[accepted] = ray_angle_deg(consensus_flow, K)[accepted]
    final = np.asarray(accepted, bool) & np.asarray(support, bool) & np.isfinite(theta)
    return theta, final


def exclusion_reason(theta: np.ndarray, accepted: np.ndarray, warp_valid: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-pixel reason code (uint8); 0 means the pixel is labelled.

    ``q`` is the t_perp / Z map.  As in the released labels it is taken at the
    same array index as the angle (it is computed on the reference grid).
    """
    reason = np.zeros(theta.shape, dtype=np.uint8)
    reason[q > MAX_BASELINE_OVER_DEPTH] = REASON_BASELINE_TOO_LARGE
    reason[~np.isfinite(q)] = REASON_NONFINITE_BASELINE
    reason[~np.isfinite(theta)] = REASON_NONFINITE_ANGLE
    reason[~accepted] = REASON_NO_CONSENSUS
    reason[~warp_valid] = REASON_OUTSIDE_WARP
    return reason
