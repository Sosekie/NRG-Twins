"""Camera handling for the per-pixel labels (paper Sec. 3.2).

Two steps are implemented here:

* ``align_twin_to_real`` expresses the twin camera trajectory in the coordinate
  frame of the real trajectory with a similarity transform fitted on the
  fitting frames (all frames except the held-out probe frames).
* ``select_reference`` picks, for one twin frame, the real fitting frame used as
  the reference.  The released labels use the real frame whose camera gives the
  smallest median lateral-baseline-to-depth ratio t_perp / Z over the valid
  pixels of its fused depth map (see ``lateral_baseline_over_depth``).

Poses are camera-to-world 4x4 matrices (OpenCV convention: x right, y down,
z forward).
"""
from __future__ import annotations

import numpy as np

#: Frames held out of all fitting in the released twins (one every ten frames).
PROBE_FRAMES = (5, 15, 25, 35, 45, 55, 65, 75)


def fit_frames(n_frames: int = 81, probe_frames=PROBE_FRAMES) -> tuple[int, ...]:
    """Return the indices of the fitting frames (all frames except the probes)."""
    probes = set(int(i) for i in probe_frames)
    return tuple(i for i in range(int(n_frames)) if i not in probes)


def align_twin_to_real(real_c2w: np.ndarray, twin_c2w: np.ndarray, frames) -> tuple[np.ndarray, dict]:
    """Map the twin trajectory into the real coordinate frame.

    The rotation is the SO(3) projection of the mean of ``R_real_i @ R_twin_i.T``
    over the fitting frames.  Scale and translation are then the least-squares
    fit of the rotated twin camera centres to the real camera centres.

    Returns the aligned twin poses for all frames and the fitted parameters.
    """
    frames = list(frames)
    real_c2w = np.asarray(real_c2w, np.float64)
    twin_c2w = np.asarray(twin_c2w, np.float64)
    rr, rc = real_c2w[frames, :3, :3], real_c2w[frames, :3, 3]
    tr, tc = twin_c2w[frames, :3, :3], twin_c2w[frames, :3, 3]
    mean_rot = np.mean(np.stack([rr[i] @ tr[i].T for i in range(len(frames))]), axis=0)
    u, _, vh = np.linalg.svd(mean_rot)
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vh)
    rot = u @ d @ vh
    x = (rot @ tc.T).T
    xm, ym = x.mean(0), rc.mean(0)
    den = float(np.sum((x - xm) ** 2))
    if den <= 0:
        raise ValueError("degenerate twin camera centres")
    scale = float(np.sum((x - xm) * (rc - ym)) / den)
    trans = ym - scale * xm
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid twin alignment scale")
    aligned = twin_c2w.copy()
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rot, aligned[:, :3, :3])
    aligned[:, :3, 3] = scale * np.einsum("ij,nj->ni", rot, aligned[:, :3, 3]) + trans
    return aligned, {"rotation": rot.tolist(), "scale": scale, "translation": trans.tolist(),
                     "fit_frame_count": len(frames)}


def unit_rays(K: np.ndarray, height: int, width: int) -> np.ndarray:
    """Unit viewing rays of every pixel of the reference grid, shape (H, W, 3).

    As in the released code, both image axes are normalised by ``K[0, 0]``.
    """
    f, cx, cy = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
    yy, xx = np.mgrid[0:height, 0:width]
    rays = np.stack(((xx - cx) / f, (yy - cy) / f, np.ones((height, width))), axis=-1)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
    return rays


def lateral_baseline(ref_c2w: np.ndarray, target_c2w: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Per-pixel component of the camera-centre baseline perpendicular to each ray.

    The baseline is the target camera centre expressed in the reference camera
    frame; ``rays`` are the unit rays of the reference grid.
    """
    rel = ref_c2w[:3, :3].T @ (target_c2w[:3, 3] - ref_c2w[:3, 3])
    dot = np.einsum("ijk,k->ij", rays, rel)
    return np.linalg.norm(rel[None, None, :] - dot[..., None] * rays, axis=-1)


def lateral_baseline_over_depth(ref_c2w, target_c2w, K, depth: np.ndarray, rays=None) -> np.ndarray:
    """t_perp / Z on the reference pixel grid (paper Sec. 3.5), float32.

    The depth is clamped from below at 1e-6 times its median valid value.  The
    map is NaN where the reference depth is NaN.
    """
    depth = np.asarray(depth)
    if rays is None:
        rays = unit_rays(K, *depth.shape)
    t_perp = lateral_baseline(np.asarray(ref_c2w, np.float64), np.asarray(target_c2w, np.float64), rays)
    zmed = float(np.median(depth[np.isfinite(depth) & (depth > 0)]))
    return (t_perp / np.maximum(depth, 1e-6 * zmed)).astype(np.float32)


def reference_score(ref_c2w, target_c2w, depth: np.ndarray, rays: np.ndarray) -> float:
    """Median t_perp / Z over the valid depth pixels of one candidate reference."""
    t_perp = lateral_baseline(np.asarray(ref_c2w, np.float64), np.asarray(target_c2w, np.float64), rays)
    valid = np.isfinite(depth) & (depth > 0)
    q = t_perp / np.maximum(depth, 1e-6 * np.median(depth[np.isfinite(depth) & (depth > 0)]))
    valid &= np.isfinite(q)
    return float(np.median(q[valid])) if valid.any() else float("inf")


def select_reference(target_index: int, target_c2w, real_c2w, depths: dict, K, candidates=None) -> tuple[int, float]:
    """Pick the real reference frame for one twin frame (paper Sec. 3.2).

    ``depths`` maps candidate real frame index -> fused depth (H, W).  The
    candidate with the smallest median t_perp / Z wins; ties are broken by the
    smaller frame distance ``|r - t|`` and then by the smaller index.
    """
    candidates = sorted(depths) if candidates is None else list(candidates)
    first = depths[candidates[0]]
    rays = unit_rays(K, *first.shape)
    scored = [(reference_score(real_c2w[r], target_c2w, depths[r], rays), abs(r - int(target_index)), r)
              for r in candidates]
    best = min(scored, key=lambda x: (x[0], x[1], x[2]))
    if not np.isfinite(best[0]):
        raise RuntimeError(f"no usable reference for target frame {target_index}")
    return int(best[2]), float(best[0])


def shared_intrinsics(real_K: np.ndarray, twin_K: np.ndarray) -> np.ndarray:
    """Return the single pinhole K shared by both videos (first real frame)."""
    real_K = np.asarray(real_K, np.float64)
    twin_K = np.asarray(twin_K, np.float64)
    if not np.allclose(real_K, twin_K, rtol=0, atol=1e-7):
        raise ValueError("real and twin intrinsics differ")
    return real_K[0] if real_K.ndim == 3 else real_K
