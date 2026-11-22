"""Point-by-point reprojection of the real reference into the twin camera (paper Sec. 3.4).

Every reference pixel with a valid fused depth is lifted to 3D, transformed
into the aligned twin camera and rounded to the nearest twin pixel.  When
several points land on one pixel the nearest one (smallest camera-axis depth)
wins.  Points outside the image, or behind/too close to the camera, give no
label.  The result is defined on the twin pixel grid: û is the integer pixel
itself.
"""
from __future__ import annotations

import numpy as np


def warp_reference(rgb: np.ndarray, depth: np.ndarray, c2w_src: np.ndarray, c2w_dst: np.ndarray,
                   K_src: np.ndarray, K_dst: np.ndarray):
    """Z-buffered forward warp of a reference image into a target camera.

    Returns ``(warped_rgb, warp_depth, valid, source_index)``: the warped RGB
    (zero where empty), the target camera-axis depth of the winning point (NaN
    where empty), the support mask and, per target pixel, the flat index
    ``y * W + x`` of the winning reference pixel (-1 where empty).
    """
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3) @ np.linalg.inv(K_src).T
    z = np.asarray(depth).reshape(-1).astype(np.float64)
    source_ids = np.flatnonzero(np.isfinite(z) & (z > (1e-5 * np.median(z[np.isfinite(z) & (z > 0)]))))
    out_rgb = np.zeros((h, w, 3), np.uint8)
    out_depth = np.full((h, w), np.nan, np.float32)
    valid = np.zeros((h, w), bool)
    source_index = np.full((h, w), -1, np.int32)
    if source_ids.size == 0:
        return out_rgb, out_depth, valid, source_index
    zv, rv = z[source_ids], rays[source_ids]
    xyz = rv * (zv / np.maximum(rv[:, 2], 1e-12))[:, None]
    world = xyz @ c2w_src[:3, :3].T + c2w_src[:3, 3]
    cam = (world - c2w_dst[:3, 3]) @ c2w_dst[:3, :3]
    zt = cam[:, 2]
    uvw = cam @ K_dst.T
    uf = uvw[:, 0] / np.maximum(uvw[:, 2], 1e-12)
    vf = uvw[:, 1] / np.maximum(uvw[:, 2], 1e-12)
    projected = (np.isfinite(zt) & (zt > (1e-5 * np.median(zt[np.isfinite(zt) & (zt > 0)])))
                 & np.isfinite(uf) & np.isfinite(vf))
    u = np.zeros(len(source_ids), np.int64)
    v = np.zeros(len(source_ids), np.int64)
    u[projected] = np.rint(uf[projected]).astype(np.int64)
    v[projected] = np.rint(vf[projected]).astype(np.int64)
    good = projected & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ids = source_ids[good]
    if ids.size:
        lin = v[good] * w + u[good]
        order = np.lexsort((zt[good], lin))          # by pixel, then nearest first
        sorted_lin = lin[order]
        first = np.r_[True, sorted_lin[1:] != sorted_lin[:-1]]
        take = ids[order[first]]
        pix = lin[order[first]]
        out_rgb.reshape(-1, 3)[pix] = rgb.reshape(-1, 3)[take]
        out_depth.reshape(-1)[pix] = zt[good][order[first]].astype(np.float32)
        valid.reshape(-1)[pix] = True
        source_index.reshape(-1)[pix] = take.astype(np.int32)
    return out_rgb, out_depth, valid, source_index


def warp_support(valid: np.ndarray, warp_depth: np.ndarray) -> np.ndarray:
    """Final support mask: z-buffer hit with depth above 1e-5 of the median warped depth."""
    dw = np.asarray(warp_depth)
    ok = np.isfinite(dw) & (dw > 0)
    if not ok.any():
        return np.zeros(dw.shape, bool)
    return np.asarray(valid, bool) & np.isfinite(dw) & (dw > (1e-5 * np.median(dw[ok])))
