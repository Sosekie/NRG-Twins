"""Fused reference depth (paper Sec. 3.3).

Each of seven depth estimates of a real frame is brought to the scale of the
real camera reconstruction with one factor per frame and method: the median of
z_k / Z_m(u_k) over the sparse reconstruction points k observed in that frame,
where z_k is the point's camera-axis depth and u_k its observed keypoint
(rounded to the nearest pixel).  The fused depth is the per-pixel median of the
seven scaled maps.  A pixel is kept only where all seven maps are valid; it is
NaN otherwise.

The seven estimates are six monocular/multi-view models (Depth Anything V2,
Depth Anything 3, MapAnything, Metric3D v2, MoGe-2, UniDepth v2), stored at the
video resolution, and the VGGT depth head, stored on VGGT's 518x518 crop and
resampled to the video resolution here.

The command-line entry point writes ``fused_depth.npz`` for the fitting frames
of one real video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

#: Names of the six depth models whose maps are stored at video resolution.
MONOCULAR_METHODS = ("da2_metric_hypersim_large", "da3_base_multiview", "mapanything_apache",
                     "metric3d_v2_vit_giant2", "moge2_vitl", "unidepth_v2_vitl14")
VGGT_SIZE = 518        # side of VGGT's square depth map
VGGT_CANVAS = 1024     # side of the padded square canvas ``original_coords`` refers to

COLMAP_LABEL = "COLMAP+BA"
VGGT_LABEL = "VGGT+BA_single"


# ---------------------------------------------------------------------------
# Sparse points
# ---------------------------------------------------------------------------

def _frame_index(name: str) -> int:
    stem = Path(name).stem
    token = stem if stem.isdigit() else stem.rsplit("_", 1)[-1]
    if not token.isdigit():
        raise ValueError(f"cannot read a frame index from image name {name!r}")
    return int(token)


def sparse_observations(model_dir, height: int, width: int, n_frames: int = 81) -> dict[int, list]:
    """Read (x, y, z) keypoint observations per frame from a COLMAP model.

    x, y are the observed keypoint coordinates and z the camera-axis depth of
    the observed 3D point in that image.  Requires ``pycolmap``.
    """
    import pycolmap

    recon = pycolmap.Reconstruction(str(model_dir))
    out: dict[int, list] = {}
    for image in recon.images.values():
        fi = _frame_index(image.name)
        cfw = image.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        T = cfw.matrix()
        pts = []
        for p2 in image.points2D:
            if not p2.has_point3D() or p2.point3D_id not in recon.points3D:
                continue
            xyz = recon.points3D[p2.point3D_id].xyz
            z = float((T[:3, :3] @ xyz + T[:3, 3])[2])
            x, y = map(float, p2.xy)
            if 0 <= fi < n_frames and np.isfinite(z) and z > 0 and 0 <= x < width and 0 <= y < height:
                pts.append((x, y, z))
        out[fi] = pts
    return out


def anchor_labels(frame_sources) -> list[str]:
    """Per-frame anchor source: COLMAP+BA where the selected camera came from COLMAP."""
    return [COLMAP_LABEL if str(x).lower() == "colmap_ba" else VGGT_LABEL for x in frame_sources]


def anchor_points(labels, vggt_obs: dict, colmap_obs: dict | None = None, colmap_scale: float | None = None) -> dict:
    """Choose the sparse points used to scale each frame.

    Frames whose camera was taken from the COLMAP reconstruction use COLMAP's
    points, with depths multiplied by the COLMAP-to-VGGT similarity scale;
    all other frames use the VGGT+BA points.
    """
    col = {}
    if colmap_obs is not None:
        if colmap_scale is None or not np.isfinite(colmap_scale) or colmap_scale <= 0:
            raise ValueError("a positive COLMAP-to-VGGT scale is required with COLMAP points")
        col = {fi: [(x, y, z * float(colmap_scale)) for x, y, z in pts] for fi, pts in colmap_obs.items()}
    return {fi: (col.get(fi, []) if labels[fi] == COLMAP_LABEL else vggt_obs.get(fi, []))
            for fi in range(len(labels))}


# ---------------------------------------------------------------------------
# Per-frame scale factors
# ---------------------------------------------------------------------------

def _summary(fi, source, vals):
    if not vals:
        return {"frame_index": fi, "anchor_source": source, "anchor_point_count": 0, "scale_median": None,
                "scale_ratio_p10": None, "scale_ratio_p90": None}
    p10, p50, p90 = np.percentile(np.asarray(vals, np.float64), [10, 50, 90])
    return {"frame_index": fi, "anchor_source": source, "anchor_point_count": len(vals),
            "scale_ratio_p10": float(p10), "scale_median": float(p50), "scale_ratio_p90": float(p90)}


def _ratios(depth_map: np.ndarray, points) -> list[float]:
    h, w = depth_map.shape
    vals = []
    for x, y, z in points:
        xx, yy = int(round(x)), int(round(y))
        if 0 <= xx < w and 0 <= yy < h:
            v = float(depth_map[yy, xx])
            if np.isfinite(v) and v > 0 and np.isfinite(z) and z > 0:
                vals.append(float(z / v))
    return vals


def fit_scales(depth: np.ndarray, points: dict, labels, fallback: dict | None = None) -> list[dict]:
    """Median ratio z_k / Z_m(u_k) per frame for one video-resolution depth model.

    ``depth`` is (N, H, W).  If a COLMAP-anchored frame has no usable point,
    the VGGT+BA points (``fallback``) are used for that frame.
    """
    rows = []
    for fi in range(depth.shape[0]):
        source = labels[fi]
        vals = _ratios(depth[fi], points.get(fi, ()))
        if not vals and source == COLMAP_LABEL and fallback is not None:
            vals = _ratios(depth[fi], fallback.get(fi, ()))
            source = "VGGT+BA_fallback_COLMAP_insufficient_points"
        rows.append(_summary(fi, source, vals))
    return rows


def fit_vggt_scales(vggt_depth: np.ndarray, confidence: np.ndarray, coords: np.ndarray, points: dict,
                    labels, fallback: dict, height: int, width: int) -> list[dict]:
    """Median ratio per frame for the VGGT depth head.

    Keypoints are mapped from video pixels into VGGT's 518x518 crop with the
    per-frame ``original_coords`` (x1, y1, x2, y2, original width, original
    height on the 1024 canvas).  Points need positive VGGT confidence.  Frames
    without any selected points fall back to the VGGT+BA points.
    """
    rows = []
    for fi in range(vggt_depth.shape[0]):
        x1, y1, x2, y2, ow, oh = map(float, coords[fi])
        vals = []
        for x, y, z in (points.get(fi, ()) or fallback.get(fi, ())):
            xx = int(round(((x * (x2 - x1) / ow) + x1) * VGGT_SIZE / VGGT_CANVAS))
            yy = int(round(((y * (y2 - y1) / oh) + y1) * VGGT_SIZE / VGGT_CANVAS))
            if 0 <= xx < VGGT_SIZE and 0 <= yy < VGGT_SIZE:
                v = float(vggt_depth[fi, yy, xx])
                c = confidence[fi, yy, xx]
                if np.isfinite(v) and v > 0 and np.isfinite(c) and c > 0 and np.isfinite(z) and z > 0:
                    vals.append(float(z / v))
        row = _summary(fi, labels[fi], vals)
        if not vals:
            row["anchor_source"] = "VGGT+BA_fallback_no_usable_selected_points"
        rows.append(row)
    return rows


def vggt_to_image(vggt_depth: np.ndarray, coords_row, height: int, width: int) -> np.ndarray:
    """Crop the VGGT map to the video's region and resize it to (height, width)."""
    x1, y1, x2, y2, _, _ = map(float, coords_row)
    left, top = int(round(x1 * VGGT_SIZE / VGGT_CANVAS)), int(round(y1 * VGGT_SIZE / VGGT_CANVAS))
    right, bottom = int(round(x2 * VGGT_SIZE / VGGT_CANVAS)), int(round(y2 * VGGT_SIZE / VGGT_CANVAS))
    crop = vggt_depth[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("empty VGGT crop")
    return cv2.resize(crop.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def _scale_of(rows, fi):
    by = {int(r["frame_index"]): r["scale_median"] for r in rows}
    sc = by.get(fi)
    return sc if sc is not None and np.isfinite(sc) and sc > 0 else None


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse_frame(scaled_maps: list[np.ndarray], valid_masks: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of the scaled maps, NaN unless every map is valid."""
    stack = np.stack(scaled_maps, axis=0).astype(np.float32, copy=False)
    fused = np.median(stack, axis=0).astype(np.float32)
    inter = np.logical_and.reduce(valid_masks)
    fused[~inter] = np.nan
    return fused


def fused_depths(method_depths: dict, method_valid: dict, method_scales: dict, vggt_depth: np.ndarray,
                 vggt_coords: np.ndarray, vggt_scales: list[dict], frames, height: int, width: int) -> dict:
    """Fused depth for each requested frame.

    ``method_depths[name]`` is the raw (N, H, W) map of a video-resolution
    model, ``method_valid[name]`` its validity mask and ``method_scales[name]``
    its per-frame rows from ``fit_scales``.  The VGGT map is resampled with
    ``vggt_to_image``.  A missing scale makes that map invalid for the frame.
    """
    out = {}
    for fi in frames:
        maps, masks = [], []
        for name in method_depths:
            sc = _scale_of(method_scales[name], fi)
            raw = np.asarray(method_depths[name][fi], np.float32)
            d = raw * np.float32(sc) if sc is not None else np.full((height, width), np.nan, np.float32)
            maps.append(d)
            masks.append(np.asarray(method_valid[name][fi], bool) & np.isfinite(d) & (d > 0))
        raw = vggt_to_image(vggt_depth[fi], vggt_coords[fi], height, width)
        sc = _scale_of(vggt_scales, fi)
        if sc is None:
            vg, vvalid = np.full((height, width), np.nan, np.float32), np.zeros((height, width), bool)
        else:
            vg = raw * np.float32(sc)
            vvalid = np.isfinite(raw) & (raw > 0) & np.isfinite(vg) & (vg > 0)
        maps.append(vg)
        masks.append(vvalid)
        out[fi] = fuse_frame(maps, masks)
    return out


# ---------------------------------------------------------------------------
# File loading and CLI
# ---------------------------------------------------------------------------

def load_method_depth(path) -> tuple[np.ndarray, np.ndarray]:
    """Load one model's raw depth (``depth_raw_f16``) and ``valid_mask`` arrays."""
    with np.load(path, allow_pickle=False) as z:
        if "frame_indices" in z.files and not np.array_equal(np.asarray(z["frame_indices"]), np.arange(len(z["frame_indices"]))):
            raise ValueError(f"{path}: frames must be stored in order")
        return np.asarray(z["depth_raw_f16"], np.float32), np.asarray(z["valid_mask"], bool)


def load_vggt(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load VGGT ``depth`` (N, 518, 518[, 1]), ``confidence`` and ``original_coords``."""
    with np.load(path, allow_pickle=False) as z:
        depth = np.asarray(z["depth"], np.float32)
        conf = np.asarray(z["confidence"], np.float32)
        coords = np.asarray(z["original_coords"], np.float32)
    if depth.ndim == 4:
        depth = depth[..., 0]
    return depth, conf, coords


def build(method_paths: dict, vggt_path, vggt_model, frame_sources, frames, colmap_model=None,
          colmap_scale=None) -> tuple[dict, dict]:
    """Fit all scale factors and fuse the depth of ``frames``.  Returns (depths, scales)."""
    depths, valids = {}, {}
    for name, path in method_paths.items():
        depths[name], valids[name] = load_method_depth(path)
    n, height, width = next(iter(depths.values())).shape
    vg_depth, vg_conf, vg_coords = load_vggt(vggt_path)
    labels = anchor_labels(frame_sources)
    vg_obs = sparse_observations(vggt_model, height, width, n)
    col_obs = sparse_observations(colmap_model, height, width, n) if COLMAP_LABEL in labels else None
    points = anchor_points(labels, vg_obs, col_obs, colmap_scale)
    scales = {name: fit_scales(depths[name], points, labels, vg_obs) for name in depths}
    vg_scales = fit_vggt_scales(vg_depth, vg_conf, vg_coords, points, labels, vg_obs, height, width)
    fused = fused_depths(depths, valids, scales, vg_depth, vg_coords, vg_scales, frames, height, width)
    return fused, {"methods": scales, "VGGT": vg_scales}


def main(argv=None):
    from .cameras import fit_frames

    ap = argparse.ArgumentParser(description="Fuse seven BA-scaled depth maps of a real video (paper Sec. 3.3).")
    ap.add_argument("--method-depth", action="append", required=True, metavar="NAME=PATH",
                    help="raw depth npz of one model (keys depth_raw_f16, valid_mask); give six times")
    ap.add_argument("--vggt-predictions", required=True, help="npz with depth, confidence, original_coords")
    ap.add_argument("--vggt-sparse-model", required=True, help="COLMAP-format VGGT+BA model of the real video")
    ap.add_argument("--colmap-sparse-model", help="COLMAP model, needed when some frames use COLMAP cameras")
    ap.add_argument("--colmap-scale", type=float, help="scale of the COLMAP-to-VGGT similarity")
    ap.add_argument("--cameras", required=True, help="cameras.npz with real_frame_source (or frame_source)")
    ap.add_argument("--out", required=True, help="output npz")
    args = ap.parse_args(argv)
    paths = dict(item.split("=", 1) for item in args.method_depth)
    with np.load(args.cameras, allow_pickle=False) as z:
        sources = z["real_frame_source"] if "real_frame_source" in z.files else z["frame_source"]
        sources = np.asarray(sources).astype(str).tolist()
    frames = fit_frames(len(sources))
    fused, scales = build(paths, args.vggt_predictions, args.vggt_sparse_model, sources, frames,
                          args.colmap_sparse_model, args.colmap_scale)
    np.savez_compressed(args.out, frame_index=np.asarray(frames, np.int32),
                        fused_depth=np.stack([fused[f] for f in frames]).astype(np.float32),
                        scales_json=np.asarray(json.dumps(scales)))


if __name__ == "__main__":
    main()
