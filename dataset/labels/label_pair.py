"""Per-pixel NRG labels for one real/twin pair (paper Sec. 3.2-3.6).

Two commands:

``run``
    End to end: align the cameras, choose a reference per twin frame, warp it
    with the fused depth, run the flow consensus, compute the NRG angle,
    validity, keyframe noise and θ_c, and write ``labels.npz`` (plus
    ``depth.npz`` and, optionally, one archive per frame).  Needs a GPU for
    SEA-RAFT and RAFT in practice.

``from-frames``
    Rebuild ``labels.npz`` from the per-frame archives
    (``frame_XXX.npz`` with ``theta_deg``, ``accepted_mask``,
    ``warp_valid_mask`` and ``q_t_perp_over_Z``), e.g. the dataset's
    ``full_truth`` tier.  CPU only.

``labels.npz`` holds, for 81 frames of 480x848 pixels:
``theta_deg`` (float16, raw angle, NaN without consensus), ``theta_c_deg``
(float16, noise-subtracted, NaN where unlabelled), ``ordinal_bin`` (uint8,
bins of θ_c with lower-inclusive edges 0.03, 0.075, 0.15, 0.3, 0.6, 1.2°;
255 = unlabelled), ``valid_mask_packed`` (uint8, validity packed along the
width, little-endian bit order), ``exclusion_reason`` (uint8, see
``angle.py``) and ``is_keyframe`` (bool per frame).
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import angle, cameras, flow, noise, reproject

METHOD_KEYS = {"DIS": "dis", "SEA_RAFT": "sea_raft", "RAFT_large": "raft_large"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_frames(path, indices) -> dict[int, np.ndarray]:
    """Decode the requested frames of a video as RGB uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    out = {}
    try:
        for i in sorted(set(int(x) for x in indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {i} of {path}")
            out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return out


def write_npz(path, members: dict) -> None:
    """Write an npz archive (deflate level 1, no pickles) atomically."""
    path = Path(path)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for key, array in members.items():
                with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_cameras(path):
    """Real and twin camera-to-world poses and the shared K from ``cameras.npz``."""
    with np.load(path, allow_pickle=False) as z:
        real_c2w = np.asarray(z["real_c2w_raw"], np.float64)
        twin_c2w = np.asarray(z["twin_c2w_raw"], np.float64)
        K = cameras.shared_intrinsics(z["real_K"], z["twin_K"])
    return real_c2w, twin_c2w, K


def load_fused_depth(path) -> dict[int, np.ndarray]:
    """Fused depth per real frame, as written by ``depth.py``."""
    with np.load(path, allow_pickle=False) as z:
        frames = np.asarray(z["frame_index"]).tolist()
        maps = np.asarray(z["fused_depth"], np.float32)
    return {int(f): maps[i] for i, f in enumerate(frames)}


# ---------------------------------------------------------------------------
# Per-frame label
# ---------------------------------------------------------------------------

def label_frame(t, twin_rgb, aligned_c2w, real_rgb, real_c2w, depths, K, estimators) -> dict:
    """All per-frame quantities for twin frame ``t``."""
    ref, _ = cameras.select_reference(t, aligned_c2w[t], real_c2w, depths, K)
    depth = depths[ref]
    warped, warp_depth, valid, src = reproject.warp_reference(real_rgb[ref], depth, real_c2w[ref],
                                                              aligned_c2w[t], K, K)
    valid = reproject.warp_support(valid, warp_depth)
    m = flow.measure(warped, twin_rgb, valid, estimators)
    theta, accepted = angle.nrg_angle(m["flow"], m["accepted"], valid, K)
    q = cameras.lateral_baseline_over_depth(real_c2w[ref], aligned_c2w[t], K, depth)
    out = {"theta_deg": theta, "accepted_mask": accepted, "warp_valid_mask": valid,
           "target_frame_index": np.int16(t), "reference_frame_index": np.int16(ref),
           "consensus_flow": m["flow"], "warped_reference_rgb": warped,
           "warp_depth": warp_depth.astype(np.float16), "zbuffer_source_pixel_index": src,
           "q_t_perp_over_Z": q, "fused_reference_depth": depth.astype(np.float16), "K": K,
           "camera_twin_aligned": aligned_c2w[t], "camera_real_reference": real_c2w[ref],
           "winner_agreement_mask": m["accepted"]}
    for name, rec in m["methods"].items():
        key = METHOD_KEYS[name]
        out[f"{key}_forward_flow_f16"] = rec["forward"].astype(np.float16)
        out[f"{key}_backward_flow_f16"] = rec["backward"].astype(np.float16)
        out[f"{key}_pass_forward"] = rec["pass_forward"]
        out[f"{key}_pass_backward"] = rec["pass_backward"]
    return out


# ---------------------------------------------------------------------------
# Pair-level assembly
# ---------------------------------------------------------------------------

def assemble_labels(frames: list[dict], keyframes, sigma2=None) -> tuple[dict, np.ndarray, dict]:
    """Build the ``labels.npz`` members from per-frame dicts.

    Each dict needs ``theta_deg`` (float32), ``accepted_mask``,
    ``warp_valid_mask`` and ``q_t_perp_over_Z``.  ``sigma2`` optionally pins
    the per-frame noise; otherwise it is estimated from the keyframes.
    Returns ``(labels, sigma2_per_frame, keyframe_rms)``.
    """
    n = len(frames)
    theta = np.stack([np.asarray(f["theta_deg"], np.float32) for f in frames])
    reason = np.stack([angle.exclusion_reason(theta[i], np.asarray(f["accepted_mask"], bool),
                                              np.asarray(f["warp_valid_mask"], bool), f["q_t_perp_over_Z"])
                       for i, f in enumerate(frames)])
    valid = reason == angle.REASON_VALID
    is_key = np.isin(np.arange(n), list(keyframes))
    if sigma2 is None:
        anchors = noise.keyframe_noise(theta, valid, keyframes)
        if not anchors:
            raise ValueError("no keyframe has labelled pixels; σ̂² is undefined")
        sigma2 = noise.interpolate_sigma2(anchors, n)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    if sigma2.shape != (n,) or not (np.isfinite(sigma2).all() and (sigma2 >= 0).all()):
        raise ValueError("invalid per-frame sigma2")
    corrected = np.empty(theta.shape, np.float16)
    ordinal = np.empty(theta.shape, np.uint8)
    for i in range(n):
        corrected[i], ordinal[i] = noise.corrected_angle(theta[i], valid[i], sigma2[i])
    labels = {"theta_deg": theta.astype(np.float16), "theta_c_deg": corrected, "ordinal_bin": ordinal,
              "valid_mask_packed": np.packbits(valid, axis=-1, bitorder="little"),
              "exclusion_reason": reason, "is_keyframe": is_key}
    return labels, sigma2, noise.keyframe_rms(theta, valid, keyframes)


def summary(sigma2, rms) -> dict:
    reason = noise.twin_exclusion(rms["raw_rms_deg"], rms["subtracted_rms_deg"])
    return {"sigma_hat_squared_deg2": sigma2.tolist(), "raw_rms_deg": rms["raw_rms_deg"],
            "subtracted_rms_deg": rms["subtracted_rms_deg"], "excluded": reason is not None,
            "exclusion_reason": reason}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args):
    real_c2w, twin_c2w, K = load_cameras(args.cameras)
    n = real_c2w.shape[0]
    fit = cameras.fit_frames(n)
    aligned, _ = cameras.align_twin_to_real(real_c2w, twin_c2w, fit)
    depths = load_fused_depth(args.fused_depth)
    missing = sorted(set(fit) - set(depths))
    if missing:
        raise ValueError(f"fused depth missing for fitting frames {missing[:10]}")
    depths = {f: depths[f] for f in fit}
    real_rgb = read_frames(args.real_video, fit)
    twin_rgb = read_frames(args.twin_video, range(n))
    if args.deterministic:
        flow.set_deterministic()
    estimators = flow.build_estimators(args.sea_raft_repo, args.sea_raft_weights, args.raft_weights, args.device)
    keyframes = [int(x) for x in args.keyframes.split(",")]
    if args.frames_dir:
        Path(args.frames_dir).mkdir(parents=True, exist_ok=True)
    frames = []
    depth_stack = np.full((n, *next(iter(depths.values())).shape), np.nan, np.float16)
    ref_index = np.full(n, -1, np.int16)
    for t in range(n):
        rec = label_frame(t, twin_rgb[t], aligned, real_rgb, real_c2w, depths, K, estimators)
        depth_stack[t] = rec["fused_reference_depth"]
        ref_index[t] = rec["reference_frame_index"]
        if args.frames_dir:
            write_npz(Path(args.frames_dir) / f"frame_{t:03d}.npz", rec)
        frames.append({k: rec[k] for k in ("theta_deg", "accepted_mask", "warp_valid_mask", "q_t_perp_over_Z")})
        print(json.dumps({"frame": t, "reference": int(ref_index[t]),
                          "accepted": int(frames[-1]["accepted_mask"].sum())}), flush=True)
    finish(args, frames, keyframes, depth={"fused_reference_depth": depth_stack,
                                           "available": np.ones(n, bool), "reference_frame_index": ref_index})


def cmd_from_frames(args):
    keyframes = [int(x) for x in args.keyframes.split(",")]
    frames = []
    for t in range(args.n_frames):
        with np.load(Path(args.frames_dir) / f"frame_{t:03d}.npz", allow_pickle=False) as z:
            if int(z["target_frame_index"]) != t:
                raise ValueError(f"frame_{t:03d}.npz holds frame {int(z['target_frame_index'])}")
            frames.append({k: np.asarray(z[k]) for k in ("theta_deg", "accepted_mask", "warp_valid_mask",
                                                          "q_t_perp_over_Z")})
    finish(args, frames, keyframes)


def finish(args, frames, keyframes, depth=None):
    sigma2 = None
    if args.sigma2:
        sigma2 = json.loads(Path(args.sigma2).read_text())
    labels, sigma2, rms = assemble_labels(frames, keyframes, sigma2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_npz(out, labels)
    if depth is not None:
        write_npz(out.with_name("depth.npz"), depth)
    info = summary(sigma2, rms)
    out.with_name("noise.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("raw_rms_deg", "subtracted_rms_deg", "excluded", "exclusion_reason")}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="compute labels for one pair end to end")
    run.add_argument("--real-video", required=True)
    run.add_argument("--twin-video", required=True)
    run.add_argument("--cameras", required=True, help="cameras.npz (real_/twin_ c2w_raw and K)")
    run.add_argument("--fused-depth", required=True, help="output of depth.py for the real video")
    run.add_argument("--sea-raft-repo", required=True, help="checkout of github.com/princeton-vl/SEA-RAFT")
    run.add_argument("--sea-raft-weights", help="model.safetensors of the spring-M weights (default: download)")
    run.add_argument("--raft-weights", help="torchvision RAFT-large C_T_SKHT_V2 state dict (default: download)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    run.add_argument("--frames-dir", help="also write one archive per frame here")
    run.set_defaults(func=cmd_run)

    frm = sub.add_parser("from-frames", help="rebuild labels.npz from per-frame archives")
    frm.add_argument("--frames-dir", required=True, help="directory with frame_000.npz ... ")
    frm.add_argument("--n-frames", type=int, default=81)
    frm.set_defaults(func=cmd_from_frames)

    for p in (run, frm):
        p.add_argument("--keyframes", required=True, help="comma-separated keyframe indices, e.g. 0,40,80")
        p.add_argument("--sigma2", help="optional JSON list of per-frame σ̂² to use instead of estimating it")
        p.add_argument("--out", required=True, help="path of labels.npz (depth.npz, noise.json go next to it)")
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
