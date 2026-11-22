"""Track test on prompt-generated videos: NRG-PD track labels and track AUROC (paper Sec. 5.3, Table 3, Supp. I).

Per clip:
  1. Track score. Every point track (848x480 video coordinates) samples the NRG-PD angle map of each
     frame bilinearly, after mapping the point into the 518x294 model grid of that video (centre crop to
     848:480, as in nrgpd.infer). A sample is valid if the point is visible, inside the 848x480 frame and
     inside the grid; with several NRG-PD checkpoints (three in the paper) the sample is the mean over
     checkpoints and must be valid for all of them. Track score = median over its valid samples; a track
     is eligible with >= 10 valid samples.
  2. Labels. Among eligible tracks, label 1 ("flagged") if the score is >= the clip's 70th percentile,
     label 0 ("clean") if <= the 30th percentile (numpy.percentile, linear), the middle is not used.
  3. Clean-track cameras. Three light bundle adjustments per clip, each on 500 label-0 tracks drawn at
     random (select_ba_tracks: seed = first 16 hex digits of sha256("LIGHTBA_CLEAN_s<k>|<clip>|500")).
  4. TN90. Each track is refitted to every camera set and its reprojection-error curve normalised by the
     clip's median error at the same frame distance from the track's seed frame; TN90 = 90th percentile
     of the normalised curve (tn90_from_curves; ported from the code used for the paper).
  5. Track AUROC. Rows = (track, draw) pairs of eligible labelled tracks with a finite TN90 that were not
     used in that draw's bundle adjustment. Per-clip AUROC uses the clip's rows; the pooled AUROC uses all
     rows of all clips (Mann-Whitney, ties count 1/2). Clips without both classes are skipped.

Steps 3-4 need the light bundle adjustment and the per-track refit, which are not part of this file:
`auroc` takes their outputs per clip as arrays tn90__LIGHTBA_CLEAN_s<k> (TN90 per track, NaN if
undefined) and inba__LIGHTBA_CLEAN_s<k> (True for the 500 BA tracks), k = 0, 1, 2.

In the paper the comparison was restricted to tracks eligible for all six compared methods (NRG-PD and
five baselines, each with its own labels) and to the 144 clips where every method had scoreable draws;
pass that restriction as a boolean array 'support' per clip.

Examples:
    python -m experiments.prompt_video.track_auroc labels --predictions s1.npz s2.npz s3.npz \
        --tracks tracks.npz --source-hw 768 1360 --out clipA_labels.npz
    python -m experiments.prompt_video.track_auroc auroc --clips clips.txt --inputs per_clip/ --out auroc.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

from evaluation.metrics import auroc

SRC_W, SRC_H = 848, 480
GRID_W, GRID_H = 518, 294
MIN_VALID = 10
LOW_PCT, HIGH_PCT = 30, 70
BA_TRACKS = 500
DRAWS = (0, 1, 2)


# ---------------------------------------------------------------- 1. track scores
def crop_box(source_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) of the centre crop to the 848:480 aspect, in source pixels (as nrgpd.infer.read_video)."""
    h, w = source_hw
    if w / h > SRC_W / SRC_H + 1e-9:
        cw, ch = int(round(h * SRC_W / SRC_H)), h
    elif w / h < SRC_W / SRC_H - 1e-9:
        cw, ch = w, int(round(w * SRC_H / SRC_W))
    else:
        cw, ch = w, h
    return (w - cw) // 2, (h - ch) // 2, cw, ch


def _bilinear(img: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x1 = np.minimum(x0 + 1, img.shape[1] - 1)
    y1 = np.minimum(y0 + 1, img.shape[0] - 1)
    wx, wy = gx - x0, gy - y0
    return (1 - wy) * ((1 - wx) * img[y0, x0] + wx * img[y0, x1]) + wy * ((1 - wx) * img[y1, x0] + wx * img[y1, x1])


def track_scores(angle_maps: list[np.ndarray], tracks_xy: np.ndarray, visible: np.ndarray,
                 source_hw: tuple[int, int], crop: tuple[int, int, int, int] | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Median NRG angle per track and its number of valid samples.

    angle_maps: one (T, 294, 518) array per NRG-PD checkpoint (degrees).
    tracks_xy: (N, T, 2) in 848x480 video coordinates; visible: (N, T) bool.
    """
    xy = np.asarray(tracks_xy, np.float64)
    vis = np.asarray(visible, bool)
    n, t_count = vis.shape
    sh, sw = source_hw
    x0, y0, wc, hc = crop if crop is not None else crop_box(source_hw)
    fin = np.isfinite(xy).all(axis=-1)
    inb = vis & fin & (xy[..., 0] >= 0) & (xy[..., 0] < SRC_W) & (xy[..., 1] >= 0) & (xy[..., 1] < SRC_H)
    th_sum = np.zeros((n, t_count))
    th_cnt = np.zeros((n, t_count))
    with np.errstate(invalid="ignore"):
        gx = (xy[..., 0] * sw / SRC_W - x0) * GRID_W / wc
        gy = (xy[..., 1] * sh / SRC_H - y0) * GRID_H / hc
        ok = inb & (gx >= 0) & (gx < GRID_W) & (gy >= 0) & (gy < GRID_H)
    for theta in angle_maps:
        theta = np.asarray(theta, np.float64)
        for t in range(t_count):
            rr = np.flatnonzero(ok[:, t])
            if rr.size:
                th_sum[rr, t] += _bilinear(theta[t], gx[rr, t], gy[rr, t])
                th_cnt[rr, t] += 1
    valid = th_cnt == len(angle_maps)
    theta = np.where(valid, th_sum / np.maximum(th_cnt, 1), np.nan)
    n_valid = valid.sum(axis=1)
    med = np.full(n, np.nan)
    for i in range(n):
        if n_valid[i] >= MIN_VALID:
            med[i] = np.median(theta[i, valid[i]])
    return med, n_valid


# ---------------------------------------------------------------- 2. labels
def relative_labels(median: np.ndarray, n_valid: np.ndarray) -> np.ndarray:
    """1 = top 30% (>= 70th percentile), 0 = bottom 30% (<= 30th percentile), -1 middle, -2 not eligible."""
    eligible = np.flatnonzero(np.asarray(n_valid) >= MIN_VALID)
    med = np.asarray(median, np.float64)
    if not np.isfinite(med[eligible]).all():
        raise ValueError("non-finite median for an eligible track")
    label = np.full(med.shape, -2, np.int8)
    if eligible.size == 0:
        return label
    lo, hi = np.percentile(med[eligible], [LOW_PCT, HIGH_PCT])
    label[eligible] = -1
    label[eligible[med[eligible] >= hi]] = 1
    label[eligible[med[eligible] <= lo]] = 0
    return label


# ---------------------------------------------------------------- 3. BA track draws
def select_ba_tracks(label: np.ndarray, ba_rows: np.ndarray, clip_id: str, draw: int, n: int = BA_TRACKS) -> np.ndarray:
    """Track rows used by clean-track bundle adjustment `draw` (n label-0 tracks among the BA-usable rows)."""
    ba_rows = np.asarray(ba_rows)
    cand = np.flatnonzero(label[ba_rows] == 0)
    if len(cand) < n:
        raise ValueError(f"only {len(cand)} clean tracks for {clip_id}")
    name = f"LIGHTBA_CLEAN_s{draw}"
    seed = int(hashlib.sha256(f"{name}|{clip_id}|{n}".encode()).hexdigest()[:16], 16)
    pick = np.sort(cand[np.random.default_rng(seed).choice(len(cand), size=n, replace=False)])
    return ba_rows[pick]


# ---------------------------------------------------------------- 4. TN90
def tn90_from_curves(curve_px: np.ndarray, fit_ok: np.ndarray, seed_frame: np.ndarray) -> np.ndarray:
    """TN90 per track from its reprojection-error curve (N, T) in pixels.

    ref[d] = median error of all fitted tracks at frame distance d >= 1 from their seed frame (needs >= 50
    samples; distances without support take the nearest supported one; floor 1e-3 px; needs >= 5 supported
    distances, else all NaN). TN90 = 90th percentile over frames of curve / ref[d]; NaN if the fit failed.
    """
    c = np.asarray(curve_px, np.float64)
    ok = np.asarray(fit_ok, bool)
    t_count = c.shape[1]
    dt = np.arange(t_count)[None, :] - np.asarray(seed_frame)[:, None]
    ref = np.full(t_count, np.nan)
    for d in range(1, t_count):
        v = c[ok][(dt[ok] == d) & np.isfinite(c[ok])]
        if len(v) >= 50:
            ref[d] = np.median(v)
    good = np.flatnonzero(np.isfinite(ref))
    if len(good) < 5:
        return np.full(c.shape[0], np.nan)
    for d in range(1, t_count):
        if not np.isfinite(ref[d]):
            ref[d] = ref[good[np.argmin(np.abs(good - d))]]
    ref = np.maximum(ref, 1e-3)
    zz = np.where(np.isfinite(c) & (dt >= 1), c / ref[np.clip(dt, 0, t_count - 1)], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanpercentile(zz, 90, axis=1)
    out[~ok] = np.nan
    return out


# ---------------------------------------------------------------- 5. AUROC
def clip_rows(label: np.ndarray, tn90: list[np.ndarray], in_ba: list[np.ndarray],
              support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(scores, labels) over draws: supported labelled tracks with finite TN90, not used in that draw's BA."""
    scores, labels = [], []
    for s, ib in zip(tn90, in_ba):
        s = np.asarray(s, float)
        rows = np.flatnonzero(support & (label >= 0) & np.isfinite(s) & ~np.asarray(ib, bool))
        if len(rows):
            scores.append(s[rows])
            labels.append(label[rows])
    if not scores:
        return np.zeros(0), np.zeros(0, np.int8)
    return np.concatenate(scores), np.concatenate(labels)


def load_clip(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.array(z[k]) for k in z.files}
    if "label" not in d:
        d["label"] = relative_labels(d["median"], d["n_valid"])
    eligible = (d["n_valid"] >= MIN_VALID) & np.isfinite(d["median"])
    d["support"] = eligible & d["support"].astype(bool) if "support" in d else eligible
    return d


def track_auroc(clips: list[str], inputs: Path, draws=DRAWS) -> dict:
    per_clip, pos_all, neg_all = [], [], []
    for cid in clips:
        p = inputs / f"{cid}.npz"
        if not p.is_file():
            continue
        d = load_clip(p)
        keys = [k for k in draws if f"tn90__LIGHTBA_CLEAN_s{k}" in d and f"inba__LIGHTBA_CLEAN_s{k}" in d]
        s, lab = clip_rows(d["label"].astype(np.int8), [d[f"tn90__LIGHTBA_CLEAN_s{k}"] for k in keys],
                           [d[f"inba__LIGHTBA_CLEAN_s{k}"] for k in keys], d["support"])
        pos, neg = s[lab == 1], s[lab == 0]
        if not len(pos) or not len(neg):
            continue
        per_clip.append({"clip": cid, "auroc": auroc(pos, neg), "flagged_rows": int(len(pos)),
                         "clean_rows": int(len(neg)), "draws": len(keys)})
        pos_all.append(pos)
        neg_all.append(neg)
    pooled = auroc(np.concatenate(pos_all), np.concatenate(neg_all)) if per_clip else None
    return {"pooled_auroc": pooled, "n_clips": len(per_clip), "per_clip": per_clip}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("labels", help="track scores and labels of one clip from NRG-PD predictions")
    a.add_argument("--predictions", nargs="+", required=True, help="NRG-PD .npz (key angle_deg), one per checkpoint")
    a.add_argument("--tracks", required=True, help="npz with tracks_xy (N,T,2) in 848x480 coords and visible (N,T)")
    a.add_argument("--source-hw", type=int, nargs=2, required=True, help="height width of the original video")
    a.add_argument("--out", required=True)
    b = sub.add_parser("auroc", help="per-clip and pooled track AUROC")
    b.add_argument("--clips", required=True, type=Path, help="text file, one clip id per line")
    b.add_argument("--inputs", required=True, type=Path,
                   help="folder with <clip>.npz: median, n_valid, [label], [support], tn90__/inba__LIGHTBA_CLEAN_s<k>")
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "labels":
        maps = []
        for p in args.predictions:
            with np.load(p, allow_pickle=False) as z:
                maps.append(z["angle_deg"].astype(np.float64) if "angle_deg" in z.files
                            else z["degree_u8"].astype(np.float64) * float(z["degree_u8_scale"]))
        with np.load(args.tracks, allow_pickle=False) as z:
            xy, vis = z["tracks_xy"], z["visible"]
        med, nv = track_scores(maps, xy, vis, tuple(args.source_hw))
        lab = relative_labels(med, nv)
        np.savez_compressed(args.out, median=med, n_valid=nv, label=lab)
        print(json.dumps({"tracks": int(len(med)), "eligible": int((nv >= MIN_VALID).sum()),
                          "flagged": int((lab == 1).sum()), "clean": int((lab == 0).sum())}))
    else:
        clips = [x.strip() for x in args.clips.read_text().splitlines() if x.strip()]
        res = track_auroc(clips, args.inputs)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(json.dumps({"pooled_auroc": res["pooled_auroc"], "n_clips": res["n_clips"]}))


if __name__ == "__main__":
    main()
