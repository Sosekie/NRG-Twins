"""Pixel-level metrics of the NRG-Twins benchmark (paper Sec. 5.1 and Supp. K).

Labels: noise-subtracted NRG angle theta_c in degrees. NRG pixels: theta_c >= 0.6; rigid pixels:
theta_c < 0.075; pixels in between are left out of AUROC, AP, precision and recall.
All functions take flat arrays; "clips" are lists with one entry per twin.
"""
from __future__ import annotations

import hashlib

import numpy as np

NRG_DEG = 0.6
RIGID_DEG = 0.075
PAIR_SEED = 20261103
PAIRS_PER_CLIP = 20000


def auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """P(s(p) > s(q)) + 0.5 P(s(p) = s(q)) for p in pos, q in neg (Mann-Whitney)."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    sn = np.sort(np.asarray(neg, np.float64), kind="mergesort")
    p = np.asarray(pos, np.float64)
    lo = np.searchsorted(sn, p, "left")
    hi = np.searchsorted(sn, p, "right")
    return float((lo.sum() + 0.5 * (hi - lo).sum()) / (len(p) * len(sn)))


def kendall_tau_clip(score: np.ndarray, theta_c: np.ndarray, sigma_hat: np.ndarray, clip_id: str) -> float | None:
    """Kendall tau over 20,000 random pixel pairs of one clip, keeping pairs whose labels differ by
    more than twice the larger per-frame noise sigma_hat (degrees). Ties count 0."""
    n = len(theta_c)
    take = min(PAIRS_PER_CLIP, max(0, n * (n - 1) // 2))
    if not take:
        return None
    rng = np.random.default_rng(PAIR_SEED + int(hashlib.sha256(clip_id.encode()).hexdigest()[:8], 16))
    a = rng.integers(0, n, take)
    b = rng.integers(0, n, take)
    ok = a != b
    a, b = a[ok], b[ok]
    keep = np.abs(theta_c[a] / 0.6 - theta_c[b] / 0.6) > 2 * np.maximum(sigma_hat[a], sigma_hat[b]) / 0.6
    v = ((score[a] - score[b]) * (theta_c[a] - theta_c[b]))[keep]
    if not len(v):
        return None
    return float(2 * ((np.count_nonzero(v > 0) + 0.5 * np.count_nonzero(v == 0)) / len(v)) - 1)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr
    if len(x) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def average_precision(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """Mean over NRG pixels of the share of NRG pixels among pixels scored at least as high."""
    if len(pos) == 0:
        return None
    s = np.concatenate([pos, neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    # ties: a pixel counts every pixel with an equal score as ranked at least as high
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp_at = np.cumsum(y)[last]
    n_at = last + 1
    group = np.repeat(np.arange(len(last)), np.diff(np.r_[-1, last]))
    prec = tp_at[group] / n_at[group]
    return float(prec[y].mean())


def topk_precision_recall(clips_score, clips_theta, frac: float = 0.05):
    """Precision and recall of the top `frac` scored labelled pixels of each clip, pooled over clips."""
    hit = sel = nrg = 0
    for s, t in zip(clips_score, clips_theta):
        k = int(np.ceil(frac * len(s)))
        if k == 0:
            continue
        top = np.argpartition(-s, k - 1)[:k]
        is_nrg = t >= NRG_DEG
        hit += int(is_nrg[top].sum())
        sel += k
        nrg += int(is_nrg.sum())
    return (hit / sel if sel else None), (hit / nrg if nrg else None)


def evaluate(clips: list[dict], angle_key: str = "angle_deg") -> dict:
    """clips: dicts with 'id', 'score' (ranking score), 'theta_c', 'sigma_hat' (per pixel, degrees)
    and optionally 'angle_deg' (predicted angle in degrees, for MAE)."""
    pos = np.concatenate([c["score"][c["theta_c"] >= NRG_DEG] for c in clips])
    neg = np.concatenate([c["score"][c["theta_c"] < RIGID_DEG] for c in clips])
    taus = [kendall_tau_clip(c["score"], c["theta_c"], c["sigma_hat"], c["id"]) for c in clips]
    taus = [t for t in taus if t is not None]
    p90s = np.array([np.quantile(c["score"], 0.9) for c in clips])
    p90t = np.array([np.quantile(c["theta_c"], 0.9) for c in clips])
    prec, rec = topk_precision_recall([c["score"] for c in clips], [c["theta_c"] for c in clips])
    out = {
        "n_clips": len(clips),
        "n_pixels": int(sum(len(c["theta_c"]) for c in clips)),
        "n_nrg": int(len(pos)), "n_rigid": int(len(neg)),
        "auroc": auroc(pos, neg),
        "kendall_tau": float(np.mean(taus)) if taus else None,
        "clip_corr": spearman(p90s, p90t),
        "average_precision": average_precision(pos, neg),
        "precision_top5": prec, "recall_top5": rec,
    }
    if all(angle_key in c for c in clips):
        err = np.concatenate([np.abs(c[angle_key] - c["theta_c"]) for c in clips])
        out["mae_deg"] = float(err.mean())
    return out
