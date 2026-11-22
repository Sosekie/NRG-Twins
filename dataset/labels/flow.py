"""Optical-flow consensus from the projected reference to the twin frame (paper Sec. 3.4).

Three estimators are run in both directions: DIS (OpenCV, medium preset),
SEA-RAFT (official code, spring-M configuration, 4 iterations) and RAFT-large
(torchvision, C_T_SKHT_V2 weights, 12 updates).

* A method *passes* at a pixel if its forward-backward error
  ``|f(x) + b(x + f(x))|`` (backward flow sampled bilinearly) is at most
  ``1 + 0.05 |f(x)|`` pixels and ``x + f(x)`` stays inside the image, and the
  pixel is inside the warp support.
* Two methods *agree* if ``|f_i - f_j| <= min(1 + 0.05 |f_i|, 1 + 0.05 |f_j|)``.
* The consensus flow is the component-wise median of the largest subset of
  passing, pairwise-agreeing methods, if it has at least two members.  Among
  equally large subsets the first in the order (DIS, SEA-RAFT), (DIS, RAFT),
  (SEA-RAFT, RAFT) is used.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import cv2
import numpy as np

METHODS = ("DIS", "SEA_RAFT", "RAFT_large")

#: SEA-RAFT model arguments of the "spring-M" configuration used for the labels.
SEA_RAFT_SPRING_M = {
    "name": "spring-M", "dataset": "spring", "gpus": [0], "use_var": True, "var_min": 0, "var_max": 10,
    "pretrain": "resnet34", "initial_dim": 64, "block_dims": [64, 128, 256], "radius": 4, "dim": 128,
    "num_blocks": 2, "iters": 4, "image_size": [480, 848], "scale": 0, "batch_size": 2, "epsilon": 1e-8,
    "lr": 0.0004, "wdecay": 0.00001, "dropout": 0, "clip": 1.0, "gamma": 0.85, "num_steps": 120000,
    "restore_ckpt": None, "coarse_config": None,
}
SEA_RAFT_ITERS = 4
#: Hugging Face repository of the SEA-RAFT weights (file ``model.safetensors``).
SEA_RAFT_WEIGHTS_REPO = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"


# ---------------------------------------------------------------------------
# Consistency and consensus
# ---------------------------------------------------------------------------

def fb_check(f: np.ndarray, b: np.ndarray):
    """Forward-backward check of flow ``f`` against the reverse flow ``b``.

    Returns ``(error, threshold, inside, passed)`` on the grid of ``f``.
    """
    h, w = f.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    ex, ey = x + f[..., 0], y + f[..., 1]
    inside = (ex >= 0) & (ex <= w - 1) & (ey >= 0) & (ey <= h - 1)
    sampled = cv2.remap(b, ex.astype(np.float32), ey.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    error = np.linalg.norm(f + sampled, axis=-1).astype(np.float32)
    threshold = (1.0 + .05 * np.linalg.norm(f, axis=-1)).astype(np.float32)
    passed = inside & np.isfinite(error) & np.isfinite(threshold) & (error <= threshold)
    return error, threshold, inside, passed


def consensus(vectors, passes):
    """Largest pairwise-agreeing subset (size 3, else 2) of passing methods.

    ``vectors`` are three (H, W, 2) forward flows in the order of ``METHODS``
    and ``passes`` the matching boolean masks.  Returns ``(flow, accepted,
    subset_bits)``: the component-wise median flow (NaN where no subset
    exists), the acceptance mask and, per pixel, a bit mask of the winning
    methods (bit i set for ``METHODS[i]``; 0 = no consensus).
    """
    shape = passes[0].shape
    mags = [np.linalg.norm(x, axis=-1) for x in vectors]
    agree = {}
    for i, j in itertools.combinations(range(3), 2):
        agree[i, j] = np.linalg.norm(vectors[i] - vectors[j], axis=-1) <= np.minimum(1.0 + .05 * mags[i],
                                                                                     1.0 + .05 * mags[j])
    win = np.zeros(shape, np.uint8)
    for size in (3, 2):
        for sub in itertools.combinations(range(3), size):
            ok = np.ones(shape, bool)
            for i in sub:
                ok &= passes[i]
            for i, j in itertools.combinations(sub, 2):
                ok &= agree[i, j]
            ok &= win == 0
            win[ok] = sum(1 << i for i in sub)
    flow = np.full((*shape, 2), np.nan, np.float32)
    for bits in np.unique(win):
        if bits == 0:
            continue
        m = win == bits
        ids = [i for i in range(3) if bits & (1 << i)]
        flow[m] = np.median(np.stack([vectors[i][m] for i in ids], axis=1), axis=1)
    return flow, win != 0, win


def agreement_count(subset_bits: np.ndarray) -> np.ndarray:
    """Number of methods in the winning subset (0, 2 or 3) per pixel."""
    bits = np.asarray(subset_bits, np.uint8)
    return ((bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class DISFlow:
    """OpenCV DIS optical flow, medium preset, on grayscale images."""

    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def __call__(self, im1: np.ndarray, im2: np.ndarray):
        g1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        return self.dis.calc(g1, g2, None).astype(np.float32), self.dis.calc(g2, g1, None).astype(np.float32)


class SeaRaftFlow:
    """SEA-RAFT from the official repository (github.com/princeton-vl/SEA-RAFT).

    ``repo_dir`` is a checkout of the repository; ``weights`` is the
    ``model.safetensors`` file of ``SEA_RAFT_WEIGHTS_REPO`` (downloaded with
    ``huggingface_hub`` when omitted).  Images are passed as float RGB in
    [0, 255]; both directions are run as one batch of two.
    """

    def __init__(self, repo_dir, weights=None, device="cuda"):
        import torch
        from safetensors.torch import load_file

        repo_dir = Path(repo_dir)
        for p in (repo_dir / "core", repo_dir):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from raft import RAFT  # SEA-RAFT's core/raft.py

        if weights is None:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(SEA_RAFT_WEIGHTS_REPO, "model.safetensors")
        model = RAFT(argparse.Namespace(**SEA_RAFT_SPRING_M))
        result = model.load_state_dict(load_file(str(weights), device="cpu"), strict=False)
        # The checkpoint omits duplicate ``.downsample.1.`` aliases; nothing else may be missing.
        other = [k for k in result.missing_keys if ".downsample.1." not in k]
        if result.unexpected_keys or other:
            raise RuntimeError(f"SEA-RAFT state_dict mismatch: {result.unexpected_keys[:5]} {other[:5]}")
        self.torch, self.device = torch, device
        self.model = model.eval().to(device)

    def __call__(self, im1, im2):
        torch = self.torch
        a = torch.from_numpy(np.stack((im1, im2))).permute(0, 3, 1, 2).contiguous().to(self.device, torch.float32)
        b = torch.from_numpy(np.stack((im2, im1))).permute(0, 3, 1, 2).contiguous().to(self.device, torch.float32)
        with torch.inference_mode():
            pred = self.model(a, b, iters=SEA_RAFT_ITERS, test_mode=True)["final"].float().cpu().numpy()
        return pred[0].transpose(1, 2, 0).copy(), pred[1].transpose(1, 2, 0).copy()


class RaftLargeFlow:
    """torchvision RAFT-large with C_T_SKHT_V2 weights and its default 12 updates."""

    def __init__(self, weights=None, device="cuda"):
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        if weights is None:
            model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False)
        else:
            model = raft_large(weights=None, progress=False)
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.torch, self.device = torch, device
        self.model = model.to(device).eval()
        self.transform = Raft_Large_Weights.C_T_SKHT_V2.transforms()

    def __call__(self, im1, im2):
        torch = self.torch
        t1 = torch.from_numpy(np.stack((im1, im2))).permute(0, 3, 1, 2).contiguous().to(self.device)
        t2 = torch.from_numpy(np.stack((im2, im1))).permute(0, 3, 1, 2).contiguous().to(self.device)
        t1, t2 = self.transform(t1, t2)
        with torch.inference_mode():
            pred = self.model(t1, t2)[-1].float().cpu().numpy()
        return pred[0].transpose(1, 2, 0).copy(), pred[1].transpose(1, 2, 0).copy()


def set_deterministic(seed: int = 0):
    """Seed and switch PyTorch to deterministic kernels, as for the released labels."""
    import random
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def build_estimators(sea_raft_repo, sea_raft_weights=None, raft_weights=None, device="cuda") -> dict:
    """The three estimators in consensus order."""
    return {"DIS": DISFlow(), "SEA_RAFT": SeaRaftFlow(sea_raft_repo, sea_raft_weights, device),
            "RAFT_large": RaftLargeFlow(raft_weights, device)}


# ---------------------------------------------------------------------------
# One image pair
# ---------------------------------------------------------------------------

def measure(im1: np.ndarray, im2: np.ndarray, support: np.ndarray, estimators: dict) -> dict:
    """Run the three estimators from ``im1`` to ``im2`` and form the consensus.

    Returns a dict with ``flow`` (consensus, NaN outside ``accepted``),
    ``accepted``, ``subset_bits`` and per-method ``forward``, ``backward``,
    ``pass_forward`` and ``pass_backward`` (the latter two are the bare
    forward-backward tests, not restricted to ``support``).  If ``support`` is
    empty no estimator is run and ``methods`` is empty.
    """
    support = np.asarray(support, bool)
    h, w = support.shape
    result = {"flow": np.full((h, w, 2), np.nan, np.float32), "accepted": np.zeros((h, w), bool),
              "subset_bits": np.zeros((h, w), np.uint8), "methods": {}}
    if not support.any():
        return result
    vectors, passes = [], []
    for name in METHODS:
        f, b = estimators[name](im1, im2)
        pf = fb_check(f, b)[3]
        pb = fb_check(b, f)[3]
        result["methods"][name] = {"forward": f, "backward": b, "pass_forward": pf, "pass_backward": pb}
        vectors.append(f)
        passes.append(support & pf)
    flow, accepted, bits = consensus(vectors, passes)
    result["flow"][accepted] = flow[accepted]
    result.update(accepted=accepted, subset_bits=bits)
    return result
