"""Train NRG-PD on the NRG-Twins training split.

Settings of the paper: one 24-frame window per step (random start, stride 1-3, half reversed),
3,000 steps, AdamW (lr 1e-4, betas 0.9/0.999, weight decay 0.01), linear warm-up over the first 300
steps and a constant learning rate afterwards, gradient clipping at 1.0, bfloat16 autocast,
seeds 20261108, 20261109 and 20261110. Loss: squared error on the regression channel plus binary
cross-entropy on the six ordinal channels, both over labelled non-keyframe pixels.
Checkpoints (trainable parameters only) are written every --save-every steps for selection on the
validation split (nrgpd/validate.py).

Example:
    python -m nrgpd.train --data NRG-Twins --seed 20261108 --out runs/seed20261108
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from .data import read_sources, training_sample
from .model import NRGPD

LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP = 300
GRAD_CLIP = 1.0
MAX_RETRIES = 32
HERE = Path(__file__).resolve().parent


def loss_fn(pred: torch.Tensor, target, ordinal, mask):
    reg = F.mse_loss(pred[:, :, 0][mask].float(), target[mask].float())
    omask = mask.unsqueeze(2).expand_as(pred[:, :, 1:])
    ordl = F.binary_cross_entropy_with_logits(pred[:, :, 1:][omask].float(), ordinal[omask].float())
    return reg + ordl, reg, ordl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--vggt-weights", default=None)
    ap.add_argument("--order", type=Path, default=HERE / "train_order.txt",
                    help="order of training twins used to draw a twin per step (as in the paper runs)")
    args = ap.parse_args()
    ids = [x.strip() for x in args.order.read_text().splitlines() if x.strip()]
    sources = read_sources(args.data)
    random.seed(args.seed)
    np.random.seed(args.seed % 2 ** 32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = NRGPD(args.vggt_weights).to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "train_log.jsonl").open("a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sample = None
        for attempt in range(MAX_RETRIES):
            sample = training_sample(args.data, ids, sources, args.seed, step, attempt)
            if sample is not None:
                break
        if sample is None:
            raise RuntimeError(f"no labelled pixels in {MAX_RETRIES} draws at step {step}")
        for g in opt.param_groups:
            g["lr"] = LR * min(step / WARMUP, 1.0)
        frames = torch.from_numpy(sample["frames"]).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        target = torch.from_numpy(sample["target"]).unsqueeze(0).to(device)
        ordinal = torch.from_numpy(sample["ordinal"]).unsqueeze(0).to(device)
        mask = torch.from_numpy(sample["mask"]).unsqueeze(0).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, reg, ordl = loss_fn(model(frames), target, ordinal, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        log.write(json.dumps({"step": step, "loss": loss.item(), "regression": reg.item(), "ordinal": ordl.item(),
                              "twin": sample["id"], "seconds": round(time.time() - t0, 1)}) + "\n")
        if step % 10 == 0:
            log.flush()
            print(f"step {step} loss {loss.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_file({k: v.float().contiguous() for k, v in model.trainable_state().items()},
                      str(args.out / f"step{step:04d}.safetensors"),
                      metadata={"seed": str(args.seed), "step": str(step)})
    log.close()


if __name__ == "__main__":
    main()
