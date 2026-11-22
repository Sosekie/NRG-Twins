"""Select an NRG-PD checkpoint on the validation split.

For every saved checkpoint, the validation loss (same loss as training) is computed on the 20
validation twins in nrgpd/val_selection.txt, all 81 frames, scored once with the four 24-frame
windows of nrgpd/infer.py. The selected checkpoint is the one with the lowest loss; checkpoints
within 1% of the lowest loss count as tied and the later step wins. The pooled AUROC on the same
pixels is reported but not used for selection. The test split is never used.

Example:
    python -m nrgpd.validate --data NRG-Twins --run runs/seed20261108
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import frame_targets, read_frames, read_sources
from .infer import windows
from .model import NRGPD

HERE = Path(__file__).resolve().parent


@torch.inference_mode()
def validate(model, root: Path, ids: list[str], sources: dict, device="cuda") -> dict:
    reg_sum = ord_sum = 0.0
    reg_n = ord_n = 0
    pos, neg = [], []
    for sid in ids:
        src = sources[sid]
        for start, first, last in windows(81):
            idx = np.arange(start, start + 24)
            target, ordinal, mask, theta_c = frame_targets(root, "val", sid, idx, src["sigma2"], src["keyframes"])
            scored = (idx >= first) & (idx <= last)
            mask = mask & scored[:, None, None]
            if not mask.any():
                continue
            frames = torch.from_numpy(read_frames(root / "val" / sid / "twin.mp4", idx)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(frames)[0].float()
            m = torch.from_numpy(mask).to(device)
            reg_sum += F.mse_loss(pred[:, 0][m], torch.from_numpy(target).to(device)[m], reduction="sum").item()
            om = m.unsqueeze(1).expand(-1, 6, -1, -1)
            ord_sum += F.binary_cross_entropy_with_logits(pred[:, 1:][om], torch.from_numpy(ordinal).to(device)[om], reduction="sum").item()
            reg_n += int(mask.sum())
            ord_n += int(mask.sum()) * 6
            s = pred[:, 0].cpu().numpy()
            pos.append(s[mask & (theta_c >= 0.6)])
            neg.append(s[mask & (theta_c < 0.075)])
    from evaluation.metrics import auroc
    return {"val_loss": reg_sum / reg_n + ord_sum / ord_n,
            "val_auroc": auroc(np.concatenate(pos), np.concatenate(neg)), "pixels": reg_n}


def select(rows: list[dict]) -> dict:
    best = min(r["val_loss"] for r in rows)
    tied = [r for r in rows if r["val_loss"] <= 1.01 * best]
    return max(tied, key=lambda r: r["step"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path, help="folder with step*.safetensors from nrgpd.train")
    ap.add_argument("--ids", type=Path, default=HERE / "val_selection.txt")
    ap.add_argument("--vggt-weights", default=None)
    args = ap.parse_args()
    ids = [x.strip() for x in args.ids.read_text().splitlines() if x.strip()]
    sources = read_sources(args.data)
    model = NRGPD(args.vggt_weights, pretrained_init=False).cuda().eval()
    rows = []
    for ckpt in sorted(args.run.glob("step*.safetensors")):
        model.load_trainable(ckpt)
        r = {"checkpoint": ckpt.name, "step": int(ckpt.stem[4:]), **validate(model, args.data, ids, sources)}
        rows.append(r)
        print(json.dumps(r), flush=True)
    chosen = select(rows)
    (args.run / "validation.json").write_text(json.dumps({"checkpoints": rows, "selected": chosen}, indent=1) + "\n")
    print("selected", chosen["checkpoint"])


if __name__ == "__main__":
    main()
