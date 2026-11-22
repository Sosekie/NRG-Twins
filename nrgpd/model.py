"""NRG-PD: VGGT-1B with LoRA adapters and a DPT head that predicts the NRG angle per pixel.

Architecture (paper Sec. 4):
- VGGT-1B aggregator (24 frame-attention and 24 global-attention blocks), frozen.
- LoRA of rank 16 on the qkv, output-projection and both MLP layers of the last 8 frame and
  last 8 global blocks.
- A DPT head initialised from VGGT's depth head, with 7 output channels: channel 0 regresses the
  noise-subtracted squared angle in units of 0.6 deg; channels 1-6 are ordinal logits for the
  edges 0.03, 0.075, 0.15, 0.3, 0.6 and 1.2 deg.

Module names match the released weight files, which hold only the trainable parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

PATCH = 14
VGGT_REPO_ID = "facebook/VGGT-1B"
VGGT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
ORDINAL_EDGES_DEG = (0.03, 0.075, 0.15, 0.3, 0.6, 1.2)
UNIT_DEG = 0.6


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus trainable low-rank matrices A (rank x in) and B (out x rank)."""

    def __init__(self, base: nn.Linear, rank: int = 16) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = 1.0  # alpha = rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def add_lora(aggregator: nn.Module, rank: int = 16, last_n: int = 8) -> None:
    for p in aggregator.parameters():
        p.requires_grad_(False)
    for blocks in (aggregator.frame_blocks[-last_n:], aggregator.global_blocks[-last_n:]):
        for block in blocks:
            for name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                parent_name, _, leaf = name.rpartition(".")
                parent = block.get_submodule(parent_name)
                setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank=rank))


def load_vggt(weights: str | Path | None = None):
    """Return the official VGGT-1B model with pretrained weights."""
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file

    if weights is None:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(VGGT_REPO_ID, "model.safetensors", revision=VGGT_REVISION)
    model = VGGT(img_size=518, patch_size=14, embed_dim=1024)
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return model


class Backbone(nn.Module):
    """VGGT aggregator returning the four DPT feature levels (layers 4, 11, 17, 23)."""

    def __init__(self, vggt_model: nn.Module, lora_rank: int = 16, lora_blocks: int = 8) -> None:
        super().__init__()
        self.aggregator = vggt_model.aggregator
        # The paper runs built an unused 6->1024 projection here; creating it keeps the same
        # random-number sequence, so the LoRA initialisation matches theirs for a given seed.
        nn.Linear(6, 1024, bias=False)
        add_lora(self.aggregator, rank=lora_rank, last_n=lora_blocks)
        self.aggregator.patch_embed.eval()

    def train(self, mode: bool = True) -> "Backbone":
        super().train(mode)
        self.aggregator.patch_embed.eval()  # frozen DINOv2 patch embedding stays in eval mode
        return self

    def features(self, frames: Tensor) -> list[Tensor]:
        from vggt.models.aggregator import slice_expand_and_flatten

        agg = self.aggregator
        B, S, _, H, W = frames.shape
        ph, pw = H // PATCH, W // PATCH
        image = frames.float().div(255.0) if frames.dtype == torch.uint8 else frames.float()
        image = (image - agg._resnet_mean.to(image)) / agg._resnet_std.to(image)
        with torch.no_grad():
            patch_tokens = agg.patch_embed(image.reshape(B * S, 3, H, W))
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        C = patch_tokens.shape[-1]
        tokens = torch.cat((
            slice_expand_and_flatten(agg.camera_token, B, S),
            slice_expand_and_flatten(agg.register_token, B, S),
            patch_tokens.reshape(B * S, ph * pw, C),
        ), dim=1)
        P = tokens.shape[1]
        pos = None
        if agg.rope is not None:
            pos = agg.position_getter(B * S, ph, pw, device=frames.device)
        if agg.patch_start_idx > 0:
            pos = pos + 1
            special = torch.zeros(B * S, agg.patch_start_idx, 2, device=frames.device, dtype=pos.dtype)
            pos = torch.cat((special, pos), dim=1)
        frame_idx = global_idx = 0
        levels: list[Tensor | None] = []
        for _ in range(agg.aa_block_num):
            for kind in agg.aa_order:
                if kind == "frame":
                    tokens, frame_idx, frame_out = agg._process_frame_attention(tokens, B, S, P, C, frame_idx, pos=pos)
                else:
                    tokens, global_idx, global_out = agg._process_global_attention(tokens, B, S, P, C, global_idx, pos=pos)
            for i in range(len(frame_out)):
                layer = len(levels)
                if layer in agg.cached_layer_indices:
                    v = torch.cat((frame_out[i], global_out[i]), dim=-1)[:, :, agg.patch_start_idx:]
                    levels.append(v.reshape(B, S, ph, pw, v.shape[-1]).permute(0, 1, 4, 2, 3).contiguous())
                else:
                    levels.append(None)
        return [levels[i] for i in (4, 11, 17, 23)]


class AngleHead(nn.Module):
    """DPT decoder initialised from VGGT's depth head, 7 output channels."""

    def __init__(self, depth_head_state: dict[str, Tensor] | None) -> None:
        super().__init__()
        from vggt.heads.dpt_head import DPTHead, custom_interpolate

        self._interp = custom_interpolate
        self.dpt = DPTHead(dim_in=2048, patch_size=PATCH, output_dim=2,
                           intermediate_layer_idx=[0, 1, 2, 3], feature_only=False)
        if depth_head_state is not None:
            current = self.dpt.state_dict()
            compatible = {k: v for k, v in depth_head_state.items()
                          if k in current and v.shape == current[k].shape and not k.startswith("scratch.output_conv2.2.")}
            self.dpt.load_state_dict(compatible, strict=False)
        self.dpt.scratch.output_conv2[2] = nn.Conv2d(32, 7, kernel_size=1, bias=True)
        nn.init.zeros_(self.dpt.scratch.output_conv2[2].weight)
        nn.init.zeros_(self.dpt.scratch.output_conv2[2].bias)

    def _chunk(self, features: Sequence[Tensor], frames: Tensor) -> Tensor:
        B, S, _, H, W = frames.shape
        ph, pw = H // PATCH, W // PATCH
        projected = []
        for i, f in enumerate(features):
            x = f.flatten(3).transpose(2, 3).reshape(B * S, ph * pw, f.shape[2])
            x = self.dpt.norm(x).permute(0, 2, 1).reshape(B * S, f.shape[2], ph, pw)
            x = self.dpt.projects[i](x)
            if self.dpt.pos_embed:
                x = self.dpt._apply_pos_embed(x, W, H)
            projected.append(self.dpt.resize_layers[i](x))
        fused = self.dpt.scratch_forward(projected)
        fused = self._interp(fused, (H, W), mode="bilinear", align_corners=True)
        if self.dpt.pos_embed:
            fused = self.dpt._apply_pos_embed(fused, W, H)
        out = self.dpt.scratch.output_conv2(fused)
        return out.reshape(B, S, 7, H, W)

    def forward(self, features: Sequence[Tensor], frames: Tensor, chunk: int = 8) -> Tensor:
        parts = []
        for s in range(0, frames.shape[1], chunk):
            e = min(s + chunk, frames.shape[1])
            feats, fr = [f[:, s:e] for f in features], frames[:, s:e]
            if self.training and torch.is_grad_enabled():
                # activation checkpointing per chunk keeps training within 24 GB
                parts.append(checkpoint(lambda *x: self._chunk(x[:4], x[4]), *feats, fr, use_reentrant=False))
            else:
                parts.append(self._chunk(feats, fr))
        return torch.cat(parts, dim=1)


class NRGPD(nn.Module):
    """Input: uint8 frames (B, T, 3, H, W), H and W divisible by 14. Output: (B, T, 7, H, W)."""

    def __init__(self, vggt_weights: str | Path | None = None, pretrained_init: bool = True) -> None:
        super().__init__()
        vggt = load_vggt(vggt_weights)
        depth_state = {k: v.detach().clone() for k, v in vggt.depth_head.state_dict().items()} if pretrained_init else None
        self.backbone = Backbone(vggt)
        self.degree_head = AngleHead(depth_state)
        del vggt

    def train(self, mode: bool = True) -> "NRGPD":
        super().train(mode)
        self.backbone.train(mode)
        return self

    def forward(self, frames: Tensor) -> Tensor:
        return self.degree_head(self.backbone.features(frames), frames)

    def trainable_state(self) -> dict[str, Tensor]:
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, path: str | Path) -> None:
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
        params = dict(self.named_parameters())
        expected = {n for n, p in params.items() if p.requires_grad}
        if set(state) != expected:
            missing, extra = sorted(expected - set(state))[:5], sorted(set(state) - expected)[:5]
            raise RuntimeError(f"weight names differ from the model: missing {missing}, unexpected {extra}")
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v.to(params[n]))


def raw_to_degrees(raw: Tensor | "np.ndarray"):
    """Channel 0 regresses (angle / 0.6 deg)^2 after noise subtraction; convert to degrees."""
    if isinstance(raw, Tensor):
        return UNIT_DEG * torch.sqrt(torch.clamp(raw, min=0))
    import numpy as np
    return UNIT_DEG * np.sqrt(np.maximum(raw, 0))
