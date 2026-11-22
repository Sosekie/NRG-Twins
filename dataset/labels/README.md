# Per-pixel NRG labels

Code for the label pipeline of the paper (Sec. 3.2–3.6). It turns a real video,
its generated twin, their cameras and the real video's depth maps into
`labels.npz`, the file shipped for every pair of NRG-Twins.

| File | Paper | Content |
|---|---|---|
| `cameras.py` | 3.2 | Similarity alignment of the twin trajectory to the real one on the fitting frames; reference choice; t⊥/Z map |
| `depth.py` | 3.3 | Per-frame scale of seven depth maps to the sparse reconstruction; all-seven median fusion. CLI writes `fused_depth.npz` |
| `reproject.py` | 3.4 | Z-buffered point-by-point warp of the reference into the twin camera |
| `flow.py` | 3.4 | DIS, SEA-RAFT and RAFT-large; forward-backward check; largest agreeing subset |
| `angle.py` | 3.5 | NRG ray angle; exclusion reason codes |
| `noise.py` | 3.6 | Keyframe noise σ̂², θ_c, ordinal bins, twin exclusion |
| `label_pair.py` | all | CLI: `run` (end to end) and `from-frames` (rebuild `labels.npz` from per-frame archives) |

## Requirements

numpy, opencv-python, pycolmap (for `depth.py`), torch, torchvision,
safetensors and huggingface_hub (for the flow models), and a checkout of
[SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT). The released labels were
produced with numpy 1.26, OpenCV 4.13, torch 2.7 and torchvision 0.22 on a CUDA
GPU with deterministic kernels; other versions or a CPU can change the learned
flows slightly.

## Usage

Run from the repository root.

```bash
# 1. Fused depth of the real video's fitting frames (Sec. 3.3)
python -m dataset.labels.depth \
  --method-depth da2_metric_hypersim_large=depth/da2.npz \
  --method-depth da3_base_multiview=depth/da3.npz \
  --method-depth mapanything_apache=depth/mapanything.npz \
  --method-depth metric3d_v2_vit_giant2=depth/metric3d.npz \
  --method-depth moge2_vitl=depth/moge2.npz \
  --method-depth unidepth_v2_vitl14=depth/unidepth.npz \
  --vggt-predictions vggt/predictions.npz --vggt-sparse-model vggt/sparse_ba \
  --colmap-sparse-model colmap/sparse_model --colmap-scale 0.0184 \
  --cameras pair/cameras.npz --out pair/fused_depth.npz

# 2. Labels (Sec. 3.2, 3.4-3.6)
python -m dataset.labels.label_pair run \
  --real-video pair/real.mp4 --twin-video pair/twin.mp4 --cameras pair/cameras.npz \
  --fused-depth pair/fused_depth.npz --keyframes 0,40,80 \
  --sea-raft-repo third_party/SEA-RAFT --out out/labels.npz --frames-dir out/frames

# Rebuild labels.npz from the full_truth tier (CPU only)
python -m dataset.labels.label_pair from-frames \
  --frames-dir full_truth/train/<source_id> --keyframes 0,40,80 --out out/labels.npz
```

Depth inputs: each model's npz holds `depth_raw_f16` (81, 480, 848) and
`valid_mask`; the VGGT file holds `depth` (81, 518, 518[, 1]), `confidence` and
`original_coords` (81, 6). The sparse models are COLMAP-format reconstructions
of the real video whose image names end in the frame index. `--colmap-*` is only
needed when `real_frame_source` marks some frames as `colmap_ba`; their points
are rescaled by the COLMAP-to-VGGT similarity scale.

`run` writes `labels.npz`, `depth.npz` (fused reference depth per twin frame,
as released) and `noise.json` (σ̂² per frame, keyframe RMS values and the twin
exclusion decision). `--frames-dir` additionally stores one archive per frame
with the same keys as the dataset's `full_truth` tier (without its provenance
strings and placeholder fields).

## Output format (`labels.npz`)

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `theta_deg` | (81, 480, 848) | float16 | Raw NRG angle in degrees; NaN where no flow consensus |
| `theta_c_deg` | (81, 480, 848) | float16 | sqrt(max(min(θ, 3°)² − σ̂²(t), 0)); NaN where not labelled |
| `ordinal_bin` | (81, 480, 848) | uint8 | Bin of θ_c (before float16 rounding), lower-inclusive edges 0.03, 0.075, 0.15, 0.3, 0.6, 1.2°, giving 0–6; 255 where not labelled |
| `valid_mask_packed` | (81, 480, 106) | uint8 | Label validity, `np.packbits(valid, axis=-1, bitorder="little")` |
| `exclusion_reason` | (81, 480, 848) | uint8 | 0 labelled, 1 outside warp support, 2 flow consensus rejected, 3 nonfinite angle, 4 nonfinite t⊥/Z, 5 t⊥/Z > 0.01 (first failing check in this order) |
| `is_keyframe` | (81,) | bool | Keyframes of the twin generation |

## Implementation details

These are the exact rules of the code that produced the released labels.

* **Fitting frames.** Frames 5, 15, …, 75 are held out; alignment and the
  reference candidates use the remaining 73 frames.
* **Alignment.** The rotation is the SO(3) projection of the mean of
  R_real R_twinᵀ; scale and translation are then a least-squares fit of the
  camera centres.
* **Reference choice.** The reference is the real fitting frame with the
  smallest median t⊥/Z over its valid fused-depth pixels; ties go to the frame
  nearest in time, then the lower index.
* **Depth fusion.** A fused pixel exists only where all seven scaled maps are
  valid. Scale anchors are the observed keypoints of the reconstruction points,
  rounded to the nearest pixel. Frames whose camera comes from COLMAP use
  COLMAP's points (rescaled by the COLMAP-to-VGGT similarity scale), falling
  back to the VGGT+BA points when none is usable.
* **Flow agreement.** Two methods agree if |f_i − f_j| ≤ min(1 + 0.05|f_i|,
  1 + 0.05|f_j|). The forward-backward test also needs the forward endpoint
  inside the image. Two-member medians equal the mean of the two flows.
* **t⊥/Z.** Computed on the reference pixel grid (both axes normalised by
  f_x) and applied at the same pixel index as the angle on the twin grid.
* **Noise.** σ̂²_k is the mean of min(|θ|, 3°)² over the labelled pixels of
  keyframe k. For the twin exclusion test (raw keyframe RMS > 0.6°, or
  noise-subtracted keyframe RMS > 0.3°), each keyframe is corrected with σ̂²
  interpolated from the other keyframes; a twin whose RMS cannot be computed is
  excluded.
