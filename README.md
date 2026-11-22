# NRG-Twins

Code for **NRG-Twins: Pixel-Level Ground Truth for Rigidity Violations in Generated Videos**
(Chenrui Fan, Qiyang Hu, Luca Scharr, Paolo Favaro; University of Bern).

Generated video is not 3D consistent: static content slowly deforms or slides. NRG-Twins measures
this per pixel. A real RealEstate10K clip is re-created by a keyframe-conditioned video generator
(the *twin*); after real keyframes are inserted so that the twin camera follows the real one, the
**NRG angle** is the angle, in degrees, between where content would appear in a rigid scene seen
along the twin's camera path and where it actually appears. **NRG-PD** predicts this angle for every
pixel from the generated video alone.

- Dataset (871 twins, labels, cameras, depth): https://huggingface.co/datasets/cvg-unibe/NRG-Twins
- NRG-PD weights (three training seeds): https://huggingface.co/cvg-unibe/NRG-PD
- Project page and paper: links will be added here.

## Install

```sh
pip install -r requirements.txt
```

The VGGT-1B weights are downloaded from Hugging Face on first use (`facebook/VGGT-1B`), or pass a
local `model.safetensors` with `--vggt-weights`.

## Run NRG-PD on a video

```sh
huggingface-cli download cvg-unibe/NRG-PD --local-dir weights
python -m nrgpd.infer my_video.mp4 --weights weights/nrgpd_seed20261108.safetensors --out pred.npz
```

`pred.npz` holds `angle_deg`, the predicted NRG angle per pixel and frame at 294 x 518, and `raw`,
the regression output used for ranking. Frames are centre-cropped to 848:480, and a clip is covered
by 24-frame windows (81 frames: windows starting at frames 0, 19, 38 and 57).

## Evaluate on the test split

```sh
huggingface-cli download cvg-unibe/NRG-Twins --repo-type dataset --exclude "full_truth/*" --local-dir NRG-Twins
for id in $(cat evaluation/test_benchmark.txt); do
  python -m nrgpd.infer NRG-Twins/test/$id/twin.mp4 --weights weights/nrgpd_seed20261108.safetensors --out preds/$id.npz
done
python -m evaluation.evaluate --data NRG-Twins --pred-dir preds --out metrics.json
```

The benchmark uses the 83 test twins in `evaluation/test_benchmark.txt` (all test twins whose
keyframes can be scored). NRG pixels have θc ≥ 0.6° and rigid pixels θc < 0.075°. The evaluator
reports pixel AUROC, Kendall τ, clip correlation, average precision, precision and recall of the top
5% pixels per clip, and the angle error (MAE, degrees). Test labels are public; please do not use
them for model selection, threshold choice or tuning.

For seed 20261108, evaluating each method on its own valid pixels, this code gives AUROC 0.8634,
Kendall τ 0.6903 and MAE 0.0899° on 483,823,593 pixels.

## Train NRG-PD

```sh
python -m nrgpd.train --data NRG-Twins --seed 20261108 --out runs/seed20261108
python -m nrgpd.validate --data NRG-Twins --run runs/seed20261108
```

Training uses one 24-frame window per step for 3,000 steps (8.3 GB peak GPU memory in our
check on an RTX 3090) and saves a
checkpoint every 250 steps. `nrgpd.validate` computes the validation loss of every checkpoint on the
20 twins in `nrgpd/val_selection.txt` and keeps the lowest; checkpoints within 1% count as tied and
the later step wins. The released weights are the checkpoints selected this way for seeds 20261108
(step 2715), 20261109 (step 3000) and 20261110 (step 3000).

## Build twins and labels for a new generator

1. Clips: `python -m dataset.sources.download_re10k download|extract` produces 81-frame 848x480
   RealEstate10K clips; `python -m dataset.sources.split` splits them by YouTube video.
2. Twins: `python -m dataset.twin_generation.insert_keyframes` re-creates a clip with Wan2.1-Fun-1.3B-InP
   segment by segment and inserts real keyframes until the largest rotation gap to the real camera is at
   most 0.8° (at most 6 insertions). To use another keyframe-conditioned generator, replace
   `dataset/twin_generation/generate.py`.
3. Labels: estimate cameras and per-method depth, fuse the depth (`python -m dataset.labels.depth`),
   then `python -m dataset.labels.label_pair run` computes the per-pixel NRG angle, the keyframe noise
   and `labels.npz` in the format of the dataset. See [dataset/labels/README.md](dataset/labels/README.md).

Run from the paper's inputs, this code reproduces the released `labels.npz` files element for element
(checked on two training twins, all 81 frames). SEA-RAFT and RAFT run on the GPU, which is not bitwise
deterministic across devices; re-running them changes the selected flow methods at 0.1–0.5% of the
pixels of a frame.

## Downstream experiments

- `experiments/masked_3dgs`: per-pixel masks from NRG-PD for 3DGS fits (pixel masked if its predicted
  angle exceeds its own median over the clip by 0.06°), and how they are passed to RobustSplat.
- `experiments/generated_frames`: the rule that drops whole frames whose mean predicted angle stands out
  from the 6 frames with the nearest cameras.
- `experiments/prompt_video`: the track test on prompt-generated video (track labels from NRG-PD, TN90
  reprojection error, track AUROC).

## Repository layout

| Folder | Paper | Contents |
|---|---|---|
| `nrgpd/` | Sec. 4 | model, inference, training, checkpoint selection |
| `evaluation/` | Sec. 5.1, Table 2 | pixel metrics and the test evaluator |
| `dataset/twin_generation/` | Sec. 3.1 | keyframe-conditioned re-creation and keyframe insertion |
| `dataset/labels/` | Sec. 3.2–3.6 | cameras, fused depth, reprojection, optical-flow consensus, NRG angle, keyframe noise |
| `dataset/sources/` | Supp. A.7 | RealEstate10K clips and the split by YouTube video |
| `experiments/` | Sec. 5.2–5.3 | masking and frame-dropping rules for 3DGS, track test on prompt-generated video |

## License

Code: MIT ([LICENSE](LICENSE)). Dataset: CC BY-NC 4.0. Third-party models and code keep their own
licenses ([THIRD_PARTY.md](THIRD_PARTY.md)).

## Citation

A BibTeX entry will be added with the paper link.
