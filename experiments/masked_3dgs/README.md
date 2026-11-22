# Masked 3DGS fits (Sec. 5.2, Supp. G)

`masks.py` turns an NRG-PD prediction into per-frame masks (angle minus its per-pixel clip median
>= 0.06 deg, nearest-neighbour to 848x480) and, optionally, into the keep archive read by the fit.

## Feeding the masks to RobustSplat

Masked pixels are left out of the photometric loss; nothing else in RobustSplat changes.

1. Write the keep archive: one entry per training frame, named after the training image
   (`clutter_<t:03d>.png`, the names of the frames in the 3DGS scene folder), value `1 - mask`
   (uint8, 480x848):

   ```sh
   python -m experiments.masked_3dgs.masks pred.npz --out masks.npz \
       --keep-out external_keep.npz --train-frames 0 1 2 ...
   ```

2. In RobustSplat's `train.py`, load the archive once (we pass its path in the environment variable
   `RS_EXT_KEEP`) and, for each training view, replace the loss by

   ```python
   keep = ext_keep[viewpoint_cam.image_name]          # (1, H, W) float on the GPU
   Ll1 = (keep * torch.abs(image - gt_image)).mean()
   Lssim = (1.0 - ssim(keep * image, keep * gt_image, size_average=False)).mean()
   loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
   ```

   (depth and other regularisers unchanged).

3. Run with RobustSplat's own learned mask switched off and the paper's schedule:

   ```sh
   RS_EXT_KEEP=external_keep.npz python train.py --source_path <scene> --model_path <out> \
       --disable_viewer --iterations 30000 --save_iterations 30000 --disable_mask
   ```

An unmasked fit uses an all-ones archive (the same loss as plain L1 + D-SSIM); the random-area control
uses the archive built from the `--random-out` mask. Held-out frames are not trained on and are rendered from their cameras for
PSNR/LPIPS. The INRIA-derived trainer we used is not part of this release.
