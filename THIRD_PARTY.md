# Third-party code and models

The scripts here call the following upstream code and weights. They are not included in this
repository; install them from their sources under their own licenses.

| Component | Used for | Source | Version we used |
|---|---|---|---|
| VGGT-1B | NRG-PD backbone, camera rotation gap, camera estimates | facebookresearch/vggt, facebook/VGGT-1B | code a288dd0f, weights revision 860abec7 |
| Wan2.1-Fun-1.3B-InP | keyframe-conditioned twin generation, generated frames in Sec. 5.2 | alibaba-pai/Wan2.1-Fun-1.3B-InP | revision 9bcbbcf6 |
| COLMAP | camera estimation | colmap/colmap | – |
| Depth Anything V2, Depth Anything 3, MapAnything, Metric3D v2, MoGe-2, UniDepth v2 | fused depth | official repositories | – |
| SEA-RAFT | optical flow | princeton-vl/SEA-RAFT | commit 9137517b |
| RAFT (torchvision), DIS (OpenCV) | optical flow | torchvision, OpenCV | – |
| RobustSplat | 3DGS fits in Sec. 5.2 | official repository | – |
| 3D Gaussian Splatting | basis of RobustSplat | graphdeco-inria/gaussian-splatting | – |
| GeCo | baseline | official repository | commit 2bb2ea9e |
| MEt3R (with MASt3R) | baseline | official repository | MEt3R ee0e1752, MASt3R e06b0093 |
| Easi3R | baseline | official repository | commit 146a77d7 |
| MonST3R | baseline | official repository | commit 574cc77a |
| MegaSaM | baseline | official repository | commit a27b4e63 |
| RoMo, VGGT4D | baselines | official repositories | – |
| CogVideoX1.5-5B, HunyuanVideo-1.5, LTX-2.5, Wan2.2-T2V-A14B | prompt-generated videos (Sec. 5.3) | official releases | – |
| CoTracker3, TAPNext | point tracks (Sec. 5.3) | official repositories | – |
