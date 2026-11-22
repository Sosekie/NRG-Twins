"""Adaptive keyframe insertion: build one NRG twin of an 81-frame RealEstate10K clip (paper Sec. 3.1, Supp. A.1).

Round 0 generates frames 0..80 from the real keyframes 0 and 80 (dataset/twin_generation/generate.py).
After every round, VGGT-1B estimates per-frame cameras of the real clip and of the twin, the twin is
aligned by the rotation Q fitted on the current keyframes, and the rotation gap delta_t is computed
(dataset/twin_generation/rotation_gap.py). If max_t delta_t <= 0.8 deg the twin is final. Otherwise the
real frame chosen by rotation_gap.choose_keyframe becomes a keyframe and only the two segments adjacent
to it are regenerated; all other frames are kept. At most 6 insertions. Keyframes are byte-identical
copies of the real frames.

Cameras: VGGT-1B (facebook/VGGT-1B, code commit a288dd0f14786c93483e45524328726ab7b1b4ce), camera head
only, bfloat16 autocast, on the frames decoded from the H.264 video (real clip: the source mp4; twin: the
encoded twin mp4), preprocessed with vggt.utils.load_fn.load_and_preprocess_images(mode="crop")
(width 518), converted with pose_encoding_to_extri_intri.

Input clip directory (from dataset/sources/download_re10k.py): frames/000000.jpg ... 000080.jpg,
video.mp4 (30 fps, 848x480). Outputs in --out: round_<k>/{frames.npy, video.mp4, cameras.npz, gap.json},
real_cameras.npz, twin.mp4 (final), keyframes.json.

Example:
    python -m dataset.twin_generation.insert_keyframes --clip-dir clips/0a72cf3652ccf6f7 \
        --out twins/0a72cf3652ccf6f7 --wan-dir models/Wan2.1-Fun-1.3B-InP --videox-fun-dir VideoX-Fun \
        --vggt-dir vggt --vggt-weights models/VGGT-1B/model.safetensors
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from dataset.twin_generation import generate as gen
from dataset.twin_generation.rotation_gap import (FRAME_COUNT, INITIAL_KEYFRAMES, MAX_INSERTIONS, decide,
                                                  rotation_gap)

TARGET_WH = (848, 480)
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"


def load_source_frames(frame_dir: Path) -> np.ndarray:
    """81 source JPEGs -> RGB uint8 (81, 480, 848, 3); INTER_AREA when shrinking, INTER_LINEAR otherwise."""
    paths = sorted(frame_dir.glob("*.jpg"))
    if len(paths) != FRAME_COUNT:
        raise ValueError(f"expected {FRAME_COUNT} JPEGs in {frame_dir}, found {len(paths)}")
    out = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        interp = cv2.INTER_AREA if bgr.shape[1] > TARGET_WH[0] or bgr.shape[0] > TARGET_WH[1] else cv2.INTER_LINEAR
        out.append(cv2.cvtColor(cv2.resize(bgr, TARGET_WH, interpolation=interp), cv2.COLOR_BGR2RGB))
    return np.ascontiguousarray(np.stack(out), dtype=np.uint8)


class VGGTCameras:
    """Camera-only VGGT-1B inference on an 81-frame video."""

    def __init__(self, vggt_dir: str | Path, weights: str | Path, device: str = "cuda"):
        import torch
        from safetensors.torch import load_file

        sys.path.insert(0, str(vggt_dir))
        from vggt.models.vggt import VGGT

        self.torch, self.device = torch, torch.device(device)
        model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
        state = load_file(str(weights), device="cpu")
        missing = model.load_state_dict(state, strict=False).missing_keys
        if missing:
            raise RuntimeError(f"VGGT weights are missing keys: {missing[:5]}")
        self.model = model.eval().to(self.device)

    def __call__(self, video_path: str | Path) -> np.ndarray:
        """World-to-camera poses (81, 4, 4) of the decoded video."""
        from PIL import Image
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            cap = cv2.VideoCapture(str(video_path))
            paths = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                p = Path(tmp) / f"frame_{len(paths):06d}.png"
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mode="RGB").save(p, format="PNG", optimize=False)
                paths.append(str(p))
            cap.release()
            if len(paths) != FRAME_COUNT:
                raise RuntimeError(f"expected {FRAME_COUNT} frames in {video_path}, got {len(paths)}")
            images = load_and_preprocess_images(paths, mode="crop")
        hw = (int(images.shape[-2]), int(images.shape[-1]))
        images = images.unsqueeze(0).to(device=self.device, dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = self.model(images)
        extr, _ = pose_encoding_to_extri_intri(pred["pose_enc"].float(), image_size_hw=hw)
        w2c = np.repeat(np.eye(4, dtype=np.float32)[None], FRAME_COUNT, axis=0)
        w2c[:, :3, :4] = extr.squeeze(0).float().cpu().numpy()
        del images, pred
        gc.collect()
        torch.cuda.empty_cache()
        return w2c


def segments_for(round_index: int, keyframes: list[int], inserted: int | None) -> list[tuple[int, int]]:
    if round_index == 0:
        return [(keyframes[0], keyframes[-1])]
    old = [k for k in keyframes if k != inserted]
    left = max(k for k in old if k < inserted)
    right = min(k for k in old if k > inserted)
    return [(left, inserted), (inserted, right)]


def run(clip_dir: Path, out: Path, pipeline, cameras: VGGTCameras, seed: int, fps: float) -> dict:
    real = load_source_frames(clip_dir / "frames")
    real_video = clip_dir / "video.mp4"
    out.mkdir(parents=True, exist_ok=True)
    real_w2c = cameras(real_video)
    np.savez(out / "real_cameras.npz", w2c_vggt=real_w2c)

    keyframes, inserted, frames, history = list(INITIAL_KEYFRAMES), None, None, []
    for k in range(MAX_INSERTIONS + 1):
        if k > 0:
            keyframes = sorted(set(keyframes) | {inserted})
        frames = gen.generate_round(pipeline, real, segments_for(k, keyframes, inserted), frames, seed)
        for i in keyframes:
            if not np.array_equal(frames[i], real[i]):
                raise RuntimeError(f"keyframe {i} is not byte-identical")
        rdir = out / f"round_{k}"
        rdir.mkdir(exist_ok=True)
        np.save(rdir / "frames.npy", frames)
        gen.write_video_h264(frames, rdir / "video.mp4", fps)
        twin_w2c = cameras(rdir / "video.mp4")
        np.savez(rdir / "cameras.npz", w2c_vggt=twin_w2c)
        delta, q = rotation_gap(real_w2c, twin_w2c, keyframes)
        step = decide(delta, keyframes, k)
        (rdir / "gap.json").write_text(json.dumps({**step, "Q": q.tolist(), "gap_deg": delta.tolist()}, indent=1))
        history.append(step)
        print(f"round {k}: keyframes {keyframes} max gap {step['max_gap_deg']:.3f} deg at frame "
              f"{step['max_gap_frame']} -> {step['stop_reason'] or 'insert ' + str(step['next_keyframe'])}", flush=True)
        if step["stop_reason"]:
            break
        inserted = step["next_keyframe"]
    shutil.copyfile(out / f"round_{len(history) - 1}" / "video.mp4", out / "twin.mp4")
    summary = {"seed": seed, "prompt": gen.PROMPT, "final_keyframes": keyframes,
               "stop_reason": history[-1]["stop_reason"], "rounds": history}
    (out / "keyframes.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-dir", required=True, type=Path, help="clip folder with frames/*.jpg and video.mp4")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--wan-dir", required=True, help="local Wan2.1-Fun-1.3B-InP weights")
    ap.add_argument("--videox-fun-dir", required=True, help="VideoX-Fun checkout")
    ap.add_argument("--vggt-dir", required=True, help="VGGT code checkout (facebookresearch/vggt)")
    ap.add_argument("--vggt-weights", required=True, help="VGGT-1B model.safetensors")
    ap.add_argument("--sequence-id", default=None, help="RealEstate10K sequence id (default: clip folder name)")
    ap.add_argument("--seed", type=int, default=None, help="override the per-clip seed")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--memory-mode", default=gen.MEMORY_MODE)
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else gen.clip_seed(args.sequence_id or args.clip_dir.name)
    cameras = VGGTCameras(args.vggt_dir, args.vggt_weights)
    pipeline = gen.load_pipeline(args.wan_dir, args.videox_fun_dir, args.memory_mode)
    run(args.clip_dir, args.out, pipeline, cameras, seed, args.fps)


if __name__ == "__main__":
    main()
