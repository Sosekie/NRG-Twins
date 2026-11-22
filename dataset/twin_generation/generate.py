"""Keyframe-conditioned segment generation with Wan2.1-Fun-1.3B-InP (paper Sec. 3.1, Supp. A.1).

A twin is built segment by segment: for two consecutive keyframes a < b, Wan2.1-Fun-1.3B-InP generates
the b - a + 1 frames between the real frames a and b (first/last-frame inpainting), and frames a and b
of the output are then replaced by the real frames byte for byte. Keyframes are multiples of 4, so every
segment has 4m + 1 frames as the Wan VAE requires.

Settings used for NRG-Twins (VideoX-Fun commit 13bb3c625e8b1e457d7124d92207b49e00e4a6fe):
848x480, 50 sampling steps (FlowMatchEulerDiscreteScheduler), guidance 6.0, shift 3.0, TeaCache
threshold 0.10 with 5 initial steps not skipped, CFG skip ratio 0.0, bfloat16 weights, the Wan default
negative prompt, one fixed text prompt for every clip, and one seed per clip
(int(md5("RE10K_<sequence id>"), 16) mod 2^31). One torch.Generator is seeded once per generation round
and advanced over the regenerated segments in chronological order. Twins are written as H.264
(libx264, CRF 12, preset fast, yuv420p) at the source frame rate (30 fps).

Needs a CUDA GPU, a VideoX-Fun checkout and the Wan2.1-Fun-1.3B-InP weights
(https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HEIGHT, WIDTH = 480, 848
VIDEOX_FUN_COMMIT = "13bb3c625e8b1e457d7124d92207b49e00e4a6fe"
INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
SHIFT = 3.0
TEACACHE_THRESHOLD = 0.10
TEACACHE_SKIP_START_STEPS = 5
CFG_SKIP_RATIO = 0.0
MEMORY_MODE = "model_cpu_offload"
CRF = 12
PRESET = "fast"
PROMPT = ("A static real-world scene of a real-estate property. The camera moves smoothly; "
          "nothing in the scene moves.")
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def clip_seed(sequence_id: str) -> int:
    """Per-clip generation seed: int(md5('RE10K_' + sequence id), 16) mod 2^31."""
    sid = sequence_id if sequence_id.startswith("RE10K_") else f"RE10K_{sequence_id}"
    return int(hashlib.md5(sid.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def load_pipeline(model_dir: str | Path, videox_fun_dir: str | Path, memory_mode: str = MEMORY_MODE,
                  device: str = "cuda") -> Any:
    """Build the VideoX-Fun WanFunInpaintPipeline with the settings above."""
    import torch

    videox_fun_dir, model_dir = Path(videox_fun_dir), Path(model_dir)
    if str(videox_fun_dir) not in sys.path:
        sys.path.insert(0, str(videox_fun_dir))
    try:
        commit = subprocess.check_output(["git", "-C", str(videox_fun_dir), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    if commit is not None and commit != VIDEOX_FUN_COMMIT:
        print(f"warning: VideoX-Fun at {commit}, NRG-Twins used {VIDEOX_FUN_COMMIT}", file=sys.stderr)

    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from videox_fun.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
    from videox_fun.models.cache_utils import get_teacache_coefficients
    from videox_fun.pipeline import WanFunInpaintPipeline
    from videox_fun.utils import apply_gpu_memory_mode, filter_kwargs

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = OmegaConf.load(str(videox_fun_dir / "config/wan2.1/wan_civitai.yaml"))
    transformer = WanTransformer3DModel.from_pretrained(
        str(model_dir / config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(
        str(model_dir / config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir / config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")))
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(model_dir / config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=dtype).eval()
    clip_image_encoder = CLIPModel.from_pretrained(
        str(model_dir / config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))).to(dtype).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"])))
    pipeline = WanFunInpaintPipeline(transformer=transformer, vae=vae, tokenizer=tokenizer,
                                     text_encoder=text_encoder, scheduler=scheduler,
                                     clip_image_encoder=clip_image_encoder)
    apply_gpu_memory_mode(pipeline, memory_mode, device, dtype)
    pipeline.transformer.enable_teacache(get_teacache_coefficients(str(model_dir)), INFERENCE_STEPS,
                                         TEACACHE_THRESHOLD, num_skip_start_steps=TEACACHE_SKIP_START_STEPS,
                                         offload=False)
    pipeline.transformer.enable_cfg_skip(CFG_SKIP_RATIO, INFERENCE_STEPS)
    return pipeline


def render_segment(pipeline: Any, start_frame: np.ndarray, end_frame: np.ndarray, frame_count: int,
                   generator: Any, prompt: str = PROMPT) -> np.ndarray:
    """Generate `frame_count` RGB uint8 frames (frame_count, 480, 848, 3) from the first and last frame.

    The returned first/last frames are the model's; the caller pastes the real keyframes back.
    """
    import torch
    from PIL import Image
    from videox_fun.utils import get_image_to_video_latent

    if frame_count < 5 or (frame_count - 1) % 4:
        raise ValueError(f"Wan needs 4m+1 frames per segment, got {frame_count}")
    video, mask, clip_image = get_image_to_video_latent(
        [Image.fromarray(start_frame, mode="RGB")], [Image.fromarray(end_frame, mode="RGB")],
        video_length=frame_count, sample_size=[HEIGHT, WIDTH])
    with torch.inference_mode():
        result = pipeline(prompt, num_frames=frame_count, negative_prompt=NEGATIVE_PROMPT, height=HEIGHT, width=WIDTH,
                          generator=generator, guidance_scale=GUIDANCE_SCALE, num_inference_steps=INFERENCE_STEPS,
                          video=video, mask_video=mask, clip_image=clip_image, shift=SHIFT)
    out = result.videos
    frames = (out[0].permute(1, 2, 3, 0).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    if frames.shape != (frame_count, HEIGHT, WIDTH, 3):
        raise RuntimeError(f"unexpected segment shape {frames.shape}")
    return np.ascontiguousarray(frames)


def generate_round(pipeline: Any, real_frames: np.ndarray, segments: list[tuple[int, int]],
                   previous: np.ndarray | None, seed: int, prompt: str = PROMPT, device: str = "cuda") -> np.ndarray:
    """Regenerate `segments` (list of (a, b) keyframe pairs, chronological) on top of `previous`.

    One generator is seeded once with `seed` and advanced over the segments in order. Keyframes a and b
    are copied back from `real_frames` byte for byte.
    """
    import torch

    frames = np.array(previous, copy=True) if previous is not None else np.zeros_like(real_frames)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for a, b in segments:
        seg = render_segment(pipeline, real_frames[a], real_frames[b], b - a + 1, generator, prompt)
        seg[0] = real_frames[a]
        seg[-1] = real_frames[b]
        frames[a:b + 1] = seg
        torch.cuda.empty_cache()
    return frames


def write_video_h264(frames: np.ndarray, path: str | Path, fps: float) -> None:
    """Encode RGB uint8 frames with libx264 (CRF 12, preset fast, yuv420p)."""
    import imageio_ffmpeg

    t, h, w, _ = frames.shape
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{w}x{h}", "-r", str(float(fps)),
           "-i", "-", "-an", "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET, "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(np.ascontiguousarray(frames).tobytes())
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')[-2000:]}")
