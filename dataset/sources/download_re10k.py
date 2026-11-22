"""Download RealEstate10K (pixelSplat frames, WebDataset shards) and extract 81-frame source clips (Supp. A.7).

NRG-Twins uses the RealEstate10K frames prepared by pixelSplat (640x360 JPEGs with per-frame timestamps
and cameras), read from the WebDataset mirror luh0502/re10k-wds on the Hugging Face Hub at revision
9d106a15d1f484c9918e0a3d62b06d5dfb8f4936. Each shard member group <seq>.meta.json, <seq>.timestamps.npy,
<seq>.cameras.npy, <seq>.<NNNNN>.jpg holds one sequence.

Clip extraction (as run for the dataset):
  * the sequence must be 640x360 in its meta.json;
  * the window is the FIRST run of 81 consecutive frames whose 80 timestamp steps all lie in
    [33000, 34000] microseconds (30 fps without dropped frames); sequences without one are skipped;
  * the 81 JPEGs are stored unchanged (frames/000000.jpg ... 000080.jpg), with timestamps_us.json,
    cameras_wds.npy (the 81 camera rows) and meta.json;
  * video.mp4: the frames resized to 848x480 (cv2.INTER_LINEAR), H.264 (libx264, default CRF, yuv420p),
    30 fps. This video is the real clip used for cameras, labels and NRG-PD;
  * optionally, with --annotation-dir, the timestamps and normalised intrinsics are checked against the
    original RealEstate10K camera files (<seq>.txt): timestamps equal, intrinsics within 1e-7.
Not included here: the 2D shake/slow-motion screen that removed clips with a 2D rotation residual
>= 0.162 deg or a 2D speed < 0.66 px/frame (see the paper supplement).

Examples:
    python -m dataset.sources.download_re10k download --out re10k_wds --split test
    python -m dataset.sources.download_re10k extract --shards re10k_wds/test/*.tar --out clips
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ID = "luh0502/re10k-wds"
REVISION = "9d106a15d1f484c9918e0a3d62b06d5dfb8f4936"
FRAMES = 81
STEP_US = (33000, 34000)
OUT_WH = (848, 480)
SOURCE_WH = (640, 360)
FPS = 30


def download(out: Path, splits: list[str], revision: str = REVISION) -> None:
    """Mirror the metadata and the shards of the given splits ('train', 'test')."""
    from huggingface_hub import snapshot_download

    patterns = ["README.md", "splits/*"] + [f"{s}/*" for s in splits]
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=revision, local_dir=str(out),
                      allow_patterns=patterns)


def first_window(timestamps_us: np.ndarray, length: int = FRAMES) -> int | None:
    """Start of the first `length`-frame window whose steps all lie in [33000, 34000] us."""
    dt = np.diff(np.asarray(timestamps_us, np.int64))
    ok = (dt >= STEP_US[0]) & (dt <= STEP_US[1])
    for s in range(max(0, len(timestamps_us) - (length - 1))):
        if bool(np.all(ok[s:s + length - 1])):
            return s
    return None


def encode_clip(jpegs: list[bytes], path: Path, fps: int = FPS) -> None:
    """848x480 H.264 clip (libx264 default CRF, yuv420p) from JPEG bytes, cv2.INTER_LINEAR resize."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{OUT_WH[0]}x{OUT_WH[1]}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for b in jpegs:
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        proc.stdin.write(cv2.resize(im, OUT_WH, interpolation=cv2.INTER_LINEAR).tobytes())
    proc.stdin.close()
    err = proc.stderr.read()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')[-2000:]}")


def check_annotation(seq: str, ts: np.ndarray, cams: np.ndarray, annotation_dir: Path) -> str | None:
    p = annotation_dir / f"{seq}.txt"
    if not p.is_file():
        return "annotation_missing"
    ann = np.loadtxt(p, skiprows=1, dtype=np.float64, ndmin=2)
    if ann.shape[0] != len(ts) or cams.shape != (len(ts), 18):
        return "metadata_length_mismatch"
    if not np.array_equal(ts, ann[:, 0].astype(np.int64)):
        return "timestamp_mismatch"
    if float(np.max(np.abs(cams[:, :4].astype(np.float64) - ann[:, 1:5]))) > 1e-7:
        return "intrinsics_mismatch"
    return None


def extract_sequence(seq: str, members: dict[str, bytes], out: Path, annotation_dir: Path | None) -> dict:
    meta = json.loads(members["meta.json"])
    if (meta.get("width"), meta.get("height")) != SOURCE_WH:
        return {"sequence_id": seq, "status": "not_640x360"}
    ts = np.load(io.BytesIO(members["timestamps.npy"]), allow_pickle=False).astype(np.int64)
    cams = np.load(io.BytesIO(members["cameras.npy"]), allow_pickle=False)
    if annotation_dir is not None:
        bad = check_annotation(seq, ts, cams, annotation_dir)
        if bad:
            return {"sequence_id": seq, "status": bad}
    start = first_window(ts)
    if start is None:
        return {"sequence_id": seq, "status": "no_81_frame_30fps_window"}
    names = [f"{start + j:05d}.jpg" for j in range(FRAMES)]
    if any(n not in members for n in names):
        return {"sequence_id": seq, "status": "missing_frames"}
    jpegs = [members[n] for n in names]
    for b in jpegs:
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (SOURCE_WH[1], SOURCE_WH[0]):
            return {"sequence_id": seq, "status": "bad_jpeg"}
    d = out / seq
    (d / "frames").mkdir(parents=True, exist_ok=True)
    for j, b in enumerate(jpegs):
        (d / "frames" / f"{j:06d}.jpg").write_bytes(b)
    (d / "timestamps_us.json").write_text(json.dumps([int(x) for x in ts[start:start + FRAMES]]) + "\n")
    np.save(d / "cameras_wds.npy", cams[start:start + FRAMES], allow_pickle=False)
    (d / "meta.json").write_text(json.dumps({**meta, "window_start_frame": int(start), "frames": FRAMES}) + "\n")
    encode_clip(jpegs, d / "video.mp4")
    youtube_id = str(meta.get("url", "")).split("v=")[-1] or None
    return {"sequence_id": seq, "status": "ok", "window_start_frame": int(start), "youtube_id": youtube_id}


def iter_sequences(shard: Path, wanted: set[str] | None):
    """Yield (sequence id, {suffix: bytes}) for each sequence in a WebDataset tar (members grouped by key)."""
    groups: dict[str, dict[str, bytes]] = defaultdict(dict)
    order: list[str] = []
    with tarfile.open(shard) as tar:
        for m in tar:
            if not m.isfile():
                continue
            name = Path(m.name).name
            seq, _, suffix = name.partition(".")
            if wanted is not None and seq not in wanted:
                continue
            if seq not in groups:
                order.append(seq)
            groups[seq][suffix] = tar.extractfile(m).read()
    for seq in order:
        yield seq, groups.pop(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="mirror the WebDataset shards from the Hugging Face Hub")
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--split", nargs="+", default=["train", "test"], choices=["train", "test"])
    d.add_argument("--revision", default=REVISION)
    e = sub.add_parser("extract", help="extract 81-frame source clips from shards")
    e.add_argument("--shards", nargs="+", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--sequence-ids", type=Path, default=None, help="text file, one sequence id per line")
    e.add_argument("--annotation-dir", type=Path, default=None, help="RealEstate10K <seq>.txt camera files")
    args = ap.parse_args()
    if args.cmd == "download":
        download(args.out, args.split, args.revision)
        return
    wanted = None
    if args.sequence_ids:
        wanted = {x.strip().removeprefix("RE10K_") for x in args.sequence_ids.read_text().splitlines() if x.strip()}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "extraction_log.jsonl").open("a") as log:
        for shard in args.shards:
            for seq, members in iter_sequences(shard, wanted):
                if "meta.json" not in members:
                    continue
                rec = extract_sequence(seq, members, args.out, args.annotation_dir)
                log.write(json.dumps({**rec, "shard": shard.name}) + "\n")
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
