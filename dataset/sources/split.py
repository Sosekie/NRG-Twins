"""Split source clips into train / val / test by YouTube video (Supp. A.7).

Clips that come from the same YouTube video always land in the same split. A clip may carry several
YouTube ids (when metadata sources disagree); all clips sharing any id are merged into one group
(union-find). Groups are sorted by their smallest YouTube id (groups without an id by clip id), shuffled
with numpy.random.default_rng(20261109), and assigned in that order: val until it holds at least
round(N * 0.1) clips, then test until it holds at least round(N * 0.1) clips, the rest train
(N = number of clips; 8:1:1). Groups listed with --force-train are put in train before the shuffle.

For the released dataset, groups that share a YouTube id with clips used for training in earlier
experiments were also forced into train; pass such groups with --force-train. The released split itself
is given in splits/*.txt of the dataset.

Input: TSV with columns clip_id and youtube_id (several ids separated by ';').
Example:
    python -m dataset.sources.split clips.tsv --out split.tsv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20261109


def norm_id(value: str) -> str:
    value = str(value).strip()
    return value.removeprefix("RE10K_")


def group_by_youtube(youtube: dict[str, set[str]]) -> dict[str, list[str]]:
    """Connected components of clips linked by shared YouTube ids; key = smallest clip id of the group."""
    parent = {c: c for c in youtube}

    def find(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    first_clip: dict[str, str] = {}
    for c in sorted(youtube):
        for y in sorted(youtube[c]):
            if y in first_clip:
                a, b = find(c), find(first_clip[y])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first_clip[y] = c
    members: dict[str, list[str]] = defaultdict(list)
    for c in sorted(youtube):
        members[find(c)].append(c)
    return dict(members)


def make_split(youtube: dict[str, set[str]], force_train: set[str] = frozenset(), seed: int = SEED,
               val_fraction: float = 0.1, test_fraction: float = 0.1) -> dict[str, str]:
    members = group_by_youtube(youtube)
    n = len(youtube)
    n_val, n_test = round(n * val_fraction), round(n * test_fraction)

    def key(g: str) -> tuple:
        ids = sorted({y for c in members[g] for y in youtube[c]})
        return (0, ids[0]) if ids else (1, members[g][0])

    split = {}
    forced = {g for g, cs in members.items() if any(c in force_train for c in cs)}
    for g in forced:
        for c in members[g]:
            split[c] = "train"
    rest = sorted((g for g in members if g not in forced), key=key)
    np.random.default_rng(seed).shuffle(rest)
    n_v = n_t = 0
    for g in rest:
        if n_v < n_val:
            s = "val"
            n_v += len(members[g])
        elif n_t < n_test:
            s = "test"
            n_t += len(members[g])
        else:
            s = "train"
        for c in members[g]:
            split[c] = s
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", type=Path, help="TSV with columns clip_id, youtube_id")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force-train", type=Path, default=None, help="text file of clip ids that must be train")
    args = ap.parse_args()
    youtube: dict[str, set[str]] = defaultdict(set)
    with args.clips.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            c = norm_id(row["clip_id"])
            youtube[c].update(y for y in str(row.get("youtube_id") or "").split(";") if y.strip())
    force = set()
    if args.force_train:
        force = {norm_id(x) for x in args.force_train.read_text().splitlines() if x.strip()}
    split = make_split(dict(youtube), force, args.seed)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["clip_id", "youtube_id", "split"])
        for c in sorted(split):
            w.writerow([c, ";".join(sorted(youtube[c])), split[c]])
    print(dict(Counter(split.values())))


if __name__ == "__main__":
    main()
