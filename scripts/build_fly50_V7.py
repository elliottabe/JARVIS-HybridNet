"""
Snapshot merge_fly50_V6 into merge_fly50_V7 via symlinks + COCO copy.

Creates a fresh dataset root that mirrors merge_fly50_V6's directory layout
using symlinks for image files and verbatim copies of the COCO annotations
and calibration params. Costs ~0 disk and gives a clean iteration surface
without touching the frozen V6 dataset.

Usage:
    python scripts/build_fly50_V7.py
    python scripts/build_fly50_V7.py --src /path/to/V6 --dst /path/to/V7 --force
"""

import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_SRC = Path("/data2/users/eabe/datasets/Johnson_lab/red_data/merge_fly50_V6")
DEFAULT_DST = Path("/data2/users/eabe/datasets/Johnson_lab/red_data/merge_fly50_V7")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    p.add_argument("--force", action="store_true",
                   help="Overwrite destination if it exists")
    return p.parse_args()


def symlink_tree(src_dir: Path, dst_dir: Path) -> int:
    """Recreate the directory tree under dst_dir, symlinking every file
    to its counterpart under src_dir. Returns the number of files linked."""
    n = 0
    for root, _dirs, files in os.walk(src_dir):
        rel = Path(root).relative_to(src_dir)
        out_dir = dst_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src_file = Path(root) / fname
            dst_file = out_dir / fname
            if dst_file.exists() or dst_file.is_symlink():
                continue
            dst_file.symlink_to(src_file.resolve())
            n += 1
    return n


def main():
    args = parse_args()
    src, dst = args.src, args.dst

    if not src.is_dir():
        raise SystemExit(f"Source dataset not found: {src}")

    if dst.exists():
        if not args.force:
            raise SystemExit(
                f"Destination already exists: {dst}\nUse --force to overwrite."
            )
        shutil.rmtree(dst)

    dst.mkdir(parents=True)

    # 1. Symlink train/ and val/ image trees
    for split in ("train", "val"):
        src_split = src / split
        if not src_split.is_dir():
            print(f"[skip] {src_split} does not exist")
            continue
        n = symlink_tree(src_split, dst / split)
        print(f"[symlink] {split}: {n} files")

    # 2. Copy annotations verbatim
    src_ann = src / "annotations"
    dst_ann = dst / "annotations"
    dst_ann.mkdir(exist_ok=True)
    for f in src_ann.iterdir():
        shutil.copy2(f, dst_ann / f.name)
        print(f"[copy] annotations/{f.name}")

    # 3. Copy calib_params verbatim
    src_calib = src / "calib_params"
    if src_calib.is_dir():
        shutil.copytree(src_calib, dst / "calib_params")
        print(f"[copy] calib_params/")

    # 4. Verification: count images in COCO and a random symlink probe
    for split in ("train", "val"):
        ann_path = dst_ann / f"instances_{split}.json"
        if not ann_path.exists():
            continue
        with open(ann_path) as fh:
            d = json.load(fh)
        print(f"[verify] {split}: {len(d['images'])} images, "
              f"{len(d['annotations'])} annotations")

    print(f"\nDone. New dataset at: {dst}")


if __name__ == "__main__":
    main()
