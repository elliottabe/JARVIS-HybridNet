"""
Collapse a multi-animal COCO so that image entries pointing at the same
physical jpg (same camera + frame number, byte-identical file) are merged
into a single image_id with all of their annotations attached.

Source dataset: merge_courtship_multianimal_V1
- Each (cam, frame) pair currently has up to 2 image entries (one with the
  "male" labelling-task session prefix, one with "female"), each carrying 1
  annotation for its own fly. The two file paths point at byte-identical jpgs
  but JARVIS Dataset2D._get_item_center loads annotations per image_id, so the
  np.maximum heatmap overlay never fires on dual-labeled frames.

After collapse: one image_id per physical frame, with 1 or 2 annotations.
JARVIS CenterDetect training then learns multi-peak heatmaps directly.

The output dataset symlinks train/ val/ calib_params/ from the source so we
don't duplicate jpgs on disk.
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SRC = Path("/data2/users/eabe/datasets/Johnson_lab/red_data/merge_courtship_multianimal_V1")
DST = Path("/data2/users/eabe/datasets/Johnson_lab/red_data/merge_courtship_multianimal_V1_collapsed")


def physical_key(file_name: str) -> tuple:
    """(cam, frame_filename) — ignores session prefix."""
    parts = file_name.split("/")
    return (parts[1], parts[2])


def collapse(coco: dict) -> dict:
    images = coco["images"]
    annotations = coco["annotations"]

    # Map image_id -> physical key
    id_to_key = {img["id"]: physical_key(img["file_name"]) for img in images}

    # For each physical key, choose the canonical image (lowest id)
    key_to_imgs = defaultdict(list)
    for img in images:
        key_to_imgs[physical_key(img["file_name"])].append(img)

    canonical = {}
    for key, group in key_to_imgs.items():
        group.sort(key=lambda i: i["id"])
        canonical[key] = group[0]

    # Build remap: old image_id -> canonical image_id
    remap = {}
    for img in images:
        canon_img = canonical[physical_key(img["file_name"])]
        remap[img["id"]] = canon_img["id"]

    # Rewrite annotations
    new_anns = []
    for a in annotations:
        a2 = dict(a)
        a2["image_id"] = remap[a["image_id"]]
        new_anns.append(a2)

    # Drop redundant images, keep only canonical entries
    new_imgs = list(canonical.values())

    out = dict(coco)
    out["images"] = new_imgs
    out["annotations"] = new_anns
    return out


def symlink_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    src, dst = args.src, args.dst

    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    if dst.exists():
        if not args.force:
            raise SystemExit(f"Destination exists: {dst}\nUse --force to overwrite.")
        shutil.rmtree(dst)

    dst.mkdir(parents=True)

    # Symlink train/, val/, calib_params/ — same physical jpgs
    for sub in ("train", "val", "calib_params"):
        s = src / sub
        if s.exists():
            (dst / sub).symlink_to(s.resolve())
            print(f"[symlink] {sub} -> {s}")

    # Collapse train + val annotations
    ann_dir = dst / "annotations"
    ann_dir.mkdir()
    for split in ("train", "val"):
        coco_path = src / "annotations" / f"instances_{split}.json"
        with open(coco_path) as fh:
            coco = json.load(fh)
        before_imgs = len(coco["images"])
        before_anns = len(coco["annotations"])
        new_coco = collapse(coco)
        after_imgs = len(new_coco["images"])
        after_anns = len(new_coco["annotations"])

        ann_per_img = Counter()
        for a in new_coco["annotations"]:
            ann_per_img[a["image_id"]] += 1
        dist = Counter(ann_per_img.values())

        out_path = ann_dir / f"instances_{split}.json"
        with open(out_path, "w") as fh:
            json.dump(new_coco, fh)
        print(
            f"[{split}] images {before_imgs} -> {after_imgs}, "
            f"annotations {before_anns} -> {after_anns}, "
            f"anns/img distribution: {dict(dist)}"
        )

    print(f"\nDone. New dataset at: {dst}")


if __name__ == "__main__":
    main()
