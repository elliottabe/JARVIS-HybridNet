"""
Build a CenterDetect-only dataset from merge_courtship_multianimal_V1_collapsed
that keeps ONLY frames where both flies are labeled (image_id has 2 annotations).

Purpose: train CenterDetect from scratch so every training frame produces a
2-peak target heatmap via np.maximum overlay (dataset2D.py:349). No frame in
the resulting dataset has the "1 ann on a 2-fly scene" suppression bug.

Output dataset symlinks train/, val/, calib_params/ from the source.
"""

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("/data2/users/eabe/datasets/Johnson_lab/red_data/merge_courtship_multianimal_V1_collapsed")
DST = Path("/data2/users/eabe/datasets/Johnson_lab/red_data/merge_courtship_dual_only")


def filter_dual(coco: dict) -> dict:
    ann_per = defaultdict(list)
    for a in coco["annotations"]:
        ann_per[a["image_id"]].append(a)
    keep_ids = {iid for iid, anns in ann_per.items() if len(anns) == 2}
    new_imgs = [i for i in coco["images"] if i["id"] in keep_ids]
    new_anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
    out = dict(coco)
    out["images"] = new_imgs
    out["annotations"] = new_anns
    return out


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    for sub in ("train", "val", "calib_params"):
        s = SRC / sub
        if s.exists():
            (DST / sub).symlink_to(s.resolve())
            print(f"[symlink] {sub} -> {s.resolve()}")

    ann_dir = DST / "annotations"
    ann_dir.mkdir()
    for split in ("train", "val"):
        with open(SRC / "annotations" / f"instances_{split}.json") as f:
            coco = json.load(f)
        before = (len(coco["images"]), len(coco["annotations"]))
        new = filter_dual(coco)
        after = (len(new["images"]), len(new["annotations"]))
        with open(ann_dir / f"instances_{split}.json", "w") as f:
            json.dump(new, f)
        print(f"[{split}] images {before[0]} -> {after[0]}, annotations {before[1]} -> {after[1]}")
    print(f"\nDone. Dataset at: {DST}")


if __name__ == "__main__":
    main()
