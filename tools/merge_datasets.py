"""
Merge one or more single-fly and (collapsed) multi-fly JARVIS datasets into a
single unified project.

By default merges merge_fly50_V7 (single) + merge_courtship_multianimal_V1_
collapsed (multi). Pass --single-roots / --ma-roots to merge any number of
sources, e.g. add a freshly collapsed batch:

  --single-roots .../merge_fly50_V7 \\
  --ma-roots .../merge_courtship_multianimal_V1_collapsed \\
             .../merge_multianimal_collapsed \\
  --out-root .../red_data_unified_V2

Multi-fly (--ma-roots) sources must already be COLLAPSED (one image_id per
physical frame carrying 1-2 annotations); use scripts/collapse_multianimal_coco.py
first. Canonical selection prefers ma-root entries (they carry the dual labels).

Deduplicates by frameset content. A "frameset" is all 7 cameras captured at one
moment. Within each source, framesets are keyed by (timestamp, frame_number).
We MD5-hash every image, then union-find framesets that share at least one
(cam, hash) pair — these are the same underlying physical moment, possibly
re-labeled in multiple (timestamp, frame_number) slots.

For each unified frameset:
  - Canonical (timestamp, frame_num): the member with the richest annotations,
    preferring MA over V7 (MA adds dual-fly labels).
  - Per-camera image: pick the entry whose annotations are richest (MA preferred).

Splits are rebuilt at the frameset level: a random dual-fly test holdout is
carved first (so no train/val/test leakage across the two source datasets),
then a stratified val split from the remainder.

Emits:
  <out>/{train,val,test}/<ts>/<cam>/Frame_<N>.jpg       (file copies)
  <out>/annotations/instances_{train,val,test}.json
  <out>/calib_params/<ts>/*.yaml                        (file copies)
  <out>/dedup_report.json
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

# Two annotations on the SAME physical image are treated as the same animal
# (a duplicate label) when their mean per-keypoint pixel offset is below this.
# Distinct flies in a courtship pair sit ~100-200px apart, so 5px cleanly
# separates "re-label of the same fly" from "the other fly".
DEDUP_PX = 5.0


def _same_animal(a, b):
    ka = np.array(a["keypoints"], dtype=float).reshape(-1, 3)[:, :2]
    kb = np.array(b["keypoints"], dtype=float).reshape(-1, 3)[:, :2]
    diff = np.abs(ka - kb)
    if np.all(np.isnan(diff)):
        return False
    return np.nanmean(diff) < DEDUP_PX


def _union_annotations(entries):
    """Merge annotations across entries of the SAME physical image, dropping
    near-duplicate labels of the same animal. Recovers a second fly that was
    split into a different source/split for the same frame."""
    merged = []
    for e in entries:
        for a in e["annotations"]:
            if not any(_same_animal(a, m) for m in merged):
                merged.append(a)
    return merged


def md5_of_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def parse_filename(file_name: str):
    ts, cam, frame = file_name.split("/")
    n = int(frame.replace("Frame_", "").replace(".jpg", ""))
    return ts, cam, n


def load_source(root: Path, source_name: str):
    """Return {(source, ts, frame_num): {cam: entry}} and full JSON blobs."""
    framesets = defaultdict(dict)
    json_by_split = {}
    for split in ("train", "val"):
        jpath = root / "annotations" / f"instances_{split}.json"
        with jpath.open() as f:
            blob = json.load(f)
        json_by_split[split] = blob
        anns_by_img = defaultdict(list)
        for a in blob["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        for im in blob["images"]:
            ts, cam, n = parse_filename(im["file_name"])
            abs_path = root / split / im["file_name"]
            entry = {
                "source": source_name,
                "split": split,
                "timestamp": ts,
                "cam": cam,
                "frame_num": n,
                "abs_path": abs_path,
                "image_id": im["id"],
                "width": im["width"],
                "height": im["height"],
                "annotations": anns_by_img.get(im["id"], []),
            }
            framesets[(source_name, ts, n)][cam] = entry
    return framesets, json_by_split


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-roots", type=Path, nargs="*",
                    default=[Path("/data2/users/eabe/datasets/Johnson_lab/red_data/merge_fly50_V7")],
                    help="one or more single-fly dataset roots")
    ap.add_argument("--ma-roots", type=Path, nargs="*",
                    default=[Path("/data2/users/eabe/datasets/Johnson_lab/red_data/merge_courtship_multianimal_V1_collapsed")],
                    help="one or more COLLAPSED multi-fly dataset roots")
    ap.add_argument("--out-root", type=Path,
                    default=Path("/data2/users/eabe/datasets/Johnson_lab/red_data/red_data_unified"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-dual-frac", type=float, default=0.20)
    ap.add_argument("--test-single-frac", type=float, default=0.00)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Build the source list: unique source name per root, tagged single/MA.
    sources = []  # (source_name, root, is_ma)
    for root in args.single_roots:
        sources.append((f"single:{root.name}", root, False))
    for root in args.ma_roots:
        sources.append((f"ma:{root.name}", root, True))
    if not sources:
        raise SystemExit("No sources given (need --single-roots and/or --ma-roots).")

    is_ma_by_source = {name: is_ma for name, _, is_ma in sources}
    src_root_by_source = {name: root for name, root, _ in sources}

    print("[1/7] Loading sources …")
    all_fs = {}
    ref_json = None
    fs_count_by_source = {}
    total_imgs = 0
    for name, root, is_ma in sources:
        fs, jblob = load_source(root, name)
        # source names are unique, so frameset keys never collide across roots
        all_fs.update(fs)
        n_fs = len(fs)
        n_imgs = sum(len(v) for v in fs.values())
        fs_count_by_source[name] = n_fs
        total_imgs += n_imgs
        print(f"  {name} ({'multi' if is_ma else 'single'}): "
              f"{n_fs} framesets / {n_imgs} images")
        if ref_json is None:
            ref_json = jblob
        else:
            for key in ("keypoint_names", "skeleton"):
                if jblob["train"].get(key) != ref_json["train"].get(key):
                    raise SystemExit(
                        f"{key} differ between {sources[0][0]} and {name}; abort.")

    keypoint_names = ref_json["train"]["keypoint_names"]
    skeleton = ref_json["train"]["skeleton"]
    categories = ref_json["train"]["categories"]

    print("[2/7] Hashing images …")
    n = total_imgs
    i = 0
    for fs_key, cams in all_fs.items():
        for cam, entry in cams.items():
            if not entry["abs_path"].exists():
                raise SystemExit(f"Missing file: {entry['abs_path']}")
            entry["md5"] = md5_of_file(entry["abs_path"])
            i += 1
            if i % 2000 == 0:
                print(f"  {i}/{n}")
    print(f"  hashed {n}")

    print("[3/7] Linking framesets that share content …")
    uf = UnionFind()
    hash_to_fs = defaultdict(list)
    for fs_key, cams in all_fs.items():
        uf.find(fs_key)
        for cam, entry in cams.items():
            hash_to_fs[entry["md5"]].append(fs_key)

    for fs_keys in hash_to_fs.values():
        if len(fs_keys) > 1:
            root = fs_keys[0]
            for k in fs_keys[1:]:
                uf.union(root, k)

    groups = defaultdict(list)
    for fs_key in all_fs:
        groups[uf.find(fs_key)].append(fs_key)
    print(f"  unified framesets: {len(groups)}")
    size_dist = defaultdict(int)
    for g in groups.values():
        size_dist[len(g)] += 1
    print(f"  group size dist (n_source_fs -> count): {dict(sorted(size_dist.items()))}")

    print("[4/7] Building canonical framesets …")
    unified_framesets = []  # list of dicts with canonical metadata

    def fs_rank(fs_key):
        cams = all_fs[fs_key]
        n_ann = sum(len(e["annotations"]) for e in cams.values())
        n_cams = len(cams)
        is_ma = 1 if is_ma_by_source[fs_key[0]] else 0
        # Prefer MA > annotations > cams
        return (is_ma, n_ann, n_cams)

    for group_root, members in groups.items():
        members.sort(key=fs_rank, reverse=True)
        canon_key = members[0]

        # Pool all entries across members by cam; pick best entry per cam.
        per_cam = defaultdict(list)
        for fs_key in members:
            for cam, entry in all_fs[fs_key].items():
                per_cam[cam].append(entry)

        def entry_rank(e):
            return (1 if is_ma_by_source[e["source"]] else 0, len(e["annotations"]))

        chosen = {}
        for cam, entries in per_cam.items():
            entries.sort(key=entry_rank, reverse=True)
            canon = dict(entries[0])
            # Union labels from every entry that is the SAME physical image
            # (identical md5) as the canonical pick, so a fly that was labeled
            # in a different split/source for this frame is not dropped. Entries
            # whose image differs (different md5) are left out of the union.
            same_image = [e for e in entries if e["md5"] == canon["md5"]]
            canon["annotations"] = _union_annotations(same_image)
            chosen[cam] = canon

        unified_framesets.append({
            "canonical_source": canon_key[0],
            "canonical_timestamp": canon_key[1],
            "canonical_frame_num": canon_key[2],
            "members": members,
            "cams": chosen,
        })

    incomplete = [u for u in unified_framesets if len(u["cams"]) != 7]
    print(f"  unified: {len(unified_framesets)}")
    print(f"  incomplete (!=7 cams): {len(incomplete)}")
    for u in incomplete[:5]:
        print(f"    {u['canonical_source']}/{u['canonical_timestamp']}/Frame_{u['canonical_frame_num']} cams={sorted(u['cams'])}")

    expected_cams = set()
    for u in unified_framesets:
        expected_cams.update(u["cams"].keys())
    expected_cams_sorted = sorted(expected_cams)

    # Drop incomplete framesets (HybridNet 3D needs all cams; EfficientTrack 2D
    # would also have inconsistent coverage). Log them for reference.
    complete = [u for u in unified_framesets if len(u["cams"]) == 7]
    print(f"  keeping {len(complete)} complete framesets (dropping {len(incomplete)})")

    print("[5/7] Splitting …")

    def is_dual(u):
        return any(len(e["annotations"]) >= 2 for e in u["cams"].values())

    dual = [u for u in complete if is_dual(u)]
    single = [u for u in complete if not is_dual(u)]
    print(f"  dual: {len(dual)}   single: {len(single)}")

    rng.shuffle(dual)
    rng.shuffle(single)

    def split_list(items, test_frac, val_frac):
        n = len(items)
        n_test = int(round(n * test_frac))
        rem = items[n_test:]
        n_val = int(round(len(rem) * val_frac))
        return rem[n_val:], rem[:n_val], items[:n_test]  # train, val, test

    d_train, d_val, d_test = split_list(dual, args.test_dual_frac, args.val_frac)
    s_train, s_val, s_test = split_list(single, args.test_single_frac, args.val_frac)

    split_for = {}
    for u in d_train + s_train:
        split_for[id(u)] = "train"
    for u in d_val + s_val:
        split_for[id(u)] = "val"
    for u in d_test + s_test:
        split_for[id(u)] = "test"

    counts = defaultdict(lambda: {"framesets": 0, "dual": 0, "images": 0, "annotations": 0})
    for u in complete:
        sp = split_for[id(u)]
        counts[sp]["framesets"] += 1
        counts[sp]["dual"] += int(is_dual(u))
        counts[sp]["images"] += len(u["cams"])
        counts[sp]["annotations"] += sum(len(e["annotations"]) for e in u["cams"].values())
    for sp in ("train", "val", "test"):
        c = counts[sp]
        print(f"  {sp}: {c['framesets']} framesets ({c['dual']} dual) / "
              f"{c['images']} imgs / {c['annotations']} anns")

    if args.dry_run:
        print("[dry-run] skipping writes")
        return

    print(f"[6/7] Copying images to {args.out_root} …")
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "annotations").mkdir(exist_ok=True)
    (args.out_root / "calib_params").mkdir(exist_ok=True)

    used_timestamps = defaultdict(set)
    total_to_copy = sum(len(u["cams"]) for u in complete)
    copied = 0
    for u in complete:
        sp = split_for[id(u)]
        ts = u["canonical_timestamp"]
        n = u["canonical_frame_num"]
        for cam, e in u["cams"].items():
            dst = args.out_root / sp / ts / cam / f"Frame_{n}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                dst.unlink()
            elif dst.exists():
                dst.unlink()
            shutil.copy2(e["abs_path"].resolve(), dst)
            used_timestamps[sp].add(ts)
            copied += 1
            if copied % 2000 == 0:
                print(f"  copied {copied}/{total_to_copy}")
    print(f"  copied {copied}/{total_to_copy}")

    # Calibrations: each timestamp's calib comes from its canonical source root.
    # (src_root_by_source was built from the source list during loading.)
    ts_source = {}
    for u in complete:
        ts_source[u["canonical_timestamp"]] = u["canonical_source"]
    all_used_ts = set()
    for s in used_timestamps.values():
        all_used_ts.update(s)
    for ts in all_used_ts:
        src_cal = src_root_by_source[ts_source[ts]] / "calib_params" / ts
        dst_cal = args.out_root / "calib_params" / ts
        if dst_cal.is_symlink():
            dst_cal.unlink()
        elif dst_cal.exists():
            shutil.rmtree(dst_cal)
        if not src_cal.exists():
            print(f"  WARN: missing calibration {src_cal}")
            continue
        shutil.copytree(src_cal, dst_cal)

    print("[7/7] Writing COCO JSONs …")
    for sp in ("train", "val", "test"):
        us = [u for u in complete if split_for[id(u)] == sp]
        us.sort(key=lambda u: (u["canonical_timestamp"], u["canonical_frame_num"]))

        images_out = []
        annotations_out = []
        framesets_out = {}
        next_img_id = 0
        next_ann_id = 0

        for u in us:
            ts = u["canonical_timestamp"]
            n = u["canonical_frame_num"]
            cam_ids_ordered = []
            for cam in expected_cams_sorted:
                e = u["cams"][cam]
                img_id = next_img_id
                next_img_id += 1
                file_name = f"{ts}/{cam}/Frame_{n}.jpg"
                images_out.append({
                    "coco_url": "",
                    "date_captured": "",
                    "file_name": file_name,
                    "flickr_url": "",
                    "height": e["height"],
                    "id": img_id,
                    "width": e["width"],
                })
                cam_ids_ordered.append(img_id)
                for a in e["annotations"]:
                    new_ann = dict(a)
                    new_ann["id"] = next_ann_id
                    next_ann_id += 1
                    new_ann["image_id"] = img_id
                    annotations_out.append(new_ann)
            framesets_out[f"{ts}/Frame_{n}"] = {
                "datasetName": ts,
                "frames": cam_ids_ordered,
            }

        calibrations_out = {}
        for ts in sorted(used_timestamps[sp]):
            cams_map = {}
            cal_dir = args.out_root / "calib_params" / ts
            for cam in expected_cams_sorted:
                yaml_file = cal_dir / f"{cam}.yaml"
                if yaml_file.exists():
                    cams_map[cam] = f"calib_params/{ts}/{cam}.yaml"
            calibrations_out[ts] = cams_map

        blob = {
            "keypoint_names": keypoint_names,
            "skeleton": skeleton,
            "categories": categories,
            "calibrations": calibrations_out,
            "images": images_out,
            "annotations": annotations_out,
            "framesets": framesets_out,
        }
        out_path = args.out_root / "annotations" / f"instances_{sp}.json"
        with out_path.open("w") as f:
            json.dump(blob, f)
        print(f"  {out_path}  ({len(images_out)} imgs, {len(annotations_out)} anns)")

    report = {
        "single_roots": [str(r) for r in args.single_roots],
        "ma_roots": [str(r) for r in args.ma_roots],
        "out_root": str(args.out_root),
        "seed": args.seed,
        "source_framesets": fs_count_by_source,
        "unified_framesets_total": len(unified_framesets),
        "unified_framesets_complete": len(complete),
        "unified_framesets_incomplete": len(incomplete),
        "group_size_dist": {str(k): v for k, v in size_dist.items()},
        "splits": {sp: dict(counts[sp]) for sp in ("train", "val", "test")},
        "test_dual_frac": args.test_dual_frac,
        "test_single_frac": args.test_single_frac,
        "val_frac": args.val_frac,
    }
    with (args.out_root / "dedup_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"  {args.out_root}/dedup_report.json")
    print("done.")


if __name__ == "__main__":
    main()
