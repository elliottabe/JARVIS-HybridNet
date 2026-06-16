#!/usr/bin/env python3
"""
Visualize the courtship annotation pairs in a multi-animal COCO export.

The source (e.g. merge_multianimal) stores each recording as a PAIR of session
folders: within a pair the two folders hold the SAME physical (cam, frame) jpgs
labeled for a DIFFERENT fly (the courtship male/female pair). This script groups
annotations by physical key (cam, frame) across the date prefixes and overlays
BOTH flies (distinct colors) + skeleton on each image so the labels can be QC'd
before collapsing/merging.

Output:
  <root>/viz_pairs/<split>/<canonical_date>/<cam>/Frame_<N>.png
  <root>/viz_pairs/<split>/montage/<canonical_date>_Frame_<N>.png   (with --montage)

Drawing mirrors jarvis.visualization.visualization_utils / create_multi_animal_
videos3D (BGR per-fly colors), kept self-contained so it needs only cv2+numpy.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = Path(
    "/data2/users/eabe/datasets/Johnson_lab/red_data/merge_multianimal"
)

# BGR, mirrors create_multi_animal_videos3D.DEFAULT_FLY_COLORS
FLY_COLORS = [
    (0, 0, 255),    # fly0: red
    (255, 180, 0),  # fly1: blue
    (0, 255, 0),    # fly2: green
    (255, 0, 255),  # fly3: magenta
]


def physical_key(file_name: str):
    """(cam, frame_filename) — ignores the session/date prefix."""
    parts = file_name.split("/")
    return (parts[1], parts[2])


def frame_num(frame_filename: str) -> int:
    return int(frame_filename.replace("Frame_", "").replace(".jpg", ""))


def build_edges(coco: dict):
    name2idx = {n: i for i, n in enumerate(coco["keypoint_names"])}
    edges = []
    for s in coco["skeleton"]:
        a = name2idx.get(s["keypointA"])
        b = name2idx.get(s["keypointB"])
        if a is not None and b is not None:
            edges.append((a, b))
    return edges


def in_bounds(x, y, w, h):
    return (not np.isnan(x)) and (not np.isnan(y)) and 0 < x < w - 1 and 0 < y < h - 1


def draw_fly(img, kpts_xy, vis, edges, color):
    h, w = img.shape[:2]
    for a, b in edges:
        if vis[a] > 0 and vis[b] > 0:
            xa, ya = kpts_xy[a]
            xb, yb = kpts_xy[b]
            if in_bounds(xa, ya, w, h) and in_bounds(xb, yb, w, h):
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 1)
    for (x, y), v in zip(kpts_xy, vis):
        if v > 0 and in_bounds(x, y, w, h):
            cv2.circle(img, (int(x), int(y)), 3, color, thickness=-1)


def render_panel(img_path: Path, flies, edges, cam, fnum):
    """flies: list of (fly_idx, keypoints_flat, bbox). Returns BGR image."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")
    for fly_idx, kp_flat, bbox in flies:
        color = FLY_COLORS[fly_idx % len(FLY_COLORS)]
        kp = np.array(kp_flat, dtype=float).reshape(-1, 3)
        draw_fly(img, kp[:, :2], kp[:, 2], edges, color)
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), color, 1)
        cv2.putText(img, f"fly{fly_idx}", (8, 18 + 16 * fly_idx),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(img, f"{cam} Frame_{fnum}", (8, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_montage(cam_panels: dict, scale=0.5):
    """cam_panels: {cam: BGR image} -> single grid image."""
    cams = sorted(cam_panels)
    panels = [cam_panels[c] for c in cams]
    h, w = panels[0].shape[:2]
    n = len(panels)
    cols = 2
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
    if scale != 1.0:
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grid


def process_split(root: Path, splits, edges, max_frames, montage, out_label):
    """Render one panel per physical (cam, frame), pooling annotations across
    all `splits`. The courtship pair is often split across instances_train.json
    and instances_val.json (fly A in one, fly B in the other), so pooling is
    needed to show both flies on the same frame."""
    # (cam, frame) -> list of (date, split, img, anns)
    phys = defaultdict(list)
    for split in splits:
        with (root / "annotations" / f"instances_{split}.json").open() as f:
            coco = json.load(f)
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        for im in coco["images"]:
            cam, frame = physical_key(im["file_name"])
            date = im["file_name"].split("/")[0]
            phys[(cam, frame)].append(
                (date, split, im, anns_by_img.get(im["id"], [])))

    out_root = root / "viz_pairs" / out_label
    n_done = n_dual = 0
    framesets = defaultdict(dict)  # (canonical_date, fnum) -> {cam: panel}

    for (cam, frame), entries in sorted(phys.items()):
        if max_frames is not None and n_done >= max_frames:
            break
        # fly index = rank of the date the annotation came from (each date in a
        # recording pair is a different fly)
        date_rank = {d: i for i, d in enumerate(
            sorted({e[0] for e in entries}))}
        canonical = min(entries, key=lambda e: (e[1] != "train", e[2]["id"]))
        canon_date, canon_split, canon_im = canonical[0], canonical[1], canonical[2]
        img_path = root / canon_split / canon_im["file_name"]

        flies = []
        for date, split, im, anns in entries:
            for a in anns:
                flies.append((date_rank[date], a["keypoints"], a.get("bbox")))
        if len({f[0] for f in flies}) >= 2:
            n_dual += 1

        fnum = frame_num(frame)
        panel = render_panel(img_path, flies, edges, cam, fnum)
        out_path = out_root / canon_date / cam / f"Frame_{fnum}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), panel)
        n_done += 1
        if montage:
            framesets[(canon_date, fnum)][cam] = panel

    if montage:
        mont_dir = out_root / "montage"
        mont_dir.mkdir(parents=True, exist_ok=True)
        for (canon_date, fnum), cam_panels in sorted(framesets.items()):
            grid = make_montage(cam_panels)
            cv2.imwrite(str(mont_dir / f"{canon_date}_Frame_{fnum}.png"), grid)

    print(f"[{out_label}] {n_done} physical-frame panels written "
          f"({n_dual} with 2 flies) -> {out_root}")
    return n_done, n_dual


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    p.add_argument("--combine-splits", action="store_true",
                   help="pool train+val by physical frame so a courtship pair "
                        "split across the two json files shows both flies")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap physical frames rendered (sampling)")
    p.add_argument("--montage", action="store_true",
                   help="also write 7-camera grid montages per frameset")
    return p.parse_args()


def main():
    args = parse_args()
    splits = ["train", "val"] if args.split == "all" else [args.split]
    # edges are identical across splits; build from the first available json
    with (args.root / "annotations" / f"instances_{splits[0]}.json").open() as f:
        edges = build_edges(json.load(f))
    if args.combine_splits:
        process_split(args.root, splits, edges, args.max_frames,
                      args.montage, out_label="combined")
    else:
        for split in splits:
            process_split(args.root, [split], edges, args.max_frames,
                          args.montage, out_label=split)


if __name__ == "__main__":
    main()
