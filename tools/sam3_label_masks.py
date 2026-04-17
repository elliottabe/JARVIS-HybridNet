"""
Offline SAM3 per-annotation mask generation for red_data_unified.

For each labeled image, run SAM3 with text prompt "insect" to get candidate
insect masks, then greedy-assign each COCO annotation to the mask whose
pixels best cover that annotation's visible keypoints. Save a compressed
.npz per image so training can hard-mask distractor flies and feed the
target fly's mask as a 4th input channel.

Output: <data-root>/sam3_masks/<split>/<ts>/<cam>/Frame_<N>.npz
  masks        bool[N_ann, H, W]   target fly mask per annotation (zero if unmatched)
  bboxes       float32[N_ann, 4]   SAM3 xyxy (zero if unmatched)
  scores       float32[N_ann]      SAM3 confidence (zero if unmatched)
  ann_ids      int64[N_ann]        COCO annotation IDs (parallel to masks)
  kp_coverage  float32[N_ann]      fraction of visible keypoints inside matched mask
  matched      bool[N_ann]         True if assigned and passed QC
  extra_masks  bool[K, H, W]       SAM3 masks not assigned to any annotation
  extra_boxes  float32[K, 4]
  extra_scores float32[K]

Plus <out-root>/mask_report.json summarizing per-split stats and rejects.
"""

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def load_annotations(json_path):
    with open(json_path) as f:
        blob = json.load(f)
    anns_by_img = defaultdict(list)
    for a in blob["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    return blob["images"], anns_by_img


def ann_centroid(ann):
    kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
    vis = kp[kp[:, 2] > 0]
    if len(vis) == 0:
        x, y, w, h = ann["bbox"]
        return float(x + w / 2), float(y + h / 2), kp
    return float(vis[:, 0].mean()), float(vis[:, 1].mean()), kp


def keypoint_coverage(kp_arr, mask):
    vis = kp_arr[kp_arr[:, 2] > 0]
    if len(vis) == 0:
        return 1.0
    H, W = mask.shape
    xs = np.clip(np.round(vis[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(vis[:, 1]).astype(int), 0, H - 1)
    return float(mask[ys, xs].mean())


def assign_masks_to_anns(anns, masks, boxes, img_w, img_h):
    """Greedy assign each annotation to the best unused SAM3 mask.

    Scoring prefers mask-pixel hit at the keypoint centroid, with
    keypoint-coverage as tiebreaker; falls back to nearest box center.
    Returns list parallel to anns: {mask_idx, score, kp_cov, kp_arr, cx, cy}
    """
    N_masks = len(masks)
    ann_info = []
    for a in anns:
        cx, cy, kp = ann_centroid(a)
        ann_info.append((cx, cy, kp))

    # Score every (ann, mask) pair.
    scores = np.full((len(anns), max(N_masks, 1)), -1e9, dtype=np.float32)
    for i, (cx, cy, kp) in enumerate(ann_info):
        xi = int(np.clip(round(cx), 0, img_w - 1))
        yi = int(np.clip(round(cy), 0, img_h - 1))
        for j in range(N_masks):
            m = masks[j]
            hit = float(m[yi, xi])
            cov = keypoint_coverage(kp, m)
            # box-center proximity (normalized, negated so larger=better)
            x0, y0, x1, y1 = boxes[j]
            bcx, bcy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            d = np.hypot(bcx - cx, bcy - cy)
            prox = -d / max(img_w, img_h)
            # priority: centroid hit > kp coverage > proximity
            scores[i, j] = hit * 10.0 + cov + prox * 0.1

    assignments = []
    used = set()
    order = sorted(range(len(anns)),
                   key=lambda i: -scores[i].max() if N_masks > 0 else 0)
    for i in order:
        (cx, cy, kp) = ann_info[i]
        best_j = -1
        best_s = -1e9
        for j in range(N_masks):
            if j in used:
                continue
            if scores[i, j] > best_s:
                best_s = scores[i, j]
                best_j = j
        assignments.append((i, best_j, cx, cy, kp))
        if best_j >= 0:
            used.add(best_j)

    # Restore original annotation order.
    assignments.sort(key=lambda t: t[0])
    out = []
    for i, j, cx, cy, kp in assignments:
        out.append({"mask_idx": j, "cx": cx, "cy": cy, "kp": kp})
    return out, used


def process_image(im, anns, img_path, proc, text_prompt, qc_thresh):
    """Return a dict of arrays for this image + (reject_records)."""
    H, W = im["height"], im["width"]
    pil = Image.open(img_path).convert("RGB")
    state = proc.set_image(pil)
    state = proc.set_text_prompt(text_prompt, state)

    masks_t = state["masks"]  # (N, 1, H, W) bool
    if masks_t.shape[0] > 0:
        masks = masks_t.squeeze(1).cpu().numpy().astype(bool)
        boxes = state["boxes"].float().cpu().numpy().astype(np.float32)
        scores = state["scores"].float().cpu().numpy().astype(np.float32)
    else:
        masks = np.zeros((0, H, W), dtype=bool)
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)

    n_ann = len(anns)
    per_ann_mask = np.zeros((n_ann, H, W), dtype=bool)
    per_ann_bbox = np.zeros((n_ann, 4), dtype=np.float32)
    per_ann_score = np.zeros(n_ann, dtype=np.float32)
    per_ann_id = np.array([a["id"] for a in anns], dtype=np.int64)
    per_ann_cov = np.zeros(n_ann, dtype=np.float32)
    per_ann_matched = np.zeros(n_ann, dtype=bool)

    rejects = []
    if n_ann > 0 and len(masks) > 0:
        assigns, used = assign_masks_to_anns(anns, masks, boxes, W, H)
        for i, info in enumerate(assigns):
            j = info["mask_idx"]
            if j < 0:
                rejects.append({"file": im["file_name"],
                                "ann_id": int(per_ann_id[i]),
                                "reason": "no_mask_available"})
                continue
            m = masks[j]
            cov = keypoint_coverage(info["kp"], m)
            per_ann_cov[i] = cov
            # Always save the matched mask; coverage is a diagnostic that the
            # training loader can threshold on. Only flag "unmatched" if
            # coverage is low enough to suggest a wrong-instance match.
            per_ann_mask[i] = m
            per_ann_bbox[i] = boxes[j]
            per_ann_score[i] = scores[j]
            if cov < qc_thresh:
                rejects.append({"file": im["file_name"],
                                "ann_id": int(per_ann_id[i]),
                                "reason": "low_kp_coverage",
                                "coverage": cov})
            else:
                per_ann_matched[i] = True
    else:
        used = set()
        if n_ann > 0:
            for aid in per_ann_id:
                rejects.append({"file": im["file_name"],
                                "ann_id": int(aid),
                                "reason": "no_sam3_detections"})

    extra_idx = [j for j in range(len(masks)) if j not in used]
    if extra_idx:
        extra_masks = masks[extra_idx]
        extra_boxes = boxes[extra_idx]
        extra_scores = scores[extra_idx]
    else:
        extra_masks = np.zeros((0, H, W), dtype=bool)
        extra_boxes = np.zeros((0, 4), dtype=np.float32)
        extra_scores = np.zeros(0, dtype=np.float32)

    return {
        "masks": per_ann_mask,
        "bboxes": per_ann_bbox,
        "scores": per_ann_score,
        "ann_ids": per_ann_id,
        "kp_coverage": per_ann_cov,
        "matched": per_ann_matched,
        "extra_masks": extra_masks,
        "extra_boxes": extra_boxes,
        "extra_scores": extra_scores,
    }, rejects


def process_split(split, data_root, out_root, proc, text_prompt,
                  qc_thresh, overwrite, limit):
    json_path = data_root / "annotations" / f"instances_{split}.json"
    if not json_path.exists():
        print(f"[skip] {json_path} does not exist")
        return {}, []

    images, anns_by_img = load_annotations(json_path)
    if limit:
        images = images[:limit]

    stats = {"images": 0, "skipped_existing": 0, "errors": 0,
             "anns_total": 0, "anns_matched": 0, "anns_rejected": 0,
             "extra_masks_total": 0}
    rejects = []

    for im in tqdm(images, desc=split):
        file_name = im["file_name"]
        ts, cam, frame_file = file_name.split("/")
        out_path = out_root / split / ts / cam / frame_file.replace(".jpg", ".npz")
        if out_path.exists() and not overwrite:
            stats["skipped_existing"] += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img_path = data_root / split / file_name
        anns = anns_by_img.get(im["id"], [])

        try:
            arrays, im_rejects = process_image(
                im, anns, img_path, proc, text_prompt, qc_thresh,
            )
        except Exception as e:
            stats["errors"] += 1
            rejects.append({"file": file_name, "reason": f"error: {e}",
                            "trace": traceback.format_exc()})
            continue

        np.savez_compressed(out_path, **arrays)
        stats["images"] += 1
        stats["anns_total"] += len(anns)
        stats["anns_matched"] += int(arrays["matched"].sum())
        stats["anns_rejected"] += int(len(anns) - arrays["matched"].sum())
        stats["extra_masks_total"] += int(arrays["extra_masks"].shape[0])
        rejects.extend(im_rejects)

    return stats, rejects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path("/data2/users/eabe/datasets/Johnson_lab/red_data/red_data_unified"))
    ap.add_argument("--out-root", type=Path, default=None,
                    help="defaults to <data-root>/sam3_masks")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--text-prompt", default="insect")
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--qc-thresh", type=float, default=0.60,
                    help="min fraction of visible keypoints inside matched mask "
                         "(mask is always saved; flag controls 'matched' field)")
    ap.add_argument("--resolution", type=int, default=1008)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only first N images per split (for debugging)")
    args = ap.parse_args()

    if args.out_root is None:
        args.out_root = args.data_root / "sam3_masks"
    args.out_root.mkdir(parents=True, exist_ok=True)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("building SAM3 …")
    model = build_sam3_image_model(device="cuda", load_from_HF=True)
    proc = Sam3Processor(model, resolution=args.resolution, device="cuda",
                         confidence_threshold=args.confidence)

    report = {"args": {
        "data_root": str(args.data_root),
        "out_root": str(args.out_root),
        "text_prompt": args.text_prompt,
        "confidence": args.confidence,
        "qc_thresh": args.qc_thresh,
        "resolution": args.resolution,
    }, "stats": {}, "rejects": {}}

    for split in args.splits:
        stats, rejects = process_split(
            split, args.data_root, args.out_root, proc,
            args.text_prompt, args.qc_thresh, args.overwrite, args.limit,
        )
        report["stats"][split] = stats
        report["rejects"][split] = rejects
        print(f"{split}: {stats}  ({len(rejects)} reject records)")

    report_path = args.out_root / "mask_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
