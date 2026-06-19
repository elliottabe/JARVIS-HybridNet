#!/usr/bin/env python3
"""
Per-condition evaluation for JARVIS KeypointDetect.

Measures the two failure modes that matter for courtship robustness, split by
condition so "more robust" is quantified rather than asserted:

  * keypoint error (px), reported for SINGLE-fly vs DUAL-fly frames, with dual
    frames further binned by inter-fly distance (close contact = the hard case);
  * CROSS-FLY CONFUSION rate -- on dual frames (both flies' GT known), the
    fraction of a target fly's predicted keypoints that land closer to the
    OTHER fly's ground-truth keypoint than to the target's. This is the direct
    measure of "keypoints confused between flies".

KeypointDetect is run on GT-centered crops (decoupling keypoint quality from
CenterDetect's detection step), mirroring the train/infer crop + 4-channel mask
logic in jarvis/dataset/dataset2D.py:_get_item_keypoints and the peak extraction
in jarvis/prediction/jarvis2D.py:forward.

Identity-swap rate (the second failure mode) is measured separately from
prediction CSVs via 3d_tracking_dataset/utils/identity_relink (see --swap-csv),
since it requires tracked video, not single frames.

Run two checkpoints (e.g. the masked and mask-free arms) and point --out at the
same JSON to accumulate a side-by-side comparison.

Usage (real eval, jarvis env + a trained checkpoint):
  python tools/eval_per_condition.py --project unified_V2_masked --split test \
      --label masked --out reports/per_condition.json
  python tools/eval_per_condition.py \
      --dataset /data2/.../red_data_unified_V2 --weights-kp /path/kp.pth \
      --instance-mask true --split test --label masked --out reports/pc.json

Validate the metric math locally (no GPU/checkpoint needed):
  python tools/eval_per_condition.py --selftest
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# jarvis is not pip-installed; make the repo root importable (mirrors run_train.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Inter-fly distance bin edges (pixels between fly bbox centers). The first bin
# is the close-contact / occlusion regime where confusion is expected.
DEFAULT_DIST_EDGES = [0.0, 75.0, 150.0, 300.0, float("inf")]


# --------------------------------------------------------------------------- #
# Pure metric functions (no torch / model dependency -> unit-testable).
# Each "record" is a dict:
#   pred       : (J, 2) predicted keypoints in image px
#   gt         : (J, 3) target GT (x, y, visibility)
#   other_gt   : (J, 3) or None   -- the other fly's GT (dual frames only)
#   condition  : 'single' | 'dual'
#   interfly   : float | None     -- distance between the two flies (dual only)
# --------------------------------------------------------------------------- #
def _visible(gt):
    return gt[:, 2] > 0


def keypoint_errors(pred, gt):
    """Per-keypoint Euclidean error (px) for visible keypoints; others -> nan."""
    vis = _visible(gt)
    err = np.full(len(gt), np.nan, dtype=float)
    if vis.any():
        d = np.linalg.norm(pred[vis] - gt[vis, :2], axis=1)
        err[vis] = d
    return err


def confusion_flags(pred, gt, other_gt):
    """Per-keypoint bool: predicted target kp is closer to the OTHER fly's GT
    than to the target's GT. Only defined where both flies' kp is visible."""
    J = len(gt)
    flags = np.full(J, np.nan, dtype=float)
    if other_gt is None:
        return flags
    both = (gt[:, 2] > 0) & (other_gt[:, 2] > 0)
    if both.any():
        d_self = np.linalg.norm(pred[both] - gt[both, :2], axis=1)
        d_other = np.linalg.norm(pred[both] - other_gt[both, :2], axis=1)
        flags[both] = (d_other < d_self).astype(float)
    return flags


def bin_index(dist, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= dist < edges[i + 1]:
            return i
    return len(edges) - 2


def aggregate(records, dist_edges=DEFAULT_DIST_EDGES, keypoint_names=None):
    """Aggregate per-annotation records into the per-condition report."""
    by_cond = defaultdict(lambda: {"errs": [], "n": 0})
    per_kp_err = defaultdict(list)          # condition -> list of (J,) err arrays
    dual_conf = []                          # per-annotation confusion rate
    bin_err = defaultdict(list)             # dist-bin -> mean errs
    bin_conf = defaultdict(list)            # dist-bin -> confusion rates
    bin_labels = [f"{dist_edges[i]:g}-{dist_edges[i+1]:g}px"
                  for i in range(len(dist_edges) - 1)]

    for r in records:
        err = keypoint_errors(r["pred"], r["gt"])
        mean_err = np.nanmean(err) if np.isfinite(err).any() else np.nan
        by_cond[r["condition"]]["errs"].append(mean_err)
        by_cond[r["condition"]]["n"] += 1
        per_kp_err[r["condition"]].append(err)

        if r["condition"] == "dual":
            flags = confusion_flags(r["pred"], r["gt"], r.get("other_gt"))
            conf_rate = np.nanmean(flags) if np.isfinite(flags).any() else np.nan
            dual_conf.append(conf_rate)
            if r.get("interfly") is not None:
                b = bin_labels[bin_index(r["interfly"], dist_edges)]
                bin_err[b].append(mean_err)
                bin_conf[b].append(conf_rate)

    def _m(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(xs)) if xs else None

    report = {
        "n_single": by_cond["single"]["n"],
        "n_dual": by_cond["dual"]["n"],
        "mean_err_px": {
            "single": _m(by_cond["single"]["errs"]),
            "dual": _m(by_cond["dual"]["errs"]),
        },
        "cross_fly_confusion_rate_dual": _m(dual_conf),
        "by_interfly_distance": {
            b: {"mean_err_px": _m(bin_err.get(b, [])),
                "confusion_rate": _m(bin_conf.get(b, [])),
                "n": len(bin_err.get(b, []))}
            for b in bin_labels
        },
    }
    # per-keypoint mean error (dual condition, where confusion bites)
    if per_kp_err["dual"]:
        stack = np.vstack(per_kp_err["dual"])
        pk = np.nanmean(stack, axis=0)
        names = keypoint_names or [f"kp{i}" for i in range(stack.shape[1])]
        report["per_keypoint_err_px_dual"] = {
            names[i]: (float(pk[i]) if np.isfinite(pk[i]) else None)
            for i in range(len(pk))
        }
    return report


# --------------------------------------------------------------------------- #
# Model-prediction path (lazy torch/jarvis imports; needs GPU + checkpoint).
# --------------------------------------------------------------------------- #
def _build_cfg(args):
    from jarvis.config.project_manager import ProjectManager
    if args.project:
        pm = ProjectManager()
        if not pm.load(args.project):
            raise SystemExit(f"Could not load project {args.project}")
        cfg = pm.cfg
        if args.instance_mask in ("true", "false"):
            cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT = (args.instance_mask == "true")
        return cfg
    # dataset + weights mode
    from jarvis.config import cfg
    cfg.PARENT_DIR = ""
    cfg.DATASET.DATASET_2D = str(args.dataset)
    cfg.DATASET.DATASET_3D = str(args.dataset)
    if args.instance_mask == "auto":
        # infer from presence of a sam3_masks dir
        cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT = os.path.isdir(
            os.path.join(args.dataset, "sam3_masks"))
    else:
        cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT = (args.instance_mask == "true")
    return cfg


def _load_masks(root, split, file_name):
    """Mirror datasetBase._load_instance_masks."""
    stem, _ = os.path.splitext(file_name)
    path = os.path.join(root, "sam3_masks", split, stem + ".npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def predict_records(args, cfg):
    """Run KeypointDetect on GT-centered crops; return per-annotation records."""
    import cv2
    import torch
    from torchvision import transforms
    from jarvis.efficienttrack.efficienttrack import EfficientTrack

    root = str(args.dataset) if args.dataset else cfg.DATASET.DATASET_2D
    split = args.split
    coco = json.load(open(os.path.join(root, "annotations",
                                        f"instances_{split}.json")))
    kp_names = coco["keypoint_names"]
    J = len(kp_names)
    cfg.KEYPOINTDETECT.NUM_JOINTS = J
    instance_mask = bool(cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT)
    bbox_hw = int(cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE / 2)

    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    weights = args.weights_kp or "latest"
    kp = EfficientTrack("KeypointDetect", cfg, weights=weights)
    model = kp.model.cuda().eval()
    mean = torch.tensor(cfg.DATASET.MEAN, device="cuda").view(3, 1, 1)
    std = torch.tensor(cfg.DATASET.STD, device="cuda").view(3, 1, 1)

    records = []
    viz_by_img = {}
    for img_id, anns in anns_by_img.items():
        if not anns:
            continue
        im = imgs[img_id]
        img = cv2.imread(os.path.join(root, split, im["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        condition = "dual" if len(anns) >= 2 else "single"
        mask_bundle = _load_masks(root, split, im["file_name"]) if instance_mask else None

        # bbox centers for inter-fly distance
        def _center(a):
            x, y, w, h = a["bbox"]
            return np.array([x + w / 2.0, y + h / 2.0])
        centers = [_center(a) for a in anns]

        img_flies = []
        for ti, a in enumerate(anns):
            bx, by, bw, bh = a["bbox"]
            cy = min(max(bbox_hw, int(by + bh / 2)), H - bbox_hw)
            cx = min(max(bbox_hw, int(bx + bw / 2)), W - bbox_hw)
            crop = img[cy - bbox_hw:cy + bbox_hw, cx - bbox_hw:cx + bbox_hw, :].copy()

            mask_crop = None
            if instance_mask and mask_bundle is not None:
                masks = mask_bundle.get("masks")
                matched = mask_bundle.get("matched")
                extra = mask_bundle.get("extra_masks")
                distractor = np.zeros((H, W), dtype=bool)
                if masks is not None and masks.shape[0] > 0:
                    for j in range(masks.shape[0]):
                        if j == ti:
                            continue
                        if matched is None or matched[j]:
                            distractor |= masks[j]
                if extra is not None and extra.shape[0] > 0:
                    for j in range(extra.shape[0]):
                        distractor |= extra[j]
                if (masks is not None and masks.shape[0] > ti
                        and (matched is None or matched[ti])):
                    distractor &= ~masks[ti]
                dcrop = distractor[cy - bbox_hw:cy + bbox_hw, cx - bbox_hw:cx + bbox_hw]
                if dcrop.any():
                    crop[dcrop] = crop.mean(axis=(0, 1))
                if (masks is not None and masks.shape[0] > ti
                        and (matched is None or matched[ti])):
                    mask_crop = masks[ti, cy - bbox_hw:cy + bbox_hw,
                                      cx - bbox_hw:cx + bbox_hw].astype(np.float32)
                else:
                    mask_crop = np.zeros((bbox_hw * 2, bbox_hw * 2), dtype=np.float32)

            t = torch.from_numpy(crop).permute(2, 0, 1).float().cuda() / 255.0
            t = (t - mean) / std
            if instance_mask:
                mc = (mask_crop if mask_crop is not None
                      else np.zeros((bbox_hw * 2, bbox_hw * 2), dtype=np.float32))
                t = torch.cat([t, torch.from_numpy(mc)[None].cuda()], dim=0)
            with torch.no_grad():
                out = model(t[None])
            hm = out[1].view(out[1].shape[0], out[1].shape[1], -1)
            m = hm.argmax(2).view(hm.shape[0], hm.shape[1], 1)
            pts = torch.cat((m % out[1].shape[3], m // out[1].shape[3]), dim=2)
            pts = pts.squeeze(0).float().cpu().numpy() * 2.0  # heatmap->crop px
            pred = pts + np.array([cx - bbox_hw, cy - bbox_hw])  # -> image px

            gt = np.array(a["keypoints"], dtype=float).reshape(-1, 3)
            other_gt = None
            interfly = None
            if condition == "dual":
                oi = 1 - ti if len(anns) == 2 else (ti + 1) % len(anns)
                other_gt = np.array(anns[oi]["keypoints"], dtype=float).reshape(-1, 3)
                interfly = float(np.linalg.norm(centers[ti] - centers[oi]))
            records.append({"pred": pred, "gt": gt, "other_gt": other_gt,
                            "condition": condition, "interfly": interfly})
            img_flies.append((ti, pred, gt))
        viz_by_img[img_id] = {"file_name": im["file_name"], "flies": img_flies,
                              "interfly": interfly, "condition": condition}
    return records, kp_names, viz_by_img


_FLY_BGR = [(0, 0, 255), (255, 180, 0), (0, 255, 0), (255, 0, 255)]


def _build_edges_from_coco(coco):
    name2idx = {n: i for i, n in enumerate(coco["keypoint_names"])}
    edges = []
    for s in coco["skeleton"]:
        a = name2idx.get(s["keypointA"])
        b = name2idx.get(s["keypointB"])
        if a is not None and b is not None:
            edges.append((a, b))
    return edges


def render_overlays(viz_by_img, root, split, edges, out_dir, max_n, label):
    """Draw predicted skeleton (lines+filled dots) vs GT (hollow rings),
    color-coded per fly, on the closest-contact dual frames."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    dual = [v for v in viz_by_img.values()
            if v["condition"] == "dual" and v["interfly"] is not None]
    dual.sort(key=lambda v: v["interfly"])
    sel = dual[:max_n]
    for v in sel:
        img = cv2.imread(os.path.join(root, split, v["file_name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        pts = []
        for fly_idx, pred, gt in v["flies"]:
            color = _FLY_BGR[fly_idx % len(_FLY_BGR)]
            for a, b in edges:
                xa, ya = pred[a]; xb, yb = pred[b]
                if 0 < xa < w and 0 < ya < h and 0 < xb < w and 0 < yb < h:
                    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, 1)
            for x, y in pred:
                if 0 < x < w and 0 < y < h:
                    cv2.circle(img, (int(x), int(y)), 2, color, -1); pts.append((x, y))
            for x, y, vis in gt:
                if vis > 0 and 0 < x < w and 0 < y < h:
                    cv2.circle(img, (int(x), int(y)), 5, color, 1); pts.append((x, y))
        # crop+zoom around the flies (frames are 1936-wide with tiny subjects)
        if pts:
            pad = 60
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            x0 = max(0, int(min(xs)) - pad); x1 = min(w, int(max(xs)) + pad)
            y0 = max(0, int(min(ys)) - pad); y1 = min(h, int(max(ys)) + pad)
            img = img[y0:y1, x0:x1]
            if img.size and img.shape[1] < 900:
                sc = 900.0 / img.shape[1]
                img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"{label} {v['interfly']:.0f}px  pred=lines/dots GT=rings",
                    (6, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        stem = v["file_name"].replace("/", "__").replace(".jpg", "")
        cv2.imwrite(os.path.join(out_dir, f"{v['interfly']:06.1f}px__{stem}.png"), img)
    print(f"  wrote {len(sel)} overlay(s) to {out_dir} (closest-contact dual frames first)")


# --------------------------------------------------------------------------- #
def _selftest():
    """Validate metric math on synthetic two-fly records."""
    J = 5
    # target fly at x=100, other fly at x=300
    gt = np.column_stack([np.full(J, 100.0), np.arange(J) * 10.0, np.ones(J)])
    other = np.column_stack([np.full(J, 300.0), np.arange(J) * 10.0, np.ones(J)])

    # perfect prediction on target -> 0 error, 0 confusion
    perfect = aggregate([{"pred": gt[:, :2].copy(), "gt": gt, "other_gt": other,
                          "condition": "dual", "interfly": 200.0}])
    assert perfect["mean_err_px"]["dual"] < 1e-6, perfect
    assert perfect["cross_fly_confusion_rate_dual"] == 0.0, perfect

    # prediction snapped onto the OTHER fly -> confusion == 1.0
    confused = aggregate([{"pred": other[:, :2].copy(), "gt": gt, "other_gt": other,
                           "condition": "dual", "interfly": 50.0}])
    assert confused["cross_fly_confusion_rate_dual"] == 1.0, confused
    assert confused["by_interfly_distance"]["0-75px"]["confusion_rate"] == 1.0, confused

    # single-fly error reported, no confusion bucket
    single = aggregate([{"pred": gt[:, :2] + 3.0, "gt": gt, "other_gt": None,
                         "condition": "single", "interfly": None}])
    assert abs(single["mean_err_px"]["single"] - np.sqrt(18.0)) < 1e-6, single
    assert single["mean_err_px"]["dual"] is None
    print("selftest OK: error, confusion, and distance-binning behave correctly")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", help="JARVIS project name (uses 'latest' weights)")
    ap.add_argument("--dataset", help="dataset root (alternative to --project)")
    ap.add_argument("--weights-kp", help="KeypointDetect .pth (dataset mode)")
    ap.add_argument("--instance-mask", choices=["auto", "true", "false"],
                    default="auto")
    ap.add_argument("--split", default="test")
    ap.add_argument("--label", default="model", help="key for this run in --out")
    ap.add_argument("--out", default="per_condition_report.json")
    ap.add_argument("--viz-dir", default=None,
                    help="also save predicted-vs-GT overlays for closest-contact dual frames")
    ap.add_argument("--viz-max", type=int, default=24)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.project and not args.dataset:
        ap.error("provide --project or --dataset")

    cfg = _build_cfg(args)
    records, kp_names, viz_by_img = predict_records(args, cfg)
    report = aggregate(records, keypoint_names=kp_names)
    report["_meta"] = {
        "label": args.label, "split": args.split,
        "instance_mask": bool(cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT),
        "n_records": len(records),
        "project": args.project, "dataset": args.dataset,
    }

    all_reports = {}
    if os.path.isfile(args.out):
        all_reports = json.load(open(args.out))
    all_reports[args.label] = report
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(all_reports, open(args.out, "w"), indent=2)

    print(f"[{args.label}] split={args.split} "
          f"instance_mask={cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT}")
    print(f"  single n={report['n_single']} mean_err_px={report['mean_err_px']['single']}")
    print(f"  dual   n={report['n_dual']} mean_err_px={report['mean_err_px']['dual']}")
    print(f"  cross-fly confusion (dual): {report['cross_fly_confusion_rate_dual']}")
    for b, v in report["by_interfly_distance"].items():
        print(f"    {b}: err={v['mean_err_px']} conf={v['confusion_rate']} n={v['n']}")
    print(f"  wrote {args.out}")

    if args.viz_dir:
        root = str(args.dataset) if args.dataset else cfg.DATASET.DATASET_2D
        coco = json.load(open(os.path.join(root, "annotations",
                                            f"instances_{args.split}.json")))
        render_overlays(viz_by_img, root, args.split,
                        _build_edges_from_coco(coco), args.viz_dir, args.viz_max,
                        args.label)


if __name__ == "__main__":
    main()
