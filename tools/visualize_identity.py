"""Identity-disambiguation check for the 4-ch KeypointDetect network.

For each picked dual-fly image, runs KeypointDetect twice — once with
the mask + distractor-gray steered at annotation 0, once steered at
annotation 1 — and renders the two crops side-by-side with GT skeleton
and predicted skeleton per target. If the mask channel is doing its job,
the two predictions should lie on two different flies, matching their
respective GTs with low per-fly error.
"""

import os
import sys
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.utils.skeleton import get_skeleton


KP_WEIGHTS = os.path.join(
    ROOT, 'projects/red_data_unified/models/KeypointDetect/'
    'phase4_kp_ft/EfficientTrack-medium_final.pth')
OUT_PATH = os.path.join(ROOT, 'tools/figures/phase4_identity.png')
N_IMAGES = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def denormalize_rgb(hwc_norm, cfg):
    rgb = hwc_norm[..., :3] * np.array(cfg.DATASET.STD) + \
        np.array(cfg.DATASET.MEAN)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def run_kp(model, hwc_norm):
    x = torch.from_numpy(np.transpose(hwc_norm, (2, 0, 1))) \
        .float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    hm = out[1][0].detach().cpu().numpy()
    K, h, w = hm.shape
    sx = hwc_norm.shape[1] / w
    sy = hwc_norm.shape[0] / h
    flat = hm.reshape(K, -1)
    m = flat.argmax(axis=1)
    xs = (m % w) * sx + sx / 2
    ys = (m // w) * sy + sy / 2
    return np.stack([xs, ys], axis=1)


def draw_skeleton(ax, pts_xy, colors, line_idxs, marker, lw, size,
                  valid, alpha=1.0, edge='white'):
    for a, b in line_idxs:
        if valid[a] and valid[b]:
            c = np.array(colors[b]) / 255.0
            ax.plot([pts_xy[a, 0], pts_xy[b, 0]],
                    [pts_xy[a, 1], pts_xy[b, 1]],
                    color=c, lw=lw, alpha=alpha)
    for i, (x, y) in enumerate(pts_xy):
        if valid[i]:
            c = np.array(colors[i]) / 255.0
            ax.plot(x, y, marker=marker, color=c, markersize=size**0.5,
                    markeredgecolor=edge, markeredgewidth=0.3, alpha=alpha)


def pick_dual_images(ds, n):
    by_img = {}
    for i, (img_id, ann_idx) in enumerate(ds.ann_index):
        by_img.setdefault(img_id, []).append((i, ann_idx))
    # Require exactly 2 annotations, prefer those spread across files
    duals = [(img_id, idxs) for img_id, idxs in by_img.items()
             if len(idxs) >= 2]
    # Stride through to get variety
    picked = []
    step = max(1, len(duals) // n)
    for i in range(0, len(duals), step):
        picked.append(duals[i])
        if len(picked) >= n:
            break
    return picked


def main():
    pm = ProjectManager()
    assert pm.load('red_data_unified'), 'project load failed'
    cfg = pm.get_cfg()
    colors, line_idxs = get_skeleton(cfg)

    kp_ds = Dataset2D(cfg=cfg, set='val', mode='KeypointDetect')
    assert kp_ds.instance_mask_input, 'INSTANCE_MASK_INPUT not active'

    print(f'[cfg] loading KP weights: {KP_WEIGHTS}')
    kp = EfficientTrack('KeypointDetectInference', cfg, weights=KP_WEIGHTS)
    kp.model.to(DEVICE).eval()

    dual_pairs = pick_dual_images(kp_ds, N_IMAGES)
    print(f'[sel] picked {len(dual_pairs)} dual-fly images')

    fig, axes = plt.subplots(len(dual_pairs), 2,
                             figsize=(9, 4.2 * len(dual_pairs)))
    if len(dual_pairs) == 1:
        axes = axes[None, :]

    for row, (img_id, idx_pairs) in enumerate(dual_pairs):
        if len(idx_pairs) > 2:
            idx_pairs = idx_pairs[:2]
        for col, (ds_idx, ann_idx) in enumerate(idx_pairs):
            sample = kp_ds[ds_idx]
            img_norm, _, kp_gt = sample
            rgb = denormalize_rgb(img_norm, cfg)
            mask = img_norm[..., 3]
            pred_xy = run_kp(kp.model, img_norm)
            gt_xy = kp_gt.reshape(-1, 3)
            valid = (gt_xy[:, 0] != 0) | (gt_xy[:, 1] != 0)

            err = np.full(len(gt_xy), np.nan)
            err[valid] = np.linalg.norm(
                pred_xy[valid] - gt_xy[valid, :2], axis=1)
            mean_err = float(np.nanmean(err)) if valid.any() else \
                float('nan')

            ax = axes[row, col]
            ax.imshow(rgb)
            mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            mask_rgba[..., 2] = 1.0
            mask_rgba[..., 3] = np.clip(mask, 0, 1) * 0.28
            ax.imshow(mask_rgba)
            draw_skeleton(ax, gt_xy[:, :2], colors, line_idxs,
                          marker='o', lw=1.2, size=30, valid=valid,
                          alpha=0.9, edge='black')
            draw_skeleton(ax, pred_xy, colors, line_idxs,
                          marker='x', lw=1.0, size=24,
                          valid=np.ones(len(pred_xy), bool),
                          alpha=0.9, edge='white')
            ax.set_title(f'img_id={img_id} · target ann={ann_idx} · '
                         f'mean px err={mean_err:.2f} (n={int(valid.sum())})')
            ax.set_axis_off()

    legend_handles = [
        mpatches.Patch(color='blue', alpha=0.3, label='SAM3 target mask (input ch3)'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='black',
                   markersize=6, label='GT keypoints'),
        plt.Line2D([0], [0], marker='x', color='gray',
                   markersize=7, lw=0, label='Predicted keypoints'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140)
    plt.close(fig)
    print(f'[out] saved figure to {OUT_PATH}')


if __name__ == '__main__':
    main()
