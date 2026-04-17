"""Render GT vs prediction visualizations for Phase-4 fine-tuned
CenterDetect and KeypointDetect on red_data_unified val samples.

Emits a PNG grid: for N picked val samples, each row shows:
  (col 0) CenterDetect full image with GT centers (red) and predicted
          peaks (green), plus a max-projected pred heatmap in a corner.
  (col 1) KeypointDetect crop (RGB channels), GT skeleton (cyan) and
          predicted skeleton (lime), with the mask channel shown as a
          blue alpha overlay to confirm the 4-ch input.
"""

import os
import sys
import numpy as np
import torch
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.utils.skeleton import get_skeleton


CENTER_WEIGHTS = os.path.join(
    ROOT, 'projects/red_data_unified/models/CenterDetect/'
    'phase4_center_ft/EfficientTrack-medium_final.pth')
KP_WEIGHTS = os.path.join(
    ROOT, 'projects/red_data_unified/models/KeypointDetect/'
    'phase4_kp_ft/EfficientTrack-medium_final.pth')
OUT_PATH = os.path.join(ROOT, 'tools/figures/phase4_val_preds.png')

N_SAMPLES = 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def denormalize_rgb(chw_or_hwc, cfg, had_mask=False):
    """Return uint8 HWC image given normalized data. Accepts HWC."""
    arr = np.array(chw_or_hwc, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    rgb = arr[..., :3] * np.array(cfg.DATASET.STD) + np.array(cfg.DATASET.MEAN)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def decode_peaks(heatmap_hw, k=2, min_ratio=0.4):
    """Return up to k (x, y, score) peaks via simple NMS."""
    h, w = heatmap_hw.shape
    flat = heatmap_hw.copy()
    peaks = []
    if flat.max() <= 0:
        return peaks
    top = flat.max()
    for _ in range(k):
        m = flat.argmax()
        score = float(flat.flat[m])
        if score < top * min_ratio:
            break
        y, x = divmod(int(m), w)
        peaks.append((x, y, score))
        r = max(h, w) // 10
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        flat[y0:y1, x0:x1] = -1
    return peaks


def pick_samples(kp_ds, n):
    """Prefer dual-fly images (two annotations on same image_id)."""
    if kp_ds.ann_index is None:
        return list(np.linspace(0, len(kp_ds) - 1, n, dtype=int))
    ann_index = kp_ds.ann_index
    seen = {}
    for i, (img_id, _) in enumerate(ann_index):
        seen.setdefault(img_id, []).append(i)
    duals = [idxs for idxs in seen.values() if len(idxs) >= 2]
    singles = [idxs for idxs in seen.values() if len(idxs) == 1]
    picked = []
    for idxs in duals:
        picked.extend(idxs[:2])
        if len(picked) >= n:
            break
    for idxs in singles:
        if len(picked) >= n:
            break
        picked.append(idxs[0])
    return picked[:n]


def run_center_detect(model, img_hwc_norm):
    """Return pred_peaks (list of (x,y,score) in image-pixel coords) and
    a low-res pred heatmap for plotting."""
    x = torch.from_numpy(np.transpose(img_hwc_norm, (2, 0, 1))) \
        .float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    hm = out[1][0, 0].detach().cpu().numpy()
    img_h, img_w = img_hwc_norm.shape[:2]
    scale_x = img_w / hm.shape[1]
    scale_y = img_h / hm.shape[0]
    peaks = decode_peaks(hm, k=3, min_ratio=0.4)
    peaks_img = [(p[0] * scale_x + scale_x / 2,
                  p[1] * scale_y + scale_y / 2, p[2]) for p in peaks]
    return peaks_img, hm


def run_kp_detect(model, img_hwc_norm):
    """Return (K, 3) array of predicted (x, y, score) in crop coords."""
    x = torch.from_numpy(np.transpose(img_hwc_norm, (2, 0, 1))) \
        .float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
    hm = out[1][0].detach().cpu().numpy()  # (K, h, w)
    K, h, w = hm.shape
    crop_h, crop_w = img_hwc_norm.shape[:2]
    scale_x = crop_w / w
    scale_y = crop_h / h
    flat = hm.reshape(K, -1)
    m = flat.argmax(axis=1)
    score = flat.max(axis=1)
    xs = (m % w) * scale_x + scale_x / 2
    ys = (m // w) * scale_y + scale_y / 2
    return np.stack([xs, ys, score], axis=1)


def draw_skeleton(ax, pts_xy, colors, line_idxs, marker='o',
                  alpha=1.0, lw=1.5, size=18, valid_mask=None):
    if valid_mask is None:
        valid_mask = np.ones(len(pts_xy), dtype=bool)
    for a, b in line_idxs:
        if valid_mask[a] and valid_mask[b]:
            c = np.array(colors[b]) / 255.0
            ax.plot([pts_xy[a, 0], pts_xy[b, 0]],
                    [pts_xy[a, 1], pts_xy[b, 1]],
                    color=c, lw=lw, alpha=alpha)
    for i, (x, y) in enumerate(pts_xy):
        if valid_mask[i]:
            c = np.array(colors[i]) / 255.0
            ax.plot(x, y, marker=marker, color=c, markersize=size**0.5,
                    markeredgecolor='white', markeredgewidth=0.3, alpha=alpha)


def main():
    pm = ProjectManager()
    assert pm.load('red_data_unified'), 'could not load project'
    cfg = pm.get_cfg()
    colors, line_idxs = get_skeleton(cfg)

    print('[cfg] building val datasets...')
    center_ds = Dataset2D(cfg=cfg, set='val', mode='CenterDetect')
    kp_ds = Dataset2D(cfg=cfg, set='val', mode='KeypointDetect')

    print(f'[cfg] loading CenterDetect weights: {CENTER_WEIGHTS}')
    cd = EfficientTrack('CenterDetectInference', cfg,
                        weights=CENTER_WEIGHTS)
    cd.model.to(DEVICE).eval()

    print(f'[cfg] loading KeypointDetect weights: {KP_WEIGHTS}')
    kp = EfficientTrack('KeypointDetectInference', cfg,
                        weights=KP_WEIGHTS)
    kp.model.to(DEVICE).eval()

    kp_idxs = pick_samples(kp_ds, N_SAMPLES)
    print(f'[sel] picked KP indices: {kp_idxs}')

    # Map KP ann indices back to a matching CenterDetect image index.
    # Both datasets iterate images in the same order via image_ids; for
    # CenterDetect we index by the position of the image_id in center_ds.
    ann_index = kp_ds.ann_index
    center_img_id_to_idx = {}
    for i, img_id in enumerate(center_ds.image_ids):
        center_img_id_to_idx[img_id] = i

    fig, axes = plt.subplots(N_SAMPLES, 2,
                             figsize=(11, 4 * N_SAMPLES))
    if N_SAMPLES == 1:
        axes = axes[None, :]

    for row, kp_idx in enumerate(kp_idxs):
        if ann_index is not None:
            image_id, target_ann_idx = ann_index[kp_idx]
        else:
            image_id = kp_ds.image_ids[kp_idx]
            target_ann_idx = 0
        cd_idx = center_img_id_to_idx.get(image_id, 0)

        # --- CenterDetect pass ---
        cd_sample = center_ds[cd_idx]
        cd_img_norm, _, cd_gt_centers = cd_sample
        cd_rgb = denormalize_rgb(cd_img_norm, cfg)
        peaks_img, cd_hm = run_center_detect(cd.model, cd_img_norm)

        ax = axes[row, 0]
        ax.imshow(cd_rgb)
        for gx, gy, gv in cd_gt_centers:
            if gv != 0:
                ax.scatter([gx], [gy], marker='x', s=140, c='red',
                           linewidths=2.5, label='GT' if row == 0 else None)
        for px, py, ps in peaks_img:
            ax.scatter([px], [py], marker='+', s=180, c='lime',
                       linewidths=2.5,
                       label='pred' if row == 0 else None)
        hm_u8 = (cd_hm / max(cd_hm.max(), 1e-6) * 255).astype(np.uint8)
        hm_u8 = cv2.resize(hm_u8, (80, 80), interpolation=cv2.INTER_LINEAR)
        ax.imshow(hm_u8, cmap='magma', extent=(
            cd_rgb.shape[1] - 82, cd_rgb.shape[1] - 2, 82, 2),
                  alpha=0.8)
        ax.set_title(f'Center · img_id={image_id} ann={target_ann_idx}')
        ax.set_axis_off()
        if row == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.7)

        # --- KeypointDetect pass ---
        kp_sample = kp_ds[kp_idx]
        kp_img_norm, _, kp_gt = kp_sample
        kp_rgb = denormalize_rgb(kp_img_norm, cfg)
        has_mask = kp_img_norm.shape[-1] == 4
        kp_mask = kp_img_norm[..., 3] if has_mask else None

        pred_xy_score = run_kp_detect(kp.model, kp_img_norm)
        gt_xy = kp_gt.reshape(-1, 3)
        gt_valid = (gt_xy[:, 0] != 0) | (gt_xy[:, 1] != 0)

        ax = axes[row, 1]
        ax.imshow(kp_rgb)
        if kp_mask is not None:
            mask_rgba = np.zeros((*kp_mask.shape, 4), dtype=np.float32)
            mask_rgba[..., 2] = 1.0
            mask_rgba[..., 3] = np.clip(kp_mask, 0, 1) * 0.25
            ax.imshow(mask_rgba)

        draw_skeleton(ax, gt_xy[:, :2], colors, line_idxs,
                      marker='o', alpha=0.9, lw=1.2, size=30,
                      valid_mask=gt_valid)
        draw_skeleton(ax, pred_xy_score[:, :2], colors, line_idxs,
                      marker='x', alpha=0.9, lw=1.0, size=22,
                      valid_mask=np.ones(len(pred_xy_score), dtype=bool))

        # Euclid err vs GT for valid points only
        err = np.full(len(gt_xy), np.nan)
        err[gt_valid] = np.linalg.norm(
            pred_xy_score[gt_valid, :2] - gt_xy[gt_valid, :2], axis=1)
        mean_err = float(np.nanmean(err)) if gt_valid.any() else float('nan')
        ax.set_title(f'KP 4-ch · mean px err={mean_err:.2f} '
                     f'(n={int(gt_valid.sum())})')
        ax.set_axis_off()

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140)
    plt.close(fig)
    print(f'[out] saved figure to {OUT_PATH}')


if __name__ == '__main__':
    main()
