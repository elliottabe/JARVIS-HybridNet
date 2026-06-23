"""Isolation diagnostic: is the V3 regression in the 2D KeypointDetect net or
in HybridNet's 3D fusion?

Replays bout 1 (cached SAM3 masks, --reuse-masks) and, inside run_kp_and_3d,
records BOTH:
  * the 2D KeypointDetect heatmap peak + spread (kp_model(imgs_4ch)[1]),
    i.e. the per-camera 2D detections BEFORE HybridNet, and
  * the 3D heatmap confidence HybridNet emits (what lands in the CSV).

Interpretation:
  - 2D peaks diffuse/low  -> KeypointDetect is the culprit.
  - 2D peaks sharp/high but 3D conf low -> HybridNet fusion is the culprit.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tools.predict3D_multianimal as P
import cv2

REC = ('2D_peak_raw', '2D_peak_norm', '2D_spread_px', '3D_conf')
ACC = {k: [] for k in REC}

# Dump raw 2D-argmax overlays for the first few processed flies so we can see
# whether the 2D detections themselves are correctly placed.
OVERLAY_DIR = os.environ.get('DIAG_OVERLAY_DIR')
_call = {'n': 0}
_REPRO = {'tool': None}

# Capture repro_tool (created in main, passed to predict_bout) into a global.
_orig_predict_bout = P.predict_bout
def _wrapped_predict_bout(bout, bout_masks, video_paths, centerDetect,
                          kp_model, hybridNet, repro_tool, *a, **k):
    _REPRO['tool'] = repro_tool
    return _orig_predict_bout(bout, bout_masks, video_paths, centerDetect,
                              kp_model, hybridNet, repro_tool, *a, **k)
P.predict_bout = _wrapped_predict_bout
_pm = P.ProjectManager(); _pm.load('unified_V3_masked'); _CFG = _pm.get_cfg()
_MEAN = np.array(_CFG.DATASET.MEAN, dtype=np.float32)
_STD = np.array(_CFG.DATASET.STD, dtype=np.float32)


def _dump_2d_overlay(imgs_4ch, heatmaps, tag, points3D=None, centerHMs=None):
    """Overlay 2D-argmax KeypointDetect (red) and 3D-reprojected HybridNet
    output (blue) on the same crop. Divergence = HybridNet positional error."""
    C, K, h, w = heatmaps.shape
    H = W = imgs_4ch.shape[2]
    bbox_hw = H // 2
    flat = heatmaps.view(C, K, -1)
    idx = flat.argmax(dim=2).cpu().numpy()        # (C,K)
    ys, xs = idx // w, idx % w
    # reproject HybridNet 3D points -> full-image px per camera -> (K, C, 2)
    repro = None
    if points3D is not None and _REPRO['tool'] is not None:
        with torch.no_grad():
            repro = _REPRO['tool'].reprojectPoint(points3D).cpu().numpy()
    chm = centerHMs.cpu().numpy() if centerHMs is not None else None
    for c in range(min(C, 4)):
        rgb = imgs_4ch[c, :3].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * _STD + _MEAN) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)[..., ::-1].copy()
        msk = imgs_4ch[c, 3].cpu().numpy() > 0.5
        rgb[msk] = (0.6 * rgb[msk] + np.array([0, 60, 0])).clip(0, 255).astype(np.uint8)
        for k in range(K):
            px = int(xs[c, k] * W / w); py = int(ys[c, k] * H / h)
            cv2.circle(rgb, (px, py), 3, (0, 0, 255), -1)   # red = 2D argmax
            if repro is not None and chm is not None:
                rx = int(repro[k, c, 0] - (chm[c, 0] - bbox_hw))
                ry = int(repro[k, c, 1] - (chm[c, 1] - bbox_hw))
                cv2.circle(rgb, (rx, ry), 3, (255, 0, 0), -1)  # blue = 3D reproj
        big = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OVERLAY_DIR, f'{tag}_cam{c}.png'), big)


def instrumented_run_kp_and_3d(kp_model, hybridNet, imgs_4ch, center3D,
                               centerHMs, cameraMatrices):
    with torch.no_grad():
        heatmaps = kp_model(imgs_4ch)[1]          # (num_cams, K, h, w)
        # --- 2D KeypointDetect stats (per cam, per keypoint) ---
        C, K, h, w = heatmaps.shape
        flat2d = heatmaps.view(C, K, -1)
        peak2d = flat2d.max(dim=2)[0]              # (C, K) raw peak value
        # spread = #pixels >= 0.5*peak (tightness of the blob)
        thr = (0.5 * peak2d).clamp(min=1e-6).view(C, K, 1)
        spread = (flat2d >= thr).sum(dim=2).float()  # (C, K)
        # heatmap scale ~255 (HybridNet conf divides 3D heatmap by 255)
        peak2d_norm = (peak2d / 255.0).clamp(max=1.0)

        # --- replicate the real HybridNet 3D fusion exactly ---
        heatmaps_b = heatmaps.unsqueeze(0)
        heatmaps_padded = F.pad(heatmaps_b, [1, 1, 1, 1], mode='constant',
                                value=0.0)
        heatmaps3D = hybridNet.reproLayer(
            heatmaps_padded, center3D.int().unsqueeze(0),
            centerHMs.unsqueeze(0), cameraMatrices.unsqueeze(0))
        heatmap_final = hybridNet.softplus(hybridNet.v2vNet(heatmaps3D / 255.))
        norm = torch.sum(heatmap_final, dim=[2, 3, 4])
        x = torch.sum(heatmap_final * hybridNet.xx, dim=[2, 3, 4]) / norm
        y = torch.sum(heatmap_final * hybridNet.yy, dim=[2, 3, 4]) / norm
        z = torch.sum(heatmap_final * hybridNet.zz, dim=[2, 3, 4]) / norm
        points3D = torch.stack([x, y, z], dim=2)
        flat = heatmap_final.view(*heatmap_final.shape[:2], -1)
        confidences = torch.clamp(torch.max(flat, dim=2)[0], max=255.) / 255.
        points3D = (points3D.transpose(0, 1) * hybridNet.grid_spacing * 2
                    - hybridNet.grid_size / 2.
                    + center3D.int().unsqueeze(0)).transpose(0, 1)

    # dump 2D-argmax (red) vs 3D-reproj (blue) overlays for first 2 flies
    if OVERLAY_DIR and _call['n'] < 2:
        _dump_2d_overlay(imgs_4ch, heatmaps, f"fly{_call['n']}",
                         points3D=points3D.squeeze(0), centerHMs=centerHMs)
    _call['n'] += 1

    # record: average 2D peak/spread over cameras -> per keypoint
    ACC['2D_peak_raw'].append(peak2d.mean(dim=0).cpu().numpy())
    ACC['2D_peak_norm'].append(peak2d_norm.mean(dim=0).cpu().numpy())
    ACC['2D_spread_px'].append(spread.mean(dim=0).cpu().numpy())
    ACC['3D_conf'].append(confidences.squeeze(0).cpu().numpy())
    return points3D.squeeze(0), confidences.squeeze(0)


P.run_kp_and_3d = instrumented_run_kp_and_3d

if __name__ == '__main__':
    P.main()
    print('\n================ KP-vs-HYBRID ISOLATION ================')
    for k in REC:
        a = np.concatenate([np.asarray(x).ravel() for x in ACC[k]]) \
            if ACC[k] else np.array([])
        if a.size:
            print(f'{k:14s}: mean {np.nanmean(a):8.3f}  median '
                  f'{np.nanmedian(a):8.3f}  p10 {np.nanpercentile(a,10):8.3f}'
                  f'  p90 {np.nanpercentile(a,90):8.3f}  n={a.size}')
    print('spread = #heatmap pixels >= 50% of peak (small = sharp/confident; '
          'large = diffuse).')
    print('2D_peak_norm is the 2D KeypointDetect peak / 255; 3D_conf is what '
          'HybridNet writes to the CSV.')
