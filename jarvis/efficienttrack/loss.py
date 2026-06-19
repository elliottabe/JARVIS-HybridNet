"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapLoss(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()

    def forward(self, outputs, heatmaps):
        heatmaps_losses = []
        for idx in range(len(outputs)):
            loss = ((outputs[idx] - heatmaps[idx])**2)
            loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
            heatmaps_losses.append(loss)
        return heatmaps_losses


def mask_containment_loss(outputs, gt_heatmaps, mask_full,
                          dilate=11, eps=1e-6):
    """Soft penalty for predicted-heatmap mass landing OFF the fly body.

    For each output scale, penalize the fraction of each keypoint's positive
    predicted heatmap mass that lies OUTSIDE the dilated target mask AND away
    from the GT keypoint. The GT heatmap is used as a 'protected' region so we
    never fight the supervision on keypoints that legitimately fall outside a
    partial / occluded SAM mask (e.g. a female leg tip past the mask edge).

    Args:
        outputs:     tuple/list of predicted heatmaps, each (B, K, h, w).
        gt_heatmaps: matching GT heatmaps, each (B, K, h, w).
        mask_full:   (B, 1, H, W) target mask in {0,1} at INPUT resolution
                     (the KP net's 4th input channel).
        dilate:      max-pool kernel (px, input res) for mask-edge slack.
    Returns:
        scalar tensor: mean off-body mass fraction over scales (0 = all on-body).
    """
    m = (mask_full > 0.5).float()
    if dilate and dilate > 1:
        m = F.max_pool2d(m, kernel_size=dilate, stride=1, padding=dilate // 2)
    total = mask_full.new_zeros(())
    n = 0
    for pred, gt in zip(outputs, gt_heatmaps):
        h, w = pred.shape[-2:]
        mlr = F.interpolate(m, size=(h, w), mode='nearest')          # (B,1,h,w)
        gtn = gt / (gt.amax(dim=(2, 3), keepdim=True) + eps)         # (B,K,h,w)
        protected = torch.clamp(torch.maximum(mlr, gtn), 0.0, 1.0)   # (B,K,h,w)
        pos = F.relu(pred)
        outside = pos * (1.0 - protected)
        frac = outside.sum(dim=(2, 3)) / (pos.sum(dim=(2, 3)) + eps)  # (B,K)
        total = total + frac.mean()
        n += 1
    return total / max(n, 1)
