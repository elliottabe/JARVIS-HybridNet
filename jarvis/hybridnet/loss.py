"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = 0
        for i,gt_batch in enumerate(gt):
            for j, gt_single in enumerate(gt_batch):
                if torch.sum(gt_single) > 1:
                    loss += torch.mean(((pred[i][j] - gt_single)**2))
        return loss


def build_skeleton_edges(keypoint_names, skeleton):
    """Map skeleton bones (pairs of keypoint NAMES) to index pairs.

    Returns a LongTensor (E, 2). Bones referencing unknown names are skipped.
    """
    name_to_idx = {n: i for i, n in enumerate(keypoint_names)}
    edges = []
    for bone in skeleton:
        a, b = bone[0], bone[1]
        if a in name_to_idx and b in name_to_idx:
            edges.append([name_to_idx[a], name_to_idx[b]])
    if not edges:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)


def graph_laplacian_loss(pred, gt, valid, ei, ej):
    """Bone-vector (graph-gradient) shape prior.

    For each skeleton bone (i, j) penalize the squared difference between the
    predicted and GT bone VECTORS: ||(pred_i - pred_j) - (gt_i - gt_j)||^2.
    This matches each bone's length AND orientation to GT, anchoring distal
    keypoints (e.g. tarsal tips) to their parents. Translation-invariant
    (uses differences); bones with a missing endpoint are masked out.

    Args:
        pred, gt: (B, K, 3) predicted and GT 3D keypoints (same world frame).
        valid:    (B, K) bool mask of present keypoints.
        ei, ej:   (E,) long tensors of bone endpoint indices.
    Returns:
        scalar tensor: mean squared bone-vector error over valid bones.
    """
    if ei.numel() == 0:
        return pred.new_zeros(())
    dpred = pred[:, ei, :] - pred[:, ej, :]          # (B, E, 3)
    dgt = gt[:, ei, :] - gt[:, ej, :]                # (B, E, 3)
    emask = (valid[:, ei] & valid[:, ej]).unsqueeze(-1).float()  # (B, E, 1)
    se = ((dpred - dgt) ** 2) * emask
    return se.sum() / emask.sum().clamp(min=1.0)
