"""Regression test for HybridNet batched-frameset support.

Before the ReprojectionLayer fix, `HybridNetBackbone.forward` collapsed the
batch dimension to 1 (ReprojectionLayer only processed index 0), so any
HYBRIDNET.BATCH_SIZE > 1 silently trained on a single frameset. This test
checks that a batched forward equals per-sample (batch-1) forwards stacked.

Requires a loadable JARVIS project with a 3D dataset + masks (default:
unified_V3_masked, override with JARVIS_TEST_PROJECT). Skips if unavailable.

Run: python tests/test_hybridnet_batch.py
"""
import os
import sys

import torch
from torch.utils.data import DataLoader


def main():
    project = os.environ.get("JARVIS_TEST_PROJECT", "unified_V3_masked")
    from jarvis.config.project_manager import ProjectManager
    from jarvis.dataset.dataset3D import Dataset3D
    from jarvis.hybridnet.hybridnet import HybridNet

    pm = ProjectManager()
    if not pm.load(project):
        print(f"SKIP: could not load project {project}")
        return 0
    cfg = pm.get_cfg()
    model = HybridNet("train", cfg).model.cuda().eval()
    ds = Dataset3D(cfg, set="train")
    img_size = torch.tensor(cfg.DATASET.IMAGE_SIZE).cuda()
    data = next(iter(DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)))
    imgs = data[0].permute(0, 1, 4, 2, 3).float().cuda()
    cHM, c3D, camM = data[2].cuda(), data[3].cuda(), data[5].cuda()
    B = imgs.shape[0]

    with torch.no_grad():
        hm_b, _, pts_b, _ = model(imgs, img_size, cHM, c3D, camM)
        hm_i = torch.cat([model(imgs[i:i+1], img_size, cHM[i:i+1],
                                c3D[i:i+1], camM[i:i+1])[0] for i in range(B)], 0)
        pts_i = torch.cat([model(imgs[i:i+1], img_size, cHM[i:i+1],
                                 c3D[i:i+1], camM[i:i+1])[2] for i in range(B)], 0)

    assert hm_b.shape[0] == B, f"batch dim collapsed: {hm_b.shape}"
    assert pts_b.shape[0] == B
    # 3D points (used by the heatmap-MSE + graph-Laplacian losses): exact match
    assert torch.allclose(pts_b, pts_i, atol=1e-3), \
        f"points mismatch {(pts_b - pts_i).abs().max()}"
    # heatmaps: equal within fp32 3D-conv numerical noise (scale ~1)
    assert torch.allclose(hm_b, hm_i, atol=5e-3), \
        f"heatmap mismatch {(hm_b - hm_i).abs().max()}"
    print(f"PASS: HybridNet batched == per-sample for B={B}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
