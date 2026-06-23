"""Evaluate the trained HybridNet on its OWN val set (Dataset3D val) and report
3D-heatmap confidence + per-keypoint 3D error. If val conf is high (~0.9) while
Session0 courtship inference is ~0.55, the weights are fine and the collapse is a
Session0 geometry/centering gap, not a training failure."""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.hybridnet.hybridnet import HybridNet

HYB = ('projects/unified_V3_masked/models/HybridNet/'
       'Run_20260620-085937/HybridNet-large_final.pth')

pm = ProjectManager(); pm.load('unified_V3_masked'); cfg = pm.get_cfg()
val = Dataset3D(cfg, set='val')
loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=4)
hn = HybridNet('inference', cfg, weights=HYB)
hn.model.cuda().eval()

confs, errs, cubes = [], [], []
with torch.no_grad():
    for n, data in enumerate(loader):
        imgs, kps, centerHM, center3D, hm3d, camMats = [d.cuda() for d in data[:6]]
        img_size = torch.tensor(cfg.DATASET.IMAGE_SIZE).cuda()
        out = hn.model(imgs, img_size, centerHM, center3D, camMats)
        pts, conf = out[2], out[3]
        kpn = kps[0].cpu().numpy()
        valid = (np.abs(kpn).sum(1) > 0)
        e = np.sqrt(((pts[0].cpu().numpy() - kpn) ** 2).sum(1))[valid]
        errs.append(e)
        confs.append(conf[0].cpu().numpy()[valid])
        kv = kpn[valid]
        cubes.append((kv.max(0) - kv.min(0)).max())  # per-axis span, max axis
        if n >= 120:
            break
c = np.concatenate(confs); e = np.concatenate(errs); cu = np.array(cubes)
print('\n================ HybridNet on its OWN val set ================')
print(f'frames={len(cubes)}  val 3D conf: mean {c.mean():.3f} median {np.median(c):.3f}'
      f'  p10 {np.percentile(c,10):.3f}')
print(f'val 3D error (units): mean {e.mean():.3f} median {np.median(e):.3f}')
print(f'val fly cube span (units): mean {cu.mean():.1f} median {np.median(cu):.1f} '
      f'max {cu.max():.1f}  (ROI_CUBE_SIZE={cfg.HYBRIDNET.ROI_CUBE_SIZE})')
print('Compare to Session0 courtship inference conf ~0.55.')
