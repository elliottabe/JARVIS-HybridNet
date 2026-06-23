"""Test whether HybridNet's collapse is driven by the center3D source mismatch:
training centers the volume on the GT-keypoint bbox center; inference centers it
on the SAM3 mask centroid. Here we (1) measure the offset between the inference
center3D (SAM3) and the DLT-triangulated keypoint centroid, and (2) re-run the
v2vNet fusion with the volume recentered on that triangulated centroid, comparing
3D confidence. If conf recovers, recentering is the fix."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.predict3D_multianimal as P

_REPRO = {'tool': None}
_orig_pb = P.predict_bout
def _wrap_pb(bout, bout_masks, video_paths, cd, kp, hn, repro_tool, *a, **k):
    _REPRO['tool'] = repro_tool
    return _orig_pb(bout, bout_masks, video_paths, cd, kp, hn, repro_tool, *a, **k)
P.predict_bout = _wrap_pb

ACC = {'offset': [], 'conf_orig': [], 'conf_recenter': []}
_n = {'i': 0}
BBOX = 224


def _fuse(hn, hm, center3D, centerHMs, camMats):
    hmb = F.pad(hm.unsqueeze(0), [1, 1, 1, 1], mode='constant', value=0.0)
    h3d = hn.reproLayer(hmb, center3D.int().unsqueeze(0), centerHMs.unsqueeze(0),
                        camMats.unsqueeze(0))
    hf = hn.softplus(hn.v2vNet(h3d / 255.))
    conf = (torch.clamp(hf.view(*hf.shape[:2], -1).max(2)[0], max=255.) / 255.).squeeze(0)
    return conf


def instrumented(kp_model, hybridNet, imgs_4ch, center3D, centerHMs, cameraMatrices):
    out = P._orig_run(kp_model, hybridNet, imgs_4ch, center3D, centerHMs, cameraMatrices)
    if _n['i'] < 40:
        rt = _REPRO['tool']
        with torch.no_grad():
            hm = kp_model(imgs_4ch)[1]
            C, K, h, w = hm.shape
            flat = hm.view(C, K, -1); peak, idx = flat.max(2)
            ys = (idx // w).float(); xs = (idx % w).float(); chm = centerHMs.float()
            fx = xs * (448.0 / w) + (chm[:, 0:1] - BBOX)
            fy = ys * (448.0 / h) + (chm[:, 1:2] - BBOX)
            pts3d = []
            for kk in range(K):
                p = torch.stack([fx[:, kk], fy[:, kk]], 0)
                X = rt.reconstructPoint(p.clone(), peak[:, kk].clamp(min=1e-3).view(C, 1, 1))
                if torch.isfinite(X).all():
                    pts3d.append(X)
            if len(pts3d) < 5:
                return out
            tri = torch.stack(pts3d)                       # (K',3)
            c_tri = tri.median(0).values                   # robust centroid
            ACC['offset'].append((c_tri - center3D).norm().item())
            ACC['conf_orig'].append(out[1].mean().item())
            ACC['conf_recenter'].append(_fuse(hybridNet, hm, c_tri, centerHMs,
                                              cameraMatrices).mean().item())
        _n['i'] += 1
    return out


P._orig_run = P.run_kp_and_3d
P.run_kp_and_3d = instrumented

if __name__ == '__main__':
    P.main()
    o = np.array(ACC['offset']); co = np.array(ACC['conf_orig']); cr = np.array(ACC['conf_recenter'])
    print('\n================ center3D offset / recenter test ================')
    if o.size:
        print(f'offset |SAM3 center - triangulated centroid| (3D units): '
              f'mean {o.mean():.2f} median {np.median(o):.2f} p90 {np.percentile(o,90):.2f}  '
              f'(ROI half = 24)')
        print(f'conf  original  (SAM3 center): mean {co.mean():.3f}')
        print(f'conf  recentered (tri centroid): mean {cr.mean():.3f}')
    print('If conf_recentered >> conf_original -> the SAM3-centroid center3D '
          'offset is the cause and recentering fixes it.')
