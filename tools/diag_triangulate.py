"""Bypass HybridNet's v2vNet: triangulate the 2D-argmax KeypointDetect points
directly via DLT (repro_tool.reconstructPoint) and measure reprojection
residual. If residual is small, the 2D detections + calibration are correct and
the collapse is purely HybridNet's v2vNet (and DLT is a viable workaround)."""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.predict3D_multianimal as P

_REPRO = {'tool': None}
_orig_pb = P.predict_bout
def _wrap_pb(bout, bout_masks, video_paths, cd, kp, hn, repro_tool, *a, **k):
    _REPRO['tool'] = repro_tool
    return _orig_pb(bout, bout_masks, video_paths, cd, kp, hn, repro_tool, *a, **k)
P.predict_bout = _wrap_pb

ACC = {'resid_px': [], 'tri_conf': [], 'hyb_conf': []}
_n = {'i': 0}
BBOX = 224  # crop half-size (448/2)


def instrumented(kp_model, hybridNet, imgs_4ch, center3D, centerHMs, cameraMatrices):
    out = P._orig_run(kp_model, hybridNet, imgs_4ch, center3D, centerHMs, cameraMatrices)
    if _n['i'] < 25:                      # only first 25 detections (speed)
        rt = _REPRO['tool']
        with torch.no_grad():
            hm = kp_model(imgs_4ch)[1]            # (C,K,h,w)
            C, K, h, w = hm.shape
            flat = hm.view(C, K, -1)
            peak, idx = flat.max(dim=2)           # (C,K)
            ys = (idx // w).float(); xs = (idx % w).float()
            chm = centerHMs.float()               # (C,2) full-image crop centers
            # 2D-argmax -> full-image px
            fx = xs * (448.0 / w) + (chm[:, 0:1] - BBOX)   # (C,K)
            fy = ys * (448.0 / h) + (chm[:, 1:2] - BBOX)
            for kk in range(K):
                pts = torch.stack([fx[:, kk], fy[:, kk]], 0)        # (2,C)
                wts = peak[:, kk].clamp(min=1e-3).view(C, 1, 1)     # (C,1,1)
                X = rt.reconstructPoint(pts.clone(), wts)           # (3,)
                if not torch.isfinite(X).all():
                    continue
                rep = rt.reprojectPoint(X.unsqueeze(0))             # (C,2)
                res = torch.sqrt(((rep - torch.stack([fx[:, kk], fy[:, kk]], 1)) ** 2)
                                 .sum(1))                            # (C,)
                ACC['resid_px'].append(res.mean().item())
            ACC['hyb_conf'].append(out[1].mean().item())
        _n['i'] += 1
    return out


P._orig_run = P.run_kp_and_3d
P.run_kp_and_3d = instrumented

if __name__ == '__main__':
    P.main()
    r = np.array(ACC['resid_px']); hc = np.array(ACC['hyb_conf'])
    print('\n================ DLT triangulation of 2D-argmax ================')
    if r.size:
        print(f'reprojection residual (px): mean {r.mean():.2f}  median {np.median(r):.2f}  '
              f'p90 {np.percentile(r,90):.2f}  (crop is 448px)')
    if hc.size:
        print(f'HybridNet 3D conf same frames: mean {hc.mean():.3f}')
    print('Small residual (~<5px) => 2D+calibration correct => collapse is the '
          'v2vNet, and DLT triangulation is a usable workaround.')
