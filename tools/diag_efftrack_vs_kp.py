"""Test the train/inference heatmap-source mismatch hypothesis for HybridNet.

HybridNet was TRAINED fusing its internal 3-ch effTrack's 2D heatmaps (mask
channel dropped), but INFERENCE feeds it the external 4-ch (mask-conditioned)
KeypointDetect heatmaps. This compares the resulting 3D confidence:
  conf_4ch : external 4-ch KP -> reproLayer/v2vNet   (current inference path)
  conf_3ch : internal effTrack on 3-ch RGB crop -> reproLayer/v2vNet  (training-matched)
If conf_3ch >> conf_4ch, the collapse is caused by the heatmap-source mismatch,
not by HybridNet training quality.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools.predict3D_multianimal as P

ACC = {'conf_4ch': [], 'conf_3ch': []}


def _fuse(hybridNet, heatmaps, center3D, centerHMs, cameraMatrices):
    heatmaps = heatmaps.unsqueeze(0)
    heatmaps_padded = F.pad(heatmaps, [1, 1, 1, 1], mode='constant', value=0.0)
    h3d = hybridNet.reproLayer(heatmaps_padded, center3D.int().unsqueeze(0),
                               centerHMs.unsqueeze(0), cameraMatrices.unsqueeze(0))
    hf = hybridNet.softplus(hybridNet.v2vNet(h3d / 255.))
    flat = hf.view(*hf.shape[:2], -1)
    conf = (torch.clamp(torch.max(flat, dim=2)[0], max=255.) / 255.).squeeze(0)
    return conf


def instrumented(kp_model, hybridNet, imgs_4ch, center3D, centerHMs, cameraMatrices):
    with torch.no_grad():
        hm_4ch = kp_model(imgs_4ch)[1]                       # external 4-ch KP
        hm_3ch = hybridNet.effTrack(imgs_4ch[:, :3])[1]      # internal 3-ch effTrack
        c4 = _fuse(hybridNet, hm_4ch, center3D, centerHMs, cameraMatrices)
        c3 = _fuse(hybridNet, hm_3ch, center3D, centerHMs, cameraMatrices)
    ACC['conf_4ch'].append(c4.cpu().numpy())
    ACC['conf_3ch'].append(c3.cpu().numpy())
    # return real inference path so the run proceeds normally
    return P._orig_run_kp_and_3d(kp_model, hybridNet, imgs_4ch, center3D,
                                 centerHMs, cameraMatrices)


P._orig_run_kp_and_3d = P.run_kp_and_3d
P.run_kp_and_3d = instrumented

if __name__ == '__main__':
    P.main()
    print('\n================ effTrack(3ch) vs KP(4ch) ================')
    for k in ('conf_4ch', 'conf_3ch'):
        a = np.concatenate([x.ravel() for x in ACC[k]]) if ACC[k] else np.array([])
        if a.size:
            print(f'{k}: mean {np.nanmean(a):.3f}  median {np.nanmedian(a):.3f}  '
                  f'p10 {np.nanpercentile(a,10):.3f}  p90 {np.nanpercentile(a,90):.3f}')
    print('If conf_3ch >> conf_4ch -> HybridNet collapse is the train/inference '
          'heatmap-source mismatch (3-ch effTrack trained vs 4-ch KP inferred).')
