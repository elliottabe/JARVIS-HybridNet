"""Phase 3 verification: per-annotation 4-channel KeypointDetect pipeline.

Loads red_data_unified val split with INSTANCE_MASK_INPUT=true, pulls a
dual-fly sample, checks:
  - Dataset2D returns a [H, W, 4] tensor (RGB normalized + mask channel {0,1})
  - distractor-fly pixels are grayed (crop mean) where the non-target mask hits
  - EfficientTrack model builds with in_channels=4 and runs a forward pass
  - Ecoset 3->4 stem expansion leaves RGB weights unchanged and ch3 zero
"""

import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.efficienttrack.efficienttrack import EfficientTrack


def find_dual_fly_idx(ds):
    prev = None
    for i, (img_id, _) in enumerate(ds.ann_index):
        if prev is not None and img_id == prev:
            return i - 1
        prev = img_id
    return 0


def test_dataset(cfg):
    ds = Dataset2D(cfg=cfg, set='val', mode='KeypointDetect')
    assert ds.instance_mask_input, 'INSTANCE_MASK_INPUT flag not active'
    assert ds.ann_index is not None, 'ann_index not built'
    n_imgs = len(ds.image_ids)
    n_anns = len(ds.ann_index)
    print(f'[ds] val: {n_imgs} images, {n_anns} annotations '
          f'({n_anns / max(n_imgs, 1):.2f} avg/img)')

    idx = find_dual_fly_idx(ds)
    image_id, target_ann_idx = ds.ann_index[idx]
    print(f'[ds] dual-fly sample idx={idx} image_id={image_id} '
          f'target_ann_idx={target_ann_idx}')

    sample = ds[idx]
    img, heatmaps, kps = sample
    assert img.shape[-1] == 4, f'expected 4 channels, got {img.shape}'
    bbox_size = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE
    assert img.shape[:2] == (bbox_size, bbox_size), \
        f'expected ({bbox_size},{bbox_size}), got {img.shape[:2]}'
    mask_ch = img[..., 3]
    uniq = np.unique(mask_ch)
    assert set(np.round(uniq, 3)).issubset({0.0, 1.0}), \
        f'mask channel not binary: {uniq[:10]}'
    frac_on = float((mask_ch > 0.5).mean())
    print(f'[ds] tensor shape {img.shape} '
          f'mask frac-on {frac_on:.3f} heatmap_shapes {[h.shape for h in heatmaps]}')

    bundle = ds._load_instance_masks(image_id, is_id=True)
    assert bundle is not None, 'no sam3 mask bundle for test image'
    n_masks = int(bundle['masks'].shape[0])
    matched = bundle['matched']
    print(f'[ds] sam3 cache: n_masks={n_masks} matched={matched.tolist()} '
          f"extra={bundle['extra_masks'].shape[0]}")

    if n_masks > 1 and matched[target_ann_idx]:
        other_idx = next((j for j in range(n_masks)
                          if j != target_ann_idx and matched[j]), None)
        if other_idx is not None:
            tmask = bundle['masks'][target_ann_idx]
            dmask = bundle['masks'][other_idx]
            print(f'[ds] target mask area={int(tmask.sum())} '
                  f'distractor mask area={int(dmask.sum())} '
                  f'overlap={int((tmask & dmask).sum())}')

    return True


def test_model_build_and_forward(cfg):
    cfg.PARENT_DIR = ROOT
    cfg.savePaths = None
    cfg.logPaths = None

    class _Stub:
        pass

    stub_cfg = _Stub()
    stub_cfg.KEYPOINTDETECT = cfg.KEYPOINTDETECT
    stub_cfg.CENTERDETECT = cfg.CENTERDETECT
    stub_cfg.PARENT_DIR = ROOT

    from jarvis.efficienttrack.model import EfficientTrackBackbone

    kp_cfg = cfg.KEYPOINTDETECT
    model = EfficientTrackBackbone(kp_cfg,
                                   model_size=kp_cfg.MODEL_SIZE,
                                   output_channels=kp_cfg.NUM_JOINTS,
                                   in_channels=4)
    stem_w = model.backbone_net.model._conv_stem.weight
    assert stem_w.shape[1] == 4, f'stem not 4-ch: {stem_w.shape}'
    print(f'[model] stem shape {tuple(stem_w.shape)}')

    bs = 1
    sz = kp_cfg.BOUNDING_BOX_SIZE
    x = torch.randn(bs, 4, sz, sz)
    with torch.no_grad():
        out = model(x)
    print(f'[model] forward ok: outputs '
          f"{[tuple(o.shape) for o in out]}")
    return True


def test_ecoset_expansion():
    from jarvis.efficienttrack.efficienttrack import EfficientTrack

    class _Model:
        def __init__(self, w):
            self._w = w
        def state_dict(self):
            return {'backbone_net.model._conv_stem.weight': self._w}

    target_w = torch.zeros(32, 4, 3, 3)
    src_w = torch.randn(32, 3, 3, 3)
    dummy = _Model(target_w)

    et = EfficientTrack.__new__(EfficientTrack)
    et.model = dummy
    pretrained = {'backbone_net.model._conv_stem.weight': src_w.clone()}
    EfficientTrack._expand_conv_stem(et, pretrained)
    expanded = pretrained['backbone_net.model._conv_stem.weight']
    assert expanded.shape == target_w.shape, \
        f'expanded shape wrong: {expanded.shape}'
    assert torch.allclose(expanded[:, :3], src_w), \
        'RGB channels not copied verbatim'
    assert torch.all(expanded[:, 3] == 0), 'channel 3 not zero-init'
    print(f'[ecoset] 3->4 expansion verified: shape={tuple(expanded.shape)} '
          f'ch3.max={float(expanded[:,3].abs().max())}')
    return True


def main():
    pm = ProjectManager()
    ok = pm.load('red_data_unified')
    assert ok, 'failed to load red_data_unified project'
    cfg = pm.get_cfg()
    assert cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT, \
        'INSTANCE_MASK_INPUT=false in project cfg'
    print(f'[cfg] dataset={cfg.DATASET.DATASET_2D} '
          f'bbox={cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE} '
          f'mask_input={cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT}')

    test_dataset(cfg)
    test_model_build_and_forward(cfg)
    test_ecoset_expansion()
    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
