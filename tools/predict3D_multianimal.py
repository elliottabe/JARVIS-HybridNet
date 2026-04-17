"""Phase-4 multi-animal 3D prediction on courtship bouts (Option B).

Per bout:
  1. Decode bout frames from each camera video.
  2. Run SAM3 video mode to get per-camera masks with persistent IDs.
  3. Cross-camera identity alignment via 3D back-projection of mask centroids.
  4. For each frame × fly:
       - CenterDetect (3-ch) → refined 2D crop center (triangulated → 3D center)
       - Build a 4-channel KeypointDetect crop: RGB with other flies
         replaced by the crop's mean color, plus the target fly's SAM3 mask
         resized to the crop as the 4th channel. This matches the
         `Dataset2D._get_item_keypoints` training preprocessing.
       - Run the standalone 4-ch KeypointDetect → 2D heatmaps per camera.
       - Feed those heatmaps into HybridNet's `reproLayer + v2vNet` stack
         (skipping HybridNet's internal 3-ch effTrack) → 3D keypoints.
  5. Emit per-bout CSVs `fly0.csv` / `fly1.csv` with JARVIS's standard schema
     (x,y,z,conf repeated per joint).

Usage:
  python tools/predict3D_multianimal.py --session \
      /data2/users/eabe/datasets/Johnson_lab/courtship/Session0/2025_10_20_13_20_04 \
      --bouts-csv courtship_bouts_unified_summary.csv \
      --out /data2/users/eabe/.../Predictions_3D_V4_phase4
"""

import argparse
import csv
import gc
import itertools
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from jarvis.config.project_manager import ProjectManager
from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.utils.reprojection import get_repro_tool
from jarvis.prediction.sam3_video_tracker import SAM3VideoTracker


DEFAULT_CENTER_WEIGHTS = os.path.join(
    ROOT, 'projects/red_data_unified/models/CenterDetect/'
    'phase4_center_ft/EfficientTrack-medium_final.pth')
DEFAULT_KP_WEIGHTS = os.path.join(
    ROOT, 'projects/red_data_unified/models/KeypointDetect/'
    'phase4_kp_ft/EfficientTrack-medium_final.pth')
DEFAULT_HYBRIDNET_WEIGHTS = 'latest'  # resolved inside HybridNet.load_weights


def build_models(cfg, center_weights, kp_weights, hybridnet_weights):
    """Instantiate the three nets. KeypointDetect is 4-ch (driven by
    cfg.KEYPOINTDETECT.INSTANCE_MASK_INPUT). HybridNet is loaded with its
    own 3-ch internal effTrack — we bypass that at inference and only use
    its reproLayer + v2vNet + grid buffers."""
    assert getattr(cfg.KEYPOINTDETECT, 'INSTANCE_MASK_INPUT', False), \
        'This tool expects a 4-channel KeypointDetect; set ' \
        'KEYPOINTDETECT.INSTANCE_MASK_INPUT: true in the project config.'

    cd = EfficientTrack('CenterDetectInference', cfg,
                        weights=center_weights)
    cd.model.cuda().eval()

    kp = EfficientTrack('KeypointDetectInference', cfg,
                        weights=kp_weights)
    kp.model.cuda().eval()
    assert kp.in_channels == 4, \
        f'KeypointDetect first conv not 4-ch (got {kp.in_channels}).'

    hn = HybridNet('inference', cfg, weights=hybridnet_weights)
    hn.model.cuda().eval()

    return cd.model, kp.model, hn.model


def parse_bouts(csv_path, session_tag):
    """Return list of dicts {bout_idx, start, end, source_fly} filtered to
    rows whose `fly_id` column matches `session_tag` (so a single CSV can
    cover a whole session tree)."""
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if session_tag and r.get('fly_id') != session_tag:
                continue
            rows.append({
                'bout_idx': int(r['bout_idx']),
                'start': int(r['start_frame']),
                'end': int(r['end_frame']),
                'source_fly': r.get('source_fly', ''),
            })
    return rows


def get_video_paths(session_dir, repro_tool):
    """Match camera folder names from the ReprojectionTool to mp4 files in
    the session directory."""
    paths = []
    for cam_name in repro_tool.cameras:
        candidate = os.path.join(session_dir, f'{cam_name}.mp4')
        if not os.path.isfile(candidate):
            raise FileNotFoundError(
                f'Missing video for camera {cam_name}: {candidate}')
        paths.append(candidate)
    return paths


def read_frame_all_cams(caps, frame_idx=None):
    """Read the next frame from each capture sequentially. Callers must
    seek each cap to the bout's start frame exactly once beforehand —
    per-frame seeks collapse GPU utilization because every seek triggers
    a decode from the preceding keyframe.
    `frame_idx` is ignored here; kept for signature stability with the
    previous seeking version.
    """
    frames = []
    for cap in caps:
        ok, img = cap.read()
        if not ok:
            return None
        frames.append(img)
    return np.stack(frames, axis=0)


# NOTE: CenterDetect-refined crop centers (as used in
# JarvisMultiAnimalPredictor3D._forward_with_precomputed_masks) are not
# wired in yet. We use SAM3 mask centroids triangulated to 3D as the crop
# center — this is close enough for Phase-4 evaluation, since the
# training code also centers crops at the annotation bbox center (roughly
# the mask centroid). Upgrade path: call multi_peak.extract_top_k_peaks
# and match each SAM3 identity to its nearest CD peak.


def clamp_crop_centers(centerHMs, img_size_xy, bbox_hw):
    centerHMs = centerHMs.clone().int()
    centerHMs[:, 0] = torch.clamp(
        centerHMs[:, 0], bbox_hw, img_size_xy[0] - bbox_hw)
    centerHMs[:, 1] = torch.clamp(
        centerHMs[:, 1], bbox_hw, img_size_xy[1] - bbox_hw)
    return centerHMs


def build_4ch_crops(imgs_rgb_u8, centerHMs, target_masks, distractor_masks,
                    bbox_hw, mean_rgb, std_rgb, device='cuda'):
    """Build normalized 4-channel input tensor for the KP net.

    Args:
        imgs_rgb_u8: (num_cams, H, W, 3) uint8 RGB full-resolution images.
        centerHMs: (num_cams, 2) int crop centers (x, y).
        target_masks: (num_cams, H, W) bool — target fly only.
        distractor_masks: (num_cams, H, W) bool — union of all other flies.
        bbox_hw: half the crop side (448/2 = 224 for red_data_unified).
        mean_rgb, std_rgb: per-channel floats, from cfg.DATASET.
    Returns:
        (num_cams, 4, crop, crop) float tensor on CUDA.
    """
    num_cams = imgs_rgb_u8.shape[0]
    crop = bbox_hw * 2
    out = np.zeros((num_cams, 4, crop, crop), dtype=np.float32)

    for i in range(num_cams):
        cx, cy = int(centerHMs[i, 0].item()), int(centerHMs[i, 1].item())
        rgb = imgs_rgb_u8[i,
                          cy - bbox_hw:cy + bbox_hw,
                          cx - bbox_hw:cx + bbox_hw, :].astype(np.float32)
        # Replace distractor-fly pixels with crop mean (matches training).
        if distractor_masks is not None:
            dmask = distractor_masks[i,
                                     cy - bbox_hw:cy + bbox_hw,
                                     cx - bbox_hw:cx + bbox_hw]
            if dmask.any():
                mean_color = rgb.mean(axis=(0, 1))
                rgb[dmask] = mean_color

        rgb /= 255.0
        rgb = (rgb - np.array(mean_rgb)) / np.array(std_rgb)
        out[i, :3] = rgb.transpose(2, 0, 1)

        if target_masks is not None:
            tmask = target_masks[i,
                                 cy - bbox_hw:cy + bbox_hw,
                                 cx - bbox_hw:cx + bbox_hw]
            out[i, 3] = tmask.astype(np.float32)

    return torch.from_numpy(out).to(device)


def run_kp_and_3d(kp_model, hybridNet, imgs_4ch, center3D, centerHMs,
                  cameraMatrices):
    """Option B: 4-ch KP heatmaps → HybridNet's reproLayer + v2vNet.
    Mirrors `HybridNetBackbone.forward`, substituting `self.effTrack` for
    our external 4-ch KP model."""
    with torch.no_grad():
        heatmaps = kp_model(imgs_4ch)[1]           # (num_cams, K, h, w)
        heatmaps = heatmaps.unsqueeze(0)           # add batch dim
        heatmaps_padded = F.pad(
            heatmaps, [1, 1, 1, 1], mode='constant', value=0.0)

        heatmaps3D = hybridNet.reproLayer(
            heatmaps_padded,
            center3D.int().unsqueeze(0),
            centerHMs.unsqueeze(0),
            cameraMatrices.unsqueeze(0),
        )
        heatmap_final = hybridNet.softplus(hybridNet.v2vNet(heatmaps3D / 255.))

        norm = torch.sum(heatmap_final, dim=[2, 3, 4])
        x = torch.sum(heatmap_final * hybridNet.xx, dim=[2, 3, 4]) / norm
        y = torch.sum(heatmap_final * hybridNet.yy, dim=[2, 3, 4]) / norm
        z = torch.sum(heatmap_final * hybridNet.zz, dim=[2, 3, 4]) / norm
        points3D = torch.stack([x, y, z], dim=2)

        flat = heatmap_final.view(*heatmap_final.shape[:2], -1)
        confidences = torch.clamp(torch.max(flat, dim=2)[0], max=255.) / 255.

        points3D = (
            points3D.transpose(0, 1) * hybridNet.grid_spacing * 2
            - hybridNet.grid_size / 2.
            + center3D.int().unsqueeze(0)
        ).transpose(0, 1)

    return points3D.squeeze(0), confidences.squeeze(0)


def csv_writer_for_fly(out_dir, fly_idx, cfg):
    """Create one CSV per fly with the standard JARVIS 3D header."""
    path = os.path.join(out_dir, f'fly{fly_idx}.csv')
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    if len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS:
        joints = list(itertools.chain.from_iterable(
            itertools.repeat(x, 4) for x in cfg.KEYPOINT_NAMES))
        coords = ['x', 'y', 'z', 'conf'] * len(cfg.KEYPOINT_NAMES)
        # Leading 'frame' column marks the absolute video frame index;
        # keeps header width == data width (50*4 + 1).
        w.writerow(['frame'] + joints)
        w.writerow(['frame'] + coords)
    return f, w


MASKS_FILE_VERSION = 1
MASKS_FILENAME = 'sam3_masks.npz'


class LoadedBoutMasks:
    """Replay previously saved SAM3 masks with the same `get_frame` /
    `num_cameras` / `num_frames` surface as `BoutMasks`, so the rest of the
    pipeline can consume them without caring about the source."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        v = int(data.get('version', np.array(0)).item()) if 'version' in data \
            else 0
        if v != MASKS_FILE_VERSION:
            raise ValueError(
                f'{npz_path}: unsupported mask file version {v} '
                f'(expected {MASKS_FILE_VERSION})')
        self.packed = data['packed']          # (A, C, F, H, W_pack) uint8
        self.valid = data['valid']            # (A, C, F) bool
        self.centroids = data['centroids']    # (A, C, F, 2) float32
        self.H, self.W = int(data['shape'][0]), int(data['shape'][1])
        self.num_animals_saved = self.packed.shape[0]
        self.num_cameras = self.packed.shape[1]
        self.num_frames = self.packed.shape[2]
        # Sentinel — mirrors BoutMasks so downstream checks pass.
        self.identity_map = [{} for _ in range(self.num_cameras)]

    def get_frame(self, frame_idx, num_animals=2):
        if not (0 <= frame_idx < self.num_frames):
            return None
        results = []
        for fly_idx in range(num_animals):
            if fly_idx >= self.num_animals_saved:
                results.append(None)
                continue
            fly_masks = np.zeros(
                (self.num_cameras, self.H, self.W), dtype=bool)
            for cam in range(self.num_cameras):
                if self.valid[fly_idx, cam, frame_idx]:
                    packed_row = self.packed[fly_idx, cam, frame_idx]
                    unpacked = np.unpackbits(
                        packed_row, axis=1, bitorder='big')[:, :self.W]
                    fly_masks[cam] = unpacked.astype(bool)
            results.append({
                'masks': torch.from_numpy(fly_masks),
                'centroids': torch.from_numpy(
                    self.centroids[fly_idx, :, frame_idx].astype(np.float32)),
                'valid': torch.from_numpy(
                    self.valid[fly_idx, :, frame_idx].astype(bool)),
            })
        return results

    def assign_identities(self, repro_tool, num_animals=2):
        """Identities are already baked in; no-op for API parity."""
        return


def save_bout_masks(bout_masks, bout_out_dir, num_animals):
    """Serialize SAM3 masks for a bout, organized by global fly identity.

    Format: compressed NPZ with packed-bit masks and metadata so it round-
    trips with `np.unpackbits`. Typical per-bout size is ~50-500 MB after
    compression (masks are sparse).

    After writing, load the file back and compare one frame's masks to
    what's in memory so we fail fast on any format regression.
    """
    if bout_masks.identity_map[0] is None:
        print('  [save_masks] no identity map — skipping')
        return

    # Probe a frame for (H, W).
    H = W = 0
    for cam in range(bout_masks.num_cameras):
        for fi in range(bout_masks.num_frames):
            if bout_masks.masks[cam][fi]:
                first = next(iter(bout_masks.masks[cam][fi].values()))
                H, W = first['mask'].shape
                break
        if H:
            break
    if H == 0:
        print('  [save_masks] no masks at all — skipping')
        return

    nc = bout_masks.num_cameras
    nf = bout_masks.num_frames
    valid = np.zeros((num_animals, nc, nf), dtype=bool)
    centroids = np.zeros((num_animals, nc, nf, 2), dtype=np.float32)
    # Per-frame packed masks along W axis (row-major), shape (H, ceil(W/8)).
    packed_W = (W + 7) // 8
    packed = np.zeros((num_animals, nc, nf, H, packed_W), dtype=np.uint8)

    for fi in range(nf):
        slots = bout_masks.get_frame(fi, num_animals=num_animals)
        if slots is None:
            continue
        for fly_idx, slot in enumerate(slots):
            if slot is None:
                continue
            v = slot['valid'].numpy()
            m = slot['masks'].numpy().astype(bool)
            c = slot['centroids'].numpy()
            for cam in range(nc):
                if v[cam]:
                    valid[fly_idx, cam, fi] = True
                    centroids[fly_idx, cam, fi] = c[cam]
                    packed[fly_idx, cam, fi] = np.packbits(
                        m[cam], axis=1, bitorder='big')

    out_path = os.path.join(bout_out_dir, MASKS_FILENAME)
    np.savez_compressed(
        out_path,
        packed=packed,
        valid=valid,
        centroids=centroids,
        shape=np.array([H, W], dtype=np.int32),
        version=np.array(MASKS_FILE_VERSION, dtype=np.int32),
    )
    sz = os.path.getsize(out_path) / 1e6
    print(f'  [save_masks] wrote {out_path} ({sz:.1f} MB)')

    # Round-trip sanity check: load a frame with valid detections and compare.
    reloaded = LoadedBoutMasks(out_path)
    probe_fi = None
    for fi in range(nf):
        if valid[:, :, fi].any():
            probe_fi = fi
            break
    if probe_fi is None:
        print('  [save_masks] roundtrip skipped — no valid frames')
        return
    live_slots = bout_masks.get_frame(probe_fi, num_animals=num_animals)
    reload_slots = reloaded.get_frame(probe_fi, num_animals=num_animals)
    for fly_idx in range(num_animals):
        a, b = live_slots[fly_idx], reload_slots[fly_idx]
        if a is None and b is None:
            continue
        assert a is not None and b is not None, \
            f'[save_masks] roundtrip mismatch: fly{fly_idx} None vs dict'
        assert bool((a['valid'] == b['valid']).all()), \
            f'[save_masks] roundtrip mismatch: fly{fly_idx} valid differs'
        assert bool((a['masks'] == b['masks']).all()), \
            f'[save_masks] roundtrip mismatch: fly{fly_idx} masks differ'
    print(f'  [save_masks] roundtrip OK on frame {probe_fi}')


def render_overlay(frames_rgb, fly_slots, per_fly_kp_2d, abs_frame, out_path,
                   num_animals, repro_tool=None):
    """Render a 1×num_cams PNG grid: RGB + per-fly mask tint + KP dots."""
    num_cams = frames_rgb.shape[0]
    fig, axes = plt.subplots(1, num_cams, figsize=(2.2 * num_cams, 2.4))
    if num_cams == 1:
        axes = [axes]
    fly_colors = [(1.0, 0.2, 0.2), (0.2, 0.5, 1.0)]

    for cam in range(num_cams):
        ax = axes[cam]
        ax.imshow(frames_rgb[cam])
        ax.set_axis_off()
        ax.set_title(f'cam{cam}', fontsize=8)

        for fly_idx in range(num_animals):
            slot = fly_slots[fly_idx] if fly_slots is not None else None
            if slot is not None and slot['valid'][cam].item():
                mk = slot['masks'][cam].numpy()
                overlay = np.zeros((*mk.shape, 4), dtype=np.float32)
                overlay[..., :3] = fly_colors[fly_idx]
                overlay[..., 3] = mk * 0.32
                ax.imshow(overlay)

            pts = per_fly_kp_2d[fly_idx] if per_fly_kp_2d else None
            if pts is not None and pts[cam] is not None:
                xs, ys = pts[cam][:, 0], pts[cam][:, 1]
                ax.scatter(xs, ys, s=3, c=[fly_colors[fly_idx]],
                           edgecolors='white', linewidths=0.2)

    fig.suptitle(f'frame {abs_frame}', fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def predict_bout(bout, bout_masks, video_paths, centerDetect, kp_model,
                 hybridNet, repro_tool, cfg, bout_out_dir, num_animals=2,
                 log_every=50, save_overlays_every=0):
    """Iterate every frame of a bout, writing one CSV per fly."""
    num_cams = len(video_paths)
    bbox_hw = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE // 2
    cd_input_size = int(cfg.CENTERDETECT.IMAGE_SIZE)
    # Anchor every CUDA op in this function to the reproTool's device — with
    # SAM3 living on a separate GPU, the process-wide default may point
    # elsewhere and `.cuda()` silently goes to the wrong card.
    dev = repro_tool.cameraMatrices.device
    cameraMatrices = repro_tool.cameraMatrices.to(dev)

    # Open caps once; seek to bout start once; read sequentially after.
    caps = [cv2.VideoCapture(p) for p in video_paths]
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, bout['start'])

    num_frames = bout['end'] - bout['start'] + 1
    files = []
    writers = []
    for fly_idx in range(num_animals):
        f, w = csv_writer_for_fly(bout_out_dir, fly_idx, cfg)
        files.append(f)
        writers.append(w)

    mean_rgb = np.array(cfg.DATASET.MEAN, dtype=np.float32)
    std_rgb = np.array(cfg.DATASET.STD, dtype=np.float32)

    img_size_xy = None
    overlay_dir = None
    if save_overlays_every and save_overlays_every > 0:
        overlay_dir = os.path.join(bout_out_dir, 'overlays')
        os.makedirs(overlay_dir, exist_ok=True)

    for f_local in range(num_frames):
        abs_frame = bout['start'] + f_local
        frames_bgr = read_frame_all_cams(caps)
        if frames_bgr is None:
            break
        if img_size_xy is None:
            img_size_xy = (frames_bgr.shape[2], frames_bgr.shape[1])

        # RGB uint8 + normalized tensor for CenterDetect.
        frames_rgb = frames_bgr[..., ::-1].copy()
        imgs = torch.from_numpy(frames_rgb).to(dev).float().permute(0, 3, 1, 2) / 255.
        mean_t = torch.tensor(mean_rgb, device=dev).view(3, 1, 1)
        std_t = torch.tensor(std_rgb, device=dev).view(3, 1, 1)
        imgs_norm = (imgs - mean_t) / std_t

        # Pull pre-computed SAM3 masks for this frame (organized by global fly).
        fly_slots = bout_masks.get_frame(f_local, num_animals=num_animals)
        if fly_slots is None:
            # No identity map → skip frame.
            _write_nan_row(writers, num_animals, abs_frame, cfg)
            continue

        # All-flies union (for building distractor masks per target fly).
        union_masks = np.zeros(
            (num_cams, frames_rgb.shape[1], frames_rgb.shape[2]),
            dtype=bool)
        for slot in fly_slots:
            if slot is None:
                continue
            mk = slot['masks'].numpy().astype(bool)
            valid = slot['valid'].numpy()
            for c in range(num_cams):
                if valid[c]:
                    union_masks[c] |= mk[c]

        per_fly_results = [None] * num_animals

        for fly_idx, slot in enumerate(fly_slots):
            if slot is None:
                continue
            valid = slot['valid'].numpy()
            if valid.sum() < 2:
                continue
            target_masks = slot['masks'].numpy().astype(bool)   # (C, H, W)
            distractor_masks = union_masks & ~target_masks      # (C, H, W)

            # Compute 3D center using this fly's SAM3 centroids as 2D
            # targets and triangulate.
            centroids = slot['centroids'].to(dev)               # (C, 2)
            mvals = torch.where(
                torch.from_numpy(valid).to(dev).unsqueeze(1),
                torch.ones(num_cams, 1, device=dev) * 0.9,
                torch.zeros(num_cams, 1, device=dev))
            center3D = repro_tool.reconstructPoint(
                centroids.transpose(0, 1), mvals.unsqueeze(1))
            if not torch.isfinite(center3D).all():
                continue

            centerHMs = repro_tool.reprojectPoint(
                center3D.unsqueeze(0)).int()
            centerHMs = clamp_crop_centers(centerHMs, img_size_xy, bbox_hw)

            imgs_4ch = build_4ch_crops(
                frames_rgb, centerHMs, target_masks, distractor_masks,
                bbox_hw, mean_rgb, std_rgb, device=dev)

            points3D, confs = run_kp_and_3d(
                kp_model, hybridNet, imgs_4ch,
                center3D, centerHMs, cameraMatrices)

            per_fly_results[fly_idx] = (
                points3D.cpu().numpy(), confs.cpu().numpy())

        for fly_idx in range(num_animals):
            row = [abs_frame]
            if per_fly_results[fly_idx] is None:
                row += ['NaN'] * (cfg.KEYPOINTDETECT.NUM_JOINTS * 4)
            else:
                pts, confs = per_fly_results[fly_idx]
                for pt, c in zip(pts, confs):
                    row += [float(pt[0]), float(pt[1]), float(pt[2]), float(c)]
            writers[fly_idx].writerow(row)

        if overlay_dir and f_local % save_overlays_every == 0:
            per_fly_kp_2d = []
            for fly_idx in range(num_animals):
                if per_fly_results[fly_idx] is None:
                    per_fly_kp_2d.append([None] * num_cams)
                    continue
                pts3d_np, _ = per_fly_results[fly_idx]
                pts3d = torch.from_numpy(pts3d_np).float().to(
                    cameraMatrices.device)
                pts2d = repro_tool.reprojectPoint(pts3d)   # (num_cams, K, 2)
                pts2d_np = pts2d.detach().cpu().numpy()
                per_fly_kp_2d.append(
                    [pts2d_np[c] for c in range(num_cams)])
            render_overlay(
                frames_rgb, fly_slots, per_fly_kp_2d, abs_frame,
                os.path.join(overlay_dir, f'frame_{abs_frame:06d}.png'),
                num_animals, repro_tool=repro_tool)

        if f_local % log_every == 0:
            print(f'    bout {bout["bout_idx"]} '
                  f'frame {f_local + 1}/{num_frames}', flush=True)

    for f in files:
        f.close()
    for cap in caps:
        cap.release()


def _write_nan_row(writers, num_animals, abs_frame, cfg):
    nan_row = [abs_frame] + ['NaN'] * (cfg.KEYPOINTDETECT.NUM_JOINTS * 4)
    for w in writers:
        w.writerow(nan_row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='red_data_unified')
    ap.add_argument('--session', required=True,
                    help='Session directory with CamXXX.mp4 + calibration/')
    ap.add_argument('--bouts-csv', default='courtship_bouts_unified_summary.csv',
                    help='Bout summary CSV path (relative to --session if '
                         'not absolute).')
    ap.add_argument('--out', required=True,
                    help='Output directory for per-bout CSVs.')
    ap.add_argument('--center-weights', default=DEFAULT_CENTER_WEIGHTS)
    ap.add_argument('--kp-weights', default=DEFAULT_KP_WEIGHTS)
    ap.add_argument('--hybridnet-weights', default=DEFAULT_HYBRIDNET_WEIGHTS)
    ap.add_argument('--num-animals', type=int, default=2)
    ap.add_argument('--sam3-gpu', type=int, default=1,
                    help='GPU index for SAM3 (keep JARVIS on a separate GPU).')
    ap.add_argument('--sam3-text', default='insect')
    ap.add_argument('--sam3-version', default='sam3',
                    choices=['sam3', 'sam3.1'],
                    help='SAM3 model version. sam3.1 uses the multiplex '
                         'video predictor (~2× faster with --sam3-compile) '
                         'but requires torch>=2.6 (bool-sort on CUDA) and '
                         'a working flash_attn_3 wheel (or use_fa3=False).')
    ap.add_argument('--sam3-compile', dest='sam3_compile',
                    action='store_true', default=True,
                    help='torch.compile the SAM 3.1 multiplex backbones '
                         '(default on; ignored for sam3 base).')
    ap.add_argument('--no-sam3-compile', dest='sam3_compile',
                    action='store_false',
                    help='Disable torch.compile for SAM 3.1 (useful for '
                         'single-bout dev runs — skips ~30–60s warm-up).')
    ap.add_argument('--sam3-checkpoint', default=None,
                    help='Explicit SAM3 checkpoint path. If omitted, the '
                         'checkpoint auto-downloads from HuggingFace.')
    ap.add_argument('--bouts', default=None,
                    help='Comma-separated bout_idx values to restrict to.')
    ap.add_argument('--save-masks', action='store_true',
                    help='Write per-bout SAM3 masks to sam3_masks.npz '
                         '(packed-bit, per-fly, per-cam, per-frame).')
    ap.add_argument('--reuse-masks', action='store_true',
                    help='If <bout>/sam3_masks.npz exists, load it and '
                         'skip running SAM3 for that bout.')
    ap.add_argument('--save-overlays-every', type=int, default=0,
                    help='If >0, write a 1×num_cams PNG overlay '
                         '(RGB + mask tint + KP dots) every N frames per '
                         'bout under <bout>/overlays/. 0 disables.')
    ap.add_argument('--save-clips', action='store_true',
                    help='After each bout, render one annotated MP4 per '
                         'camera under <bout>/clips/ via '
                         'jarvis.visualization.create_multi_animal_videos3D '
                         '(skeleton + SAM3 mask overlays).')
    args = ap.parse_args()

    pm = ProjectManager()
    assert pm.load(args.project), f'could not load project {args.project}'
    cfg = pm.get_cfg()
    repro_tool = get_repro_tool(cfg, None)
    assert repro_tool is not None, 'ReprojectionTool not available'

    centerDetect, kp_model, hybridNet = build_models(
        cfg, args.center_weights, args.kp_weights, args.hybridnet_weights)

    session_dir = os.path.abspath(args.session)
    session_tag = '/'.join(session_dir.rstrip('/').split('/')[-2:])

    csv_path = args.bouts_csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(session_dir, csv_path)
    bouts = parse_bouts(csv_path, session_tag)

    if args.bouts is not None:
        keep = {int(x) for x in args.bouts.split(',') if x}
        bouts = [b for b in bouts if b['bout_idx'] in keep]

    print(f'[cfg] session={session_tag} bouts={len(bouts)} '
          f'out={args.out}')

    video_paths = get_video_paths(session_dir, repro_tool)
    os.makedirs(args.out, exist_ok=True)

    # Defer SAM3 load until we actually need it — if every bout has a
    # cached NPZ and --reuse-masks is set, we can skip the predictor.
    sam3 = None
    def _get_sam3():
        nonlocal sam3
        if sam3 is None:
            sam3 = SAM3VideoTracker(
                gpu_id=args.sam3_gpu,
                text_prompt=args.sam3_text,
                sam3_version=args.sam3_version,
                compile=args.sam3_compile,
                checkpoint_path=args.sam3_checkpoint,
            )
        return sam3

    for bout in bouts:
        bout_out = os.path.join(args.out, f'bout_{bout["bout_idx"]:05d}')
        os.makedirs(bout_out, exist_ok=True)
        n = bout['end'] - bout['start'] + 1
        print(f'[bout {bout["bout_idx"]}] frames '
              f'{bout["start"]}–{bout["end"]} ({n})')

        cache_path = os.path.join(bout_out, MASKS_FILENAME)
        if args.reuse_masks and os.path.isfile(cache_path):
            t0 = time.time()
            bout_masks = LoadedBoutMasks(cache_path)
            print(f'  loaded cached masks ({os.path.getsize(cache_path)/1e6:.1f} MB) '
                  f'from {cache_path} in {time.time() - t0:.1f}s')
        else:
            t0 = time.time()
            bout_masks = _get_sam3().process_bout(
                video_paths, bout['start'], n, num_animals=args.num_animals)
            bout_masks.assign_identities(
                repro_tool, num_animals=args.num_animals)
            print(f'  SAM3+ID in {time.time() - t0:.1f}s')

            if args.save_masks:
                save_bout_masks(bout_masks, bout_out, args.num_animals)

        t0 = time.time()
        predict_bout(bout, bout_masks, video_paths, centerDetect, kp_model,
                     hybridNet, repro_tool, cfg, bout_out,
                     num_animals=args.num_animals,
                     save_overlays_every=args.save_overlays_every)
        print(f'  3D predict in {time.time() - t0:.1f}s')

        if args.save_clips:
            from jarvis.visualization.create_multi_animal_videos3D import (
                create_multi_animal_videos3D,
            )
            data_csvs = {
                f'fly{i}': os.path.join(bout_out, f'fly{i}.csv')
                for i in range(args.num_animals)
                if os.path.isfile(os.path.join(bout_out, f'fly{i}.csv'))
            }
            if data_csvs:
                t0 = time.time()
                create_multi_animal_videos3D(
                    project_name=args.project,
                    recording_path=session_dir,
                    data_csvs=data_csvs,
                    dataset_name=None,
                    frame_start=bout['start'],
                    number_frames=n,
                    video_cam_list=None,
                    output_dir=os.path.join(bout_out, 'clips'),
                    n_jobs=min(len(video_paths), 8),
                    mask_file=os.path.join(bout_out, MASKS_FILENAME)
                    if os.path.isfile(os.path.join(bout_out, MASKS_FILENAME))
                    else None,
                )
                print(f'  clips in {time.time() - t0:.1f}s')

        del bout_masks
        gc.collect()
        torch.cuda.empty_cache()

    print('[done]')


if __name__ == '__main__':
    main()
