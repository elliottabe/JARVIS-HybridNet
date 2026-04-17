"""
Multi-animal 3D visualization for JARVIS-HybridNet.

Overlays multiple animals' 3D skeletons onto camera videos with distinct
per-animal colors so identity can be visually verified.
"""

import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.reprojection import ReprojectionTool, load_reprojection_tools
from jarvis.utils.skeleton import get_skeleton
from jarvis.visualization.create_videos3D import (
    get_video_paths_and_cam_index,
    create_video_writer_and_reader,
    read_images,
)
import jarvis.visualization.visualization_utils as utils


# Distinct base colors per fly (BGR format for OpenCV)
DEFAULT_FLY_COLORS = [
    (0, 0, 255),      # fly0: red
    (255, 180, 0),     # fly1: cyan/blue
    (0, 255, 0),       # fly2: green
    (255, 0, 255),     # fly3: magenta
    (0, 255, 255),     # fly4: yellow
    (255, 128, 0),     # fly5: orange-blue
]


def load_csv_data(csv_path):
    """Load 3D keypoint data from a JARVIS CSV file.

    Returns:
        points3D: (num_frames, num_joints*3) array of x,y,z values
        confidences: (num_frames, num_joints) array of confidence values
        Returns (None, None) if the file has no valid data.
    """
    data = np.genfromtxt(csv_path, delimiter=',')
    if data.ndim == 1:
        return None, None
    # Skip header rows (contain NaN from text headers)
    if np.isnan(data[0, 0]):
        data = data[2:]
    if len(data) == 0:
        return None, None
    # Some writers prepend a frame-index column (ncols == num_joints*4 + 1).
    # Strip it so downstream slicing on 4-wide (x,y,z,conf) groups lines up.
    if data.shape[1] % 4 == 1:
        data = data[:, 1:]
    # Extract x,y,z (remove every 4th column which is confidence)
    points3D = np.delete(data, list(range(3, data.shape[1], 4)), axis=1)
    confidences = data[:, 3::4]
    return points3D, confidences


class _MaskStore:
    """Unified lookup for two on-disk SAM3 mask layouts.

    Legacy: flat NPZ with per-frame keys `f{NNNNNN}_{fly_id}` plus a
        `_meta` array of `[H, W, num_cameras]`.
    Packed-ACF (new, from tools/predict3D_multianimal.py): arrays
        `packed[A, C, F, H, W_pack]`, `valid[A, C, F]`, `centroids`,
        `shape=[H, W]`.
    """

    def __init__(self, data):
        self.data = data
        if 'packed' in data.files:
            self.format = 'acf'
            self.packed = data['packed']
            self.valid = data['valid']
            shape = data['shape']
            self.H, self.W = int(shape[0]), int(shape[1])
            self.num_animals = int(self.packed.shape[0])
            self.num_cameras = int(self.packed.shape[1])
            self.num_frames = int(self.packed.shape[2])
        else:
            self.format = 'legacy'
            meta_arr = data['_meta']
            self.H = int(meta_arr[0])
            self.W = int(meta_arr[1])
            self.num_cameras = int(meta_arr[2])

    @property
    def meta(self):
        return (self.H, self.W, self.num_cameras)

    def get_masks_for(self, frame_num, fly_id=None, fly_idx=None):
        """Return (num_cameras, H, W) bool array, or None."""
        if self.format == 'acf':
            if fly_idx is None or fly_idx >= self.num_animals:
                return None
            if frame_num >= self.num_frames:
                return None
            out = np.zeros(
                (self.num_cameras, self.H, self.W), dtype=bool)
            for cam in range(self.num_cameras):
                if not bool(self.valid[fly_idx, cam, frame_num]):
                    continue
                row = self.packed[fly_idx, cam, frame_num]
                unpacked = np.unpackbits(
                    row, axis=1, bitorder='big')[:, :self.W]
                out[cam] = unpacked.astype(bool)
            return out
        # legacy
        if fly_id is None:
            return None
        key = f'f{frame_num:06d}_{fly_id}'
        if key not in self.data:
            return None
        packed = self.data[key]
        flat = np.unpackbits(packed)
        total = self.num_cameras * self.H * self.W
        return flat[:total].reshape(
            self.num_cameras, self.H, self.W).astype(bool)


def load_mask_data(mask_file):
    """Load saved SAM3 masks from an .npz file.

    Returns:
        (_MaskStore, (H, W, num_cameras)) or (None, None) if the file is
        missing. The store auto-detects the legacy per-frame-key layout
        and the newer packed-(A,C,F) layout.
    """
    if mask_file is None or not os.path.exists(mask_file):
        return None, None
    data = np.load(mask_file, allow_pickle=True)
    store = _MaskStore(data)
    return store, store.meta


def overlay_mask(img, mask, color, alpha=0.3):
    """Overlay a binary mask as a semi-transparent colored region.

    Args:
        img: (H, W, 3) uint8 BGR image (modified in-place)
        mask: (H, W) bool array
        color: (B, G, R) tuple
        alpha: transparency (0=invisible, 1=opaque)
    """
    if mask is None or not mask.any():
        return
    overlay = img.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    # Draw mask contour for crisp boundary
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 1)


def create_multi_animal_videos3D(
    project_name,
    recording_path,
    data_csvs,
    dataset_name,
    frame_start=0,
    number_frames=-1,
    video_cam_list=None,
    fly_colors=None,
    output_dir=None,
    n_jobs=12,
    mask_file=None,
):
    """
    Create visualization videos with multiple animals' skeletons overlaid.

    Each animal is drawn in a distinct color with its fly_id label near
    the head (Antenna_Base keypoint).

    Args:
        project_name: JARVIS project name
        recording_path: path to video folder
        data_csvs: dict mapping fly_id -> path to that fly's data3D CSV
        dataset_name: calibration folder path
        frame_start: starting frame number
        number_frames: frames to visualize (-1 for all available)
        video_cam_list: list of camera names to create videos for (None = all)
        fly_colors: optional dict mapping fly_id -> (B, G, R) color tuple
        output_dir: output directory (auto-generated if None)
        n_jobs: parallel jobs for frame reading
        mask_file: path to .npz file with SAM3 masks (None to skip)

    Returns:
        output_dir path, or None on failure
    """
    # Load project
    project = ProjectManager()
    if not project.load(project_name):
        print(f"Could not load project: {project_name}!")
        return None
    cfg = project.cfg

    # Get reprojection tool
    reproTools = load_reprojection_tools(cfg, device='cpu')
    if dataset_name is not None and os.path.isdir(dataset_name):
        import json
        dataset_dir = os.path.join(
            cfg.PARENT_DIR, cfg.DATASET.DATASET_ROOT_DIR, cfg.DATASET.DATASET_3D
        )
        with open(os.path.join(dataset_dir, 'annotations', 'instances_val.json')) as f:
            data = json.load(f)
        calibPaths = {}
        calibParams = list(data['calibrations'].keys())[0]
        for cam in data['calibrations'][calibParams]:
            calibPaths[cam] = data['calibrations'][calibParams][cam].split('/')[-1]
        reproTool = ReprojectionTool(dataset_name, calibPaths, 'cpu')
    elif len(reproTools) >= 1:
        reproTool = reproTools[list(reproTools.keys())[0]]
    else:
        print("Could not load reprojection tool!")
        return None

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(
            project.parent_dir, cfg.PROJECTS_ROOT_PATH, project_name,
            'visualization',
            f'Videos_3D_multi_{time.strftime("%Y%m%d-%H%M%S")}'
        )
    os.makedirs(output_dir, exist_ok=True)

    # Get skeleton bone connections
    _, line_idxs = get_skeleton(cfg)

    # Load per-fly data
    fly_ids = sorted(data_csvs.keys())
    fly_data = {}
    for fly_id in fly_ids:
        pts, confs = load_csv_data(data_csvs[fly_id])
        if pts is not None:
            fly_data[fly_id] = {'points3D': pts, 'confidences': confs}
            print(f"  {fly_id}: {pts.shape[0]} frames loaded")
        else:
            print(f"  {fly_id}: no valid data, skipping")

    # Load mask data if available
    mask_data, mask_meta = load_mask_data(mask_file)
    if mask_data is not None:
        print(f"  Loaded SAM3 masks from {mask_file}")

    if not fly_data:
        print("No valid fly data found!")
        return None

    # Determine number of frames from the data
    data_num_frames = min(d['points3D'].shape[0] for d in fly_data.values())
    if number_frames == -1 or number_frames > data_num_frames:
        number_frames = data_num_frames

    # Assign colors
    if fly_colors is None:
        fly_colors = {}
    for i, fly_id in enumerate(fly_ids):
        if fly_id not in fly_colors:
            fly_colors[fly_id] = DEFAULT_FLY_COLORS[i % len(DEFAULT_FLY_COLORS)]

    # Setup video readers/writers
    if video_cam_list is None or len(video_cam_list) == 0:
        videos = os.listdir(recording_path)
        video_cam_list = [
            v.split('.')[0] for v in videos
            if v.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

    # Use a simple params-like object for the existing helper functions
    class _Params:
        pass
    params = _Params()
    params.recording_path = recording_path
    params.video_cam_list = video_cam_list
    params.frame_start = frame_start
    params.output_dir = output_dir

    video_paths, make_video_index = get_video_paths_and_cam_index(
        recording_path, reproTool, video_cam_list
    )
    caps, outs, img_size = create_video_writer_and_reader(
        params, reproTool, video_paths, make_video_index
    )

    # Pre-allocate image buffer
    imgs_orig = np.zeros(
        (len(caps), img_size[1], img_size[0], 3)
    ).astype(np.uint8)

    viz_tag = os.path.basename(output_dir.rstrip('/'))
    print(f"  [viz {viz_tag}] starting: {number_frames} frames, "
          f"{len(fly_data)} flies, {sum(make_video_index)} cameras",
          flush=True)
    _viz_t0 = time.time()
    _viz_log_every = max(50, number_frames // 20)

    for frame_num in range(number_frames):
        if frame_num and (frame_num % _viz_log_every == 0):
            elapsed = time.time() - _viz_t0
            fps = frame_num / elapsed if elapsed > 0 else 0
            eta = (number_frames - frame_num) / fps if fps > 0 else 0
            pct = 100 * frame_num / number_frames
            print(f"  [viz {viz_tag}] {frame_num}/{number_frames} "
                  f"({pct:.0f}%) | {fps:.1f} fps | ETA {eta:.0f}s",
                  flush=True)
        # Read frames from all cameras
        Parallel(n_jobs=n_jobs, require='sharedmem')(
            delayed(read_images)(cap, idx, imgs_orig)
            for idx, cap in enumerate(caps)
        )

        # Draw SAM3 mask overlays (before skeletons so they appear on top)
        if mask_data is not None:
            H_mask, W_mask, n_cams_mask = mask_meta
            for i, fly_id in enumerate(fly_ids):
                masks_bool = mask_data.get_masks_for(
                    frame_num, fly_id=fly_id, fly_idx=i)
                if masks_bool is None:
                    continue
                color = fly_colors.get(fly_id, (128, 128, 128))
                for cam_idx in range(len(outs)):
                    if not make_video_index[cam_idx]:
                        continue
                    if cam_idx < n_cams_mask:
                        cam_mask = masks_bool[cam_idx]
                        if (cam_mask.shape[0] != img_size[1]
                                or cam_mask.shape[1] != img_size[0]):
                            cam_mask = cv2.resize(
                                cam_mask.astype(np.uint8),
                                (img_size[0], img_size[1]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        overlay_mask(imgs_orig[cam_idx], cam_mask, color)

        # Draw each fly's skeleton
        for fly_id in fly_ids:
            if fly_id not in fly_data:
                continue

            pts_row = fly_data[fly_id]['points3D'][frame_num]

            # Skip if all NaN (no detection this frame)
            if np.all(np.isnan(pts_row)):
                continue

            points3D_net = torch.from_numpy(pts_row.reshape(-1, 3)).float()
            color = fly_colors[fly_id]

            # Reproject 3D -> 2D for all cameras
            points2D = reproTool.reprojectPoint(points3D_net).numpy()
            points2D = np.array(points2D)

            # Draw on each camera view
            for cam_idx in range(len(outs)):
                if not make_video_index[cam_idx]:
                    continue

                # Draw skeleton bones
                for line in line_idxs:
                    utils.draw_line(
                        imgs_orig[cam_idx], line,
                        points2D[:, cam_idx], img_size, color
                    )

                # Draw joint points
                for j, points in enumerate(points2D):
                    utils.draw_point(
                        imgs_orig[cam_idx], points[cam_idx],
                        img_size, color
                    )

                # Draw fly_id label near the head (keypoint 0 = Antenna_Base)
                head_pt = points2D[0, cam_idx]
                if (not np.isnan(head_pt).any()
                        and 0 < head_pt[0] < img_size[0]
                        and 0 < head_pt[1] < img_size[1]):
                    label_pos = (int(head_pt[0]) + 5, int(head_pt[1]) - 8)
                    cv2.putText(
                        imgs_orig[cam_idx], fly_id, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,
                        thickness=1, lineType=cv2.LINE_AA
                    )

        # Write annotated frames
        for cam_idx, out in enumerate(outs):
            if make_video_index[cam_idx]:
                out.write(imgs_orig[cam_idx])

    # Cleanup
    for cam_idx, out in enumerate(outs):
        if make_video_index[cam_idx]:
            out.release()
    for cap in caps:
        cap.release()

    _viz_total = time.time() - _viz_t0
    _viz_fps = number_frames / _viz_total if _viz_total > 0 else 0
    print(f"  [viz {viz_tag}] done: {number_frames} frames in "
          f"{_viz_total:.0f}s ({_viz_fps:.1f} fps)", flush=True)

    # cv2 writes avc1/H.264 directly but leaves the moov atom at the end of
    # the file, which VSCode's preview can't handle. Stream-copy with
    # -movflags +faststart so the moov atom moves to the front. This is a
    # near-instant remux (no re-encode).
    if shutil.which('ffmpeg'):
        for mp4 in os.listdir(output_dir):
            if not mp4.endswith('.mp4'):
                continue
            src = os.path.join(output_dir, mp4)
            tmp = src + '.faststart.mp4'
            ret = subprocess.run(
                ['ffmpeg', '-y', '-i', src,
                 '-c', 'copy', '-movflags', '+faststart',
                 '-loglevel', 'error', tmp],
                capture_output=True,
            )
            if ret.returncode == 0:
                os.replace(tmp, src)
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)

    print(f"  Visualization saved to: {output_dir}")
    return output_dir
