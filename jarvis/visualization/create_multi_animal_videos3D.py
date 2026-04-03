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
from tqdm import tqdm
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
    # Extract x,y,z (remove every 4th column which is confidence)
    points3D = np.delete(data, list(range(3, data.shape[1], 4)), axis=1)
    confidences = data[:, 3::4]
    return points3D, confidences


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

    print(f"  Creating visualization: {number_frames} frames, "
          f"{len(fly_data)} flies, {sum(make_video_index)} cameras")

    for frame_num in tqdm(range(number_frames), desc="Visualizing"):
        # Read frames from all cameras
        Parallel(n_jobs=n_jobs, require='sharedmem')(
            delayed(read_images)(cap, idx, imgs_orig)
            for idx, cap in enumerate(caps)
        )

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

    # Re-encode to H.264 via ffmpeg for broad playback compatibility
    if shutil.which('ffmpeg'):
        for mp4 in os.listdir(output_dir):
            if not mp4.endswith('.mp4'):
                continue
            src = os.path.join(output_dir, mp4)
            tmp = src + '.h264.mp4'
            ret = subprocess.run(
                ['ffmpeg', '-y', '-i', src, '-c:v', 'libx264',
                 '-pix_fmt', 'yuv420p', '-loglevel', 'error', tmp],
                capture_output=True,
            )
            if ret.returncode == 0:
                os.replace(tmp, src)
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
        print("  Re-encoded videos to H.264")

    print(f"  Visualization saved to: {output_dir}")
    return output_dir
