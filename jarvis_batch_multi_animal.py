#!/usr/bin/env python3
"""
Multi-Animal Batch 3D Prediction Script for JARVIS-HybridNet

Extends the standard batch prediction pipeline to detect and track multiple
animals (e.g., male + female flies during courtship). Produces separate CSV
files for each animal with persistent identity assignment.

Usage:
    python jarvis_batch_multi_animal.py --project merge_courtship_V3 \
        --config batch_config.json --num_animals 2

    Or with individual folders:
    python jarvis_batch_multi_animal.py --project merge_courtship_V3 \
        --video_folder /path/to/videos --calib_folder /path/to/calib \
        --num_animals 2

    Predict only specific bouts (from a CSV with start_frame/end_frame):
    python jarvis_batch_multi_animal.py --project merge_courtship_V3 \
        --video_folder /path/to/videos --calib_folder /path/to/calib \
        --bouts_csv /path/to/courtship_bouts_summary.csv \
        --num_animals 2

    Predict only specific frame ranges (from a text file):
    python jarvis_batch_multi_animal.py --project merge_courtship_V3 \
        --video_folder /path/to/videos --calib_folder /path/to/calib \
        --frame_file /path/to/frames.txt \
        --num_animals 2

    Resume an interrupted run:
    python jarvis_batch_multi_animal.py --project merge_courtship_V3 \
        --video_folder /path/to/videos --calib_folder /path/to/calib \
        --bouts_csv /path/to/courtship_bouts_summary.csv \
        --num_animals 2 --resume /path/to/Predictions_3D_20260402-123456

    Multi-GPU (2 GPUs locally):
    python jarvis_batch_multi_animal.py --project merge_courtship_V3 \
        --video_folder /path/to/videos --calib_folder /path/to/calib \
        --bouts_csv /path/to/courtship_bouts_summary.csv \
        --num_animals 2 --gpus 0 1

    Multi-GPU on cluster (8 GPUs, full video):
    python jarvis_batch_multi_animal.py --project merge_courtship_V3 \
        --video_folder /path/to/videos --calib_folder /path/to/calib \
        --num_animals 2 --gpus 0 1 2 3 4 5 6 7

Bout CSV format (columns: fly_id, bout_idx, start_frame, end_frame):
    fly_id is the recording/session identifier (not the individual animal).
    bout_idx,start_frame,end_frame are used to define frame ranges.
    Example:
        fly_id,bout_idx,start_frame,end_frame
        Session0/2025_10_20_13_20_04,1,14045,14557
        Session0/2025_10_20_13_20_04,2,88111,88678

Frame file format (plain text, one range per line):
    # Comments start with '#'
    0 1000
    2500 4000

Config file format: same as jarvis_batch_frame_range.py

Output per job:
    Predictions_3D_<timestamp>/
        data3D_fly0.csv          # Largest animal (female for 2-fly courtship)
        data3D_fly1.csv          # Smallest animal (male for 2-fly courtship)
        ...
        tracking_info.json       # Tracking statistics (swap counts, body sizes)
        info.yaml                # Standard JARVIS metadata
"""

import argparse
import csv
import gc
import glob
import itertools
import json
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from ruamel.yaml import YAML
from tqdm import tqdm

from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis3D_multi import JarvisMultiAnimalPredictor3D
from jarvis.prediction.tracker import MultiAnimalTracker
from jarvis.utils.reprojection import ReprojectionTool, load_reprojection_tools
from jarvis.visualization.create_multi_animal_videos3D import create_multi_animal_videos3D


class CLIColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def gpu_print(*args, gpu_id=None, **kwargs):
    """Print with optional [GPU X] prefix; strip ANSI codes on non-tty."""
    msg = ' '.join(str(a) for a in args)
    if gpu_id is not None:
        msg = f"[GPU {gpu_id}] {msg}"
    if not sys.stdout.isatty():
        msg = _ANSI_RE.sub('', msg)
    kwargs.setdefault('flush', True)
    print(msg, **kwargs)


class ProgressLogger:
    """Lightweight progress reporter for multi-GPU and non-tty environments.

    When gpu_id is None and stdout is a tty, delegates to tqdm unchanged.
    Otherwise prints periodic summary lines prefixed with [GPU X].
    """

    def __init__(self, iterable, desc="", total=None, gpu_id=None,
                 report_interval=30):
        self._iterable = iterable
        self._desc = desc
        self._total = total if total is not None else (
            len(iterable) if hasattr(iterable, '__len__') else None)
        self._gpu_id = gpu_id
        self._interval = report_interval
        self._use_tqdm = (gpu_id is None and sys.stdout.isatty())

    def __iter__(self):
        if self._use_tqdm:
            yield from tqdm(self._iterable, desc=self._desc,
                            total=self._total)
            return

        t0 = time.time()
        last_report = t0
        count = 0
        gpu_print(f"{self._desc}: starting ({self._total} frames)",
                  gpu_id=self._gpu_id)

        for item in self._iterable:
            yield item
            count += 1
            now = time.time()
            if now - last_report >= self._interval:
                elapsed = now - t0
                fps = count / elapsed if elapsed > 0 else 0
                pct = (count / self._total * 100) if self._total else 0
                eta = (self._total - count) / fps if fps > 0 and self._total else 0
                gpu_print(f"{self._desc}: {count}/{self._total} "
                          f"({pct:.0f}%) | {fps:.1f} fps | "
                          f"ETA {eta:.0f}s",
                          gpu_id=self._gpu_id)
                last_report = now

        elapsed = time.time() - t0
        fps = count / elapsed if elapsed > 0 else 0
        gpu_print(f"{self._desc}: done ({count} frames in {elapsed:.0f}s, "
                  f"{fps:.1f} fps)",
                  gpu_id=self._gpu_id)


def parse_frame_file(frame_file_path):
    """Parse a text file containing frame ranges (one per line)."""
    frame_ranges = []
    with open(frame_file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r'[\s,]+', line)
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    frame_ranges.append([start, end])
                except ValueError:
                    print(f"{CLIColors.WARNING}Warning: skipping invalid line "
                          f"{line_num}: '{line}'{CLIColors.ENDC}")
            else:
                print(f"{CLIColors.WARNING}Warning: skipping invalid line "
                      f"{line_num}: '{line}'{CLIColors.ENDC}")
    frame_ranges.sort(key=lambda x: x[0])
    return frame_ranges


def parse_bouts_csv(bouts_csv_path):
    """
    Parse a bouts summary CSV file into frame ranges.

    Expected columns: fly_id, bout_idx, start_frame, end_frame
    Where fly_id is the recording/session identifier.

    Args:
        bouts_csv_path: Path to the bouts CSV file.

    Returns:
        List of [start_frame, end_frame] pairs, sorted by start_frame.
    """
    import pandas as pd
    df = pd.read_csv(bouts_csv_path)

    required = ['start_frame', 'end_frame']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"{CLIColors.FAIL}Error: bouts CSV missing columns: "
              f"{missing}. Found: {list(df.columns)}{CLIColors.ENDC}")
        return []

    frame_ranges = []
    for _, row in df.iterrows():
        start = int(row['start_frame'])
        end = int(row['end_frame'])
        frame_ranges.append([start, end])

    frame_ranges.sort(key=lambda x: x[0])
    print(f"Loaded {len(frame_ranges)} bouts from {bouts_csv_path}")
    total_predicted = sum(e - s + 1 for s, e in frame_ranges)
    print(f"  Total frames to predict: {total_predicted}")
    return frame_ranges


def build_predict_set(frame_ranges, total_frames):
    """Build a set of frame numbers that should be predicted."""
    predict_set = set()
    resolved_ranges = []
    for start, end in frame_ranges:
        if end == -1:
            end = total_frames - 1
        end = min(end, total_frames - 1)
        resolved_ranges.append((start, end))
        for f in range(start, end + 1):
            predict_set.add(f)

    global_start = min(s for s, _ in resolved_ranges)
    global_end = max(e for _, e in resolved_ranges)
    return predict_set, global_start, global_end


def get_repro_tool(cfg, calib_folder, cameras_to_use=None, device="cuda"):
    """Load reprojection tool from a calibration folder."""
    reproTools = load_reprojection_tools(
        cfg, cameras_to_use=cameras_to_use, device=device
    )

    if calib_folder is not None and os.path.isdir(calib_folder):
        dataset_dir = os.path.join(
            cfg.PARENT_DIR,
            cfg.DATASET.DATASET_ROOT_DIR,
            cfg.DATASET.DATASET_3D,
        )
        dataset_json_path = os.path.join(
            dataset_dir, "annotations", "instances_val.json"
        )

        with open(dataset_json_path) as dataset_json:
            data = json.load(dataset_json)

        calibPaths = {}
        calibParams = list(data["calibrations"].keys())[0]
        for cam in data["calibrations"][calibParams]:
            if cameras_to_use is None or cam in cameras_to_use:
                calibPaths[cam] = (
                    data["calibrations"][calibParams][cam].split("/")[-1]
                )

        reproTool = ReprojectionTool(calib_folder, calibPaths, device)
    elif len(reproTools) == 1:
        reproTool = reproTools[list(reproTools.keys())[0]]
    elif len(reproTools) > 1:
        reproTool = reproTools[list(reproTools.keys())[0]]
    else:
        print(f"{CLIColors.FAIL}Could not load reprojection Tool{CLIColors.ENDC}")
        return None

    return reproTool


def convert2jarviscalib(input_folder, output_folder):
    """Convert calibration files to JARVIS format if needed."""
    cam_names = []
    for file in glob.glob(input_folder + "/*.yaml"):
        file_name = file.split("/")
        cam_names.append(file_name[-1].split(".")[0])
    cam_names.sort()

    for idx in range(len(cam_names)):
        input_file_name = os.path.join(input_folder, f"{cam_names[idx]}.yaml")
        fs = cv2.FileStorage(input_file_name, cv2.FILE_STORAGE_READ)

        intrinsic_node = fs.getNode("intrinsicMatrix")
        if not intrinsic_node.empty():
            import shutil
            output_filename = os.path.join(
                output_folder, f"{cam_names[idx]}.yaml"
            )
            shutil.copy(input_file_name, output_filename)
            continue

        intrinsicMatrix = fs.getNode("camera_matrix").mat()
        if intrinsicMatrix is not None:
            intrinsicMatrix = intrinsicMatrix.T
        distortionCoefficients = fs.getNode("distortion_coefficients").mat()
        if distortionCoefficients is not None:
            distortionCoefficients = distortionCoefficients.T
        R = fs.getNode("rc_ext").mat()
        if R is not None:
            R = R.T
        T = fs.getNode("tc_ext").mat()

        output_filename = os.path.join(output_folder, f"{cam_names[idx]}.yaml")
        s = cv2.FileStorage(output_filename, cv2.FileStorage_WRITE)
        s.write("intrinsicMatrix", intrinsicMatrix)
        s.write("distortionCoefficients", distortionCoefficients)
        s.write("R", R)
        s.write("T", T)
        s.release()
        fs.release()


def get_video_paths(recording_path, reproTool):
    """Get video paths matching camera names."""
    videos = os.listdir(recording_path)
    video_paths = []
    for i, camera in enumerate(reproTool.cameras):
        found = False
        for video in videos:
            if camera == video.split(".")[0]:
                video_paths.append(os.path.join(recording_path, video))
                found = True
                break
        if not found:
            raise ValueError(f"Missing Recording for camera {camera}")
    return video_paths


def _viz_bout_worker(
    project_name, recording_path, calib_folder, bout_csvs,
    frame_start, number_frames, viz_cameras, mask_file, output_dir,
):
    """Top-level worker (picklable) that runs per-bout viz in a subprocess."""
    # Import inside the worker so the parent process doesn't pay the cost.
    from jarvis.visualization.create_multi_animal_videos3D import (
        create_multi_animal_videos3D,
    )
    create_multi_animal_videos3D(
        project_name=project_name,
        recording_path=recording_path,
        data_csvs=bout_csvs,
        dataset_name=calib_folder,
        frame_start=frame_start,
        number_frames=number_frames,
        video_cam_list=viz_cameras,
        mask_file=mask_file,
        output_dir=output_dir,
    )
    return output_dir


def create_video_reader(video_paths, frame_start=0):
    """Create video capture objects."""
    caps = []
    img_size = [0, 0]
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        img_size_new = [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        assert img_size == [0, 0] or img_size == img_size_new, \
            "All videos need to have the same resolution"
        img_size = img_size_new
        caps.append(cap)
    return caps, img_size


def seek(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


def read_images(cap, slice_idx, imgs):
    ret, img = cap.read()
    if ret and img is not None:
        imgs[slice_idx] = img.astype(np.uint8)


def create_header(writer, cfg):
    """Create CSV header with keypoint names."""
    joints = list(
        itertools.chain.from_iterable(
            itertools.repeat(x, 4) for x in cfg.KEYPOINT_NAMES
        )
    )
    coords = ["x", "y", "z", "confidence"] * len(cfg.KEYPOINT_NAMES)
    writer.writerow(joints)
    writer.writerow(coords)


def create_info_file(output_dir, recording_path, dataset_name,
                     frame_start, number_frames, num_animals):
    """Create info.yaml file with prediction metadata."""
    with open(os.path.join(output_dir, "info.yaml"), "w") as file:
        yaml = YAML()
        yaml.dump(
            {
                "recording_path": recording_path,
                "dataset_name": dataset_name,
                "frame_start": frame_start,
                "number_frames": number_frames,
                "num_animals": num_animals,
            },
            file,
        )


def run_prediction(
    project_name,
    video_folder,
    calib_folder,
    num_animals=2,
    suppression_radius=15,
    mask_scale=1.5,
    frame_start=0,
    frame_end=-1,
    frame_ranges=None,
    cameras_to_use=None,
    weights_center_detect="latest",
    weights_hybridnet="latest",
    trt_mode="off",
    n_jobs=17,
    use_sam3_mask=False,
    sam3_device="cuda",
    sam3_text_prompt="fly",
    sam3_constrain_keypoints=False,
    sam3_chunk_size=1000,
    resume_dir=None,
    output_name=None,
    output_dir=None,
    gpu_id=None,
    sam3_init_lock=None,
    run_viz=False,
    viz_cameras=None,
    viz_max_workers=2,
    min_animal_separation_mm=0.0,
):
    """
    Run multi-animal 3D prediction on a single video folder.

    Args:
        output_dir: override output directory (used by multi-GPU dispatcher)
        gpu_id: CUDA device index (used by multi-GPU dispatcher)

    Returns:
        Dict with output info, or None on failure.
    """
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    gpu_print(f"\n{CLIColors.HEADER}{'=' * 60}{CLIColors.ENDC}",
              gpu_id=gpu_id)
    gpu_print(f"{CLIColors.OKBLUE}Processing: {video_folder}{CLIColors.ENDC}",
              gpu_id=gpu_id)
    gpu_print(f"{CLIColors.OKCYAN}Calibration: {calib_folder}{CLIColors.ENDC}",
              gpu_id=gpu_id)
    gpu_print(f"{CLIColors.OKCYAN}Num animals: {num_animals}{CLIColors.ENDC}",
              gpu_id=gpu_id)
    gpu_print(f"{CLIColors.HEADER}{'=' * 60}{CLIColors.ENDC}\n",
              gpu_id=gpu_id)

    # Load project
    project = ProjectManager()
    if not project.load(project_name):
        gpu_print(f"{CLIColors.FAIL}Could not load project: "
                  f"{project_name}!{CLIColors.ENDC}", gpu_id=gpu_id)
        return None
    cfg = project.cfg

    if cameras_to_use is not None:
        cfg.HYBRIDNET.NUM_CAMERAS = len(cameras_to_use)

    # Check if calibration needs conversion
    jarvis_calib_folder = os.path.join(
        os.path.dirname(calib_folder),
        os.path.basename(calib_folder) + "_jarvis"
    )
    calib_files = glob.glob(os.path.join(calib_folder, "*.yaml"))
    if calib_files:
        fs = cv2.FileStorage(calib_files[0], cv2.FILE_STORAGE_READ)
        intrinsic_node = fs.getNode("intrinsicMatrix")
        projection_node = fs.getNode("projectionMatrix")
        fs.release()

        if intrinsic_node.empty() and projection_node.empty():
            if not os.path.isdir(jarvis_calib_folder):
                gpu_print(f"{CLIColors.WARNING}Converting calibration files to "
                          f"JARVIS format...{CLIColors.ENDC}",
                          gpu_id=gpu_id)
                os.makedirs(jarvis_calib_folder, exist_ok=True)
                convert2jarviscalib(calib_folder, jarvis_calib_folder)
            calib_folder = jarvis_calib_folder

    # Initialize multi-animal predictor
    # SAM3 masking is handled externally via SAM3StreamingTracker (video
    # propagation), not inside the predictor. The predictor receives
    # precomputed_masks per frame when SAM3 is enabled.
    tracker_cfg = getattr(cfg, 'TRACKER', {}) or {}
    def _tg(key, default):
        if isinstance(tracker_cfg, dict):
            return tracker_cfg.get(key, default)
        return getattr(tracker_cfg, key, default)

    # Identity-collapse guard threshold. When > 0, two tracks whose
    # triangulated 3D centers are within this many millimetres are
    # collapsed down to one in both multi_peak and the tracker, so the
    # downstream CSV will never contain two tracks centered on the same
    # physical animal. Default 0 keeps existing single-species behaviour.
    # Caller-supplied value takes precedence; otherwise read from project
    # config under TRACKER.MIN_ANIMAL_SEPARATION_MM. 0 keeps legacy behaviour.
    if not min_animal_separation_mm:
        min_animal_separation_mm = float(
            _tg('MIN_ANIMAL_SEPARATION_MM', 0.0)
        )
    else:
        min_animal_separation_mm = float(min_animal_separation_mm)

    jarvisPredictor = JarvisMultiAnimalPredictor3D(
        cfg,
        num_animals=num_animals,
        suppression_radius=_tg('SUPPRESSION_RADIUS', suppression_radius),
        confidence_threshold=_tg('CONFIDENCE_THRESHOLD', 0.5),
        mask_scale=mask_scale,
        weights_center_detect=weights_center_detect,
        weights_hybridnet=weights_hybridnet,
        trt_mode=trt_mode,
        use_sam3_mask=False,  # SAM3 handled externally via streaming tracker
        sam3_constrain_keypoints=sam3_constrain_keypoints,
        multi_peak_trained=True,  # CenterDetect was trained on dual-fly heatmaps
        min_animal_separation_mm=min_animal_separation_mm,
    )

    # Initialize identity tracker
    tracker = MultiAnimalTracker(
        keypoint_names=cfg.KEYPOINT_NAMES,
        num_animals=num_animals,
        disable_swap_check=use_sam3_mask,
        max_jump_mm=_tg('MAX_JUMP_MM', 5.0),
        ema_alpha=_tg('EMA_ALPHA', 0.05),
        swap_check_frames=_tg('SWAP_CHECK_FRAMES', 50),
        velocity_alpha=_tg('VELOCITY_ALPHA', 0.5),
        cost_size_weight=_tg('COST_SIZE_WEIGHT', 0.5),
        disable_velocity_pred=_tg('DISABLE_VELOCITY_PRED', False),
        min_animal_separation_mm=min_animal_separation_mm,
    )

    # Load reprojection tool
    reproTool = get_repro_tool(cfg, calib_folder, cameras_to_use=cameras_to_use)
    if reproTool is None:
        return None

    # Setup output directory — save alongside the recording videos
    pred_base = video_folder
    if output_dir is not None:
        # Multi-GPU worker: directory provided by dispatcher
        pass
    elif resume_dir is not None and os.path.isdir(resume_dir):
        output_dir = resume_dir
        gpu_print(f"{CLIColors.OKCYAN}Resuming from: {output_dir}{CLIColors.ENDC}",
                  gpu_id=gpu_id)
    elif output_name:
        # Use provided name (e.g. SLURM job ID) — auto-resume if it already exists
        output_dir = os.path.join(pred_base, f"Predictions_3D_{output_name}")
        if os.path.isdir(output_dir):
            resume_dir = output_dir
            gpu_print(f"{CLIColors.OKCYAN}Output dir exists, auto-resuming: "
                      f"{output_dir}{CLIColors.ENDC}", gpu_id=gpu_id)
    else:
        output_dir = os.path.join(
            pred_base,
            f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}',
        )
    os.makedirs(output_dir, exist_ok=True)
    gpu_print(f"{CLIColors.OKGREEN}Output directory: {output_dir}{CLIColors.ENDC}",
              gpu_id=gpu_id)

    # Get video paths
    video_paths = get_video_paths(video_folder, reproTool)
    caps, img_size = create_video_reader(video_paths, 0)

    # Determine what to predict
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_ranges is not None and len(frame_ranges) > 0:
        # Bout mode: process each bout independently (seek to each bout)
        resolved_bouts = []
        for start, end in frame_ranges:
            if end == -1:
                end = total_frames - 1
            end = min(end, total_frames - 1)
            resolved_bouts.append((start, end))
        total_bout_frames = sum(e - s + 1 for s, e in resolved_bouts)
        gpu_print(f"Bout mode: {len(resolved_bouts)} bouts, "
                  f"{total_bout_frames} frames to predict", gpu_id=gpu_id)
    else:
        # Contiguous mode: single range
        if frame_end == -1:
            frame_end = total_frames - 1
        actual_start = frame_start
        actual_end = min(frame_end, total_frames - 1)
        resolved_bouts = [(actual_start, actual_end)]
        total_bout_frames = actual_end - actual_start + 1
        gpu_print(f"Processing frames {actual_start} to {actual_end} "
                  f"({total_bout_frames} frames)", gpu_id=gpu_id)

    # Create info file
    if resume_dir is None:
        create_info_file(
            output_dir, video_folder, calib_folder,
            resolved_bouts[0][0], total_bout_frames, num_animals
        )

    # Determine resume state: count completed frames from existing CSVs
    frames_completed = 0
    has_header = (len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS)
    header_rows = 2 if has_header else 0

    if resume_dir is not None:
        fly_ids = tracker.fly_ids
        # Count data rows in the first fly's CSV to determine progress
        first_csv = os.path.join(output_dir, f'data3D_{fly_ids[0]}.csv')
        if os.path.exists(first_csv):
            with open(first_csv, 'r') as f:
                total_lines = sum(1 for _ in f)
            frames_completed = max(0, total_lines - header_rows)
            gpu_print(f"{CLIColors.OKCYAN}Found {frames_completed} frames "
                      f"already completed{CLIColors.ENDC}", gpu_id=gpu_id)

    # Figure out which bouts to skip and where to resume within a bout
    bouts_to_skip = 0
    resume_frame_offset = 0
    if frames_completed > 0:
        cumulative = 0
        for bout_start, bout_end in resolved_bouts:
            bout_len = bout_end - bout_start + 1
            if cumulative + bout_len <= frames_completed:
                cumulative += bout_len
                bouts_to_skip += 1
            else:
                resume_frame_offset = frames_completed - cumulative
                break
        if bouts_to_skip > 0:
            gpu_print(f"{CLIColors.OKCYAN}Skipping {bouts_to_skip} completed "
                      f"bout(s){CLIColors.ENDC}", gpu_id=gpu_id)
        if resume_frame_offset > 0:
            gpu_print(f"{CLIColors.OKCYAN}Resuming bout {bouts_to_skip + 1} "
                      f"from frame offset {resume_frame_offset}{CLIColors.ENDC}",
                      gpu_id=gpu_id)

    # Open per-animal CSV files
    fly_ids = tracker.fly_ids
    csvfiles = {}
    writers = {}
    file_mode = 'a' if frames_completed > 0 else 'w'
    for fly_id in fly_ids:
        fpath = os.path.join(output_dir, f'data3D_{fly_id}.csv')
        csvfiles[fly_id] = open(fpath, file_mode, newline='')
        writers[fly_id] = csv.writer(
            csvfiles[fly_id], delimiter=',',
            quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        if file_mode == 'w' and has_header:
            create_header(writers[fly_id], cfg)

    # NaN row for frames with no detection
    nan_row = ["NaN"] * (cfg.KEYPOINTDETECT.NUM_JOINTS * 4)

    # Pre-allocate image buffer
    imgs_orig = np.zeros(
        (len(caps), img_size[1], img_size[0], 3)
    ).astype(np.uint8)

    camera_matrices_cuda = reproTool.cameraMatrices.cuda()

    # Initialize SAM3 streaming tracker if requested
    # When running multi-GPU, a lock serializes SAM3 full-model loading
    # to prevent simultaneous peak memory usage from OOM-killing workers.
    _sam3_lock_held = False
    sam3_tracker = None
    if use_sam3_mask:
        sam3_gpu_id = gpu_id if gpu_id is not None else 0
        if sam3_gpu_id == 0 and ':' in sam3_device:
            sam3_gpu_id = int(sam3_device.split(':')[1])
        from jarvis.prediction.sam3_video_tracker import SAM3LowLatencyTracker
        if sam3_init_lock is not None:
            sam3_init_lock.acquire()
            _sam3_lock_held = True
            gpu_print("Acquired SAM3 init lock", gpu_id=gpu_id)
        sam3_tracker = SAM3LowLatencyTracker(
            gpu_id=sam3_gpu_id,
            text_prompt=sam3_text_prompt,
        )

    # Per-bout viz pool: each bout's drawing/encoding runs in a subprocess
    # while the next bout's GPU inference proceeds. CPU-bound, so it does
    # not contend with the inference GPU.
    viz_executor = None
    viz_futures = []
    viz_base_dir = None
    if run_viz:
        viz_base_dir = os.path.join(
            output_dir,
            f'visualization_multi_{time.strftime("%Y%m%d-%H%M%S")}',
        )
        os.makedirs(viz_base_dir, exist_ok=True)
        # Use spawn so viz workers don't inherit the parent's CUDA context
        # (forking after CUDA init silently breaks the child interpreter).
        import multiprocessing as _mp
        viz_executor = ProcessPoolExecutor(
            max_workers=viz_max_workers,
            mp_context=_mp.get_context('spawn'),
        )
        gpu_print(f"{CLIColors.OKCYAN}Per-bout viz enabled "
                  f"(pool={viz_max_workers}, out={viz_base_dir})"
                  f"{CLIColors.ENDC}", gpu_id=gpu_id)

    # Process each bout
    for bout_idx, (bout_start, bout_end) in enumerate(resolved_bouts):
        bout_len = bout_end - bout_start + 1

        # Skip fully completed bouts on resume
        if bout_idx < bouts_to_skip:
            gpu_print(f"Bout {bout_idx + 1}/{len(resolved_bouts)} "
                      f"[{bout_start}-{bout_end}]: skipped (already complete)",
                      gpu_id=gpu_id)
            continue

        # Reset identity tracker so this bout doesn't inherit stale predicted
        # positions / velocities from the previous bout (otherwise the
        # max_jump cap holds every detection out → all-NaN bout).
        tracker.reset()

        # Determine start offset within this bout (for partial resume)
        start_offset = 0
        if bout_idx == bouts_to_skip and resume_frame_offset > 0:
            start_offset = resume_frame_offset

        desc = (f"Bout {bout_idx + 1}/{len(resolved_bouts)} "
                f"[{bout_start}-{bout_end}]")

        # Dict to collect masks for visualization (saved as .npz after bout)
        mask_npz = {} if use_sam3_mask else None

        # Per-bout CSV writers (used for parallel viz). We dual-write so the
        # concatenated CSVs (resume + downstream API) stay intact.
        bout_csv_paths = {}
        bout_csv_files = {}
        bout_csv_writers = {}
        if viz_executor is not None:
            for fly_id in fly_ids:
                bp = os.path.join(
                    viz_base_dir, f'data3D_{fly_id}_bout{bout_idx}.csv'
                )
                bf = open(bp, 'w', newline='')
                bw = csv.writer(
                    bf, delimiter=',', quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                )
                if has_header:
                    create_header(bw, cfg)
                bout_csv_paths[fly_id] = bp
                bout_csv_files[fly_id] = bf
                bout_csv_writers[fly_id] = bw

        # SAM3 chunking: for long bouts, re-initialize SAM3 every chunk_size
        # frames with overlap for smooth transitions.
        sam3_overlap = 50

        def _init_sam3_chunk(chunk_video_start, chunk_len):
            """Initialize SAM3 for a chunk. Returns tracker ref or None."""
            if sam3_tracker is None:
                return None
            sam3_tracker.start_bout(
                video_paths, chunk_video_start, chunk_len)
            detection_ok = sam3_tracker.detect_frame0(
                num_animals=num_animals, max_retry_frames=10)
            if detection_ok:
                sam3_tracker.assign_identities(
                    reproTool, num_animals=num_animals)
                sam3_tracker.start_propagation()
                return sam3_tracker
            else:
                gpu_print(f"{CLIColors.WARNING}SAM3 detection failed for "
                          f"chunk — falling back to mask-and-redetect."
                          f"{CLIColors.ENDC}", gpu_id=gpu_id)
                sam3_tracker.close()
                return None

        # Determine chunk boundaries for this bout
        if (sam3_tracker is not None and sam3_chunk_size > 0
                and bout_len > sam3_chunk_size):
            # Multiple chunks needed
            chunk_boundaries = []  # (chunk_bout_start, chunk_bout_end)
            pos = 0
            while pos < bout_len:
                chunk_end = min(pos + sam3_chunk_size, bout_len)
                chunk_boundaries.append((pos, chunk_end))
                pos = chunk_end - sam3_overlap
                if bout_len - pos <= sam3_overlap:
                    break  # remaining frames too few for a new chunk
            gpu_print(f"\n{CLIColors.OKCYAN}Bout {bout_idx + 1}: {bout_len} "
                      f"frames → {len(chunk_boundaries)} SAM3 chunks "
                      f"(size={sam3_chunk_size}, overlap={sam3_overlap})"
                      f"{CLIColors.ENDC}", gpu_id=gpu_id)
        else:
            # Single chunk = whole bout
            chunk_boundaries = [(0, bout_len)]

        # Initialize first chunk
        first_chunk_start, first_chunk_end = chunk_boundaries[0]
        first_chunk_len = first_chunk_end - first_chunk_start
        bout_sam3 = None
        current_chunk_idx = 0
        sam3_frame_in_chunk = 0  # tracks position within current SAM3 chunk

        if sam3_tracker is not None:
            gpu_print(f"\n{CLIColors.OKCYAN}Starting SAM3 video propagation for "
                      f"bout {bout_idx + 1} ({bout_len} frames, "
                      f"{len(video_paths)} cameras)"
                      + (f" chunk 1/{len(chunk_boundaries)}"
                         if len(chunk_boundaries) > 1 else "")
                      + f"...{CLIColors.ENDC}", gpu_id=gpu_id)
            bout_sam3 = _init_sam3_chunk(
                bout_start + first_chunk_start, first_chunk_len)
            # Release the init lock after detection (full model now freed)
            if _sam3_lock_held:
                sam3_init_lock.release()
                _sam3_lock_held = False
                gpu_print("Released SAM3 init lock", gpu_id=gpu_id)
            if bout_sam3 is not None:
                gpu_print(f"{CLIColors.OKGREEN}SAM3 streaming ready "
                          f"(init frame: {sam3_tracker.init_frame})."
                          f"{CLIColors.ENDC}", gpu_id=gpu_id)

        # Seek all cameras to bout start (+ offset for partial resume)
        for cap in caps:
            seek(cap, bout_start + start_offset)

        # Producer thread: decodes the next frame's 7 cams in parallel into a
        # bounded queue while the main loop runs GPU work on the previous
        # frame. This pipelines CPU JPEG decode with GPU compute, eliminating
        # the bubble that was holding GPU utilisation at ~75%.
        n_cams = len(caps)
        n_decode_workers = min(n_cams, max(2, n_jobs))
        prefetch_queue = queue.Queue(maxsize=4)
        prefetch_stop = threading.Event()

        def _producer():
            with ThreadPoolExecutor(max_workers=n_decode_workers) as ex:
                for _ in range(start_offset, bout_len):
                    if prefetch_stop.is_set():
                        return
                    buf = np.empty(
                        (n_cams, img_size[1], img_size[0], 3), dtype=np.uint8
                    )
                    futs = [
                        ex.submit(read_images, cap, i, buf)
                        for i, cap in enumerate(caps)
                    ]
                    for f in futs:
                        f.result()
                    prefetch_queue.put(buf)
            prefetch_queue.put(None)  # sentinel

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        for frame_num in ProgressLogger(range(start_offset, bout_len), desc=desc, gpu_id=gpu_id):
            # Check if we need to switch to the next SAM3 chunk
            if (sam3_tracker is not None
                    and current_chunk_idx + 1 < len(chunk_boundaries)):
                next_chunk_start = chunk_boundaries[
                    current_chunk_idx + 1][0]
                if frame_num >= next_chunk_start:
                    # Close current chunk and start next
                    if bout_sam3 is not None:
                        bout_sam3.close()
                    current_chunk_idx += 1
                    cs, ce = chunk_boundaries[current_chunk_idx]
                    gpu_print(f"Starting SAM3 chunk "
                              f"{current_chunk_idx + 1}/"
                              f"{len(chunk_boundaries)} "
                              f"[frames {cs}-{ce}]...",
                              gpu_id=gpu_id)
                    bout_sam3 = _init_sam3_chunk(
                        bout_start + cs, ce - cs)
                    sam3_frame_in_chunk = 0
                    if bout_sam3 is not None:
                        # Validate identity continuity with tracker state
                        prev_centers_by_idx = {
                            int(fid.replace('fly', '')): center
                            for fid, center in tracker.prev_centers.items()
                        }
                        swapped = bout_sam3.validate_identity_continuity(
                            prev_centers_by_idx, reproTool,
                            num_animals=num_animals)
                        if swapped:
                            gpu_print(
                                f"{CLIColors.WARNING}SAM3 identities "
                                f"corrected at chunk boundary"
                                f"{CLIColors.ENDC}",
                                gpu_id=gpu_id)
                        gpu_print("SAM3 chunk ready.", gpu_id=gpu_id)

            # Pull next pre-decoded frame from producer thread
            imgs_orig = prefetch_queue.get()
            if imgs_orig is None:
                break

            # Convert to tensor
            imgs = (
                torch.from_numpy(imgs_orig)
                .cuda()
                .float()
                .permute(0, 3, 1, 2)[:, [2, 1, 0]]
                / 255.0
            )

            # Get SAM3 masks for this frame (streaming propagation)
            frame_masks = None
            if bout_sam3 is not None:
                if sam3_frame_in_chunk == bout_sam3.init_frame:
                    frame_masks = bout_sam3.get_frame0_masks(
                        num_animals=num_animals)
                elif sam3_frame_in_chunk > bout_sam3.init_frame:
                    frame_masks = bout_sam3.get_next_frame(
                        num_animals=num_animals)
                sam3_frame_in_chunk += 1

            # Run multi-animal prediction
            detections = jarvisPredictor(
                imgs, camera_matrices_cuda,
                precomputed_masks=frame_masks,
            )

            # Assign identities via tracker
            assignments = tracker.assign_identities(detections)

            # Save masks for visualization (keyed by assigned identity)
            if frame_masks is not None and mask_npz is not None:
                # Map assigned detections back to frame_masks indices
                for fly_id in fly_ids:
                    det = assignments.get(fly_id)
                    if det is not None:
                        # Find which detection index this is
                        for di, d in enumerate(detections):
                            if d is det and di < len(frame_masks):
                                fm = frame_masks[di]
                                key = f'f{frame_num:06d}_{fly_id}'
                                mask_npz[key] = np.packbits(
                                    fm['masks'].numpy())
                                break

            # Write results to per-animal CSVs (and per-bout CSVs if viz)
            for fly_id in fly_ids:
                det = assignments.get(fly_id)
                if det is not None and det.get('points3D') is not None:
                    points3D = det['points3D']
                    confidences = det['confidences']
                    if points3D.dim() != 2:
                        writers[fly_id].writerow(nan_row)
                        if fly_id in bout_csv_writers:
                            bout_csv_writers[fly_id].writerow(nan_row)
                        continue
                    row = []
                    for point, conf in zip(
                        points3D.squeeze(),
                        confidences.squeeze().cpu().numpy()
                    ):
                        row = row + point.tolist() + [conf]
                    writers[fly_id].writerow(row)
                    if fly_id in bout_csv_writers:
                        bout_csv_writers[fly_id].writerow(row)
                else:
                    writers[fly_id].writerow(nan_row)
                    if fly_id in bout_csv_writers:
                        bout_csv_writers[fly_id].writerow(nan_row)

        # Save masks for visualization
        if mask_npz:
            mask_path = os.path.join(
                output_dir, f'masks_bout{bout_idx}.npz')
            mask_npz['_meta'] = np.array([
                img_size[1], img_size[0], len(caps)  # H, W, num_cameras
            ])
            np.savez_compressed(mask_path, **mask_npz)
            gpu_print(f"Saved {len(mask_npz) - 1} mask frames to {mask_path}",
                      gpu_id=gpu_id)

        # Stop producer thread (drain any leftover items)
        prefetch_stop.set()
        try:
            while True:
                prefetch_queue.get_nowait()
        except queue.Empty:
            pass
        producer_thread.join(timeout=5)

        # Clean up streaming tracker and free memory for this bout
        if bout_sam3 is not None:
            bout_sam3.close()
        del mask_npz
        gc.collect()
        torch.cuda.empty_cache()

        # Close per-bout CSVs and submit viz job (runs in background while
        # the next bout's GPU inference proceeds).
        if viz_executor is not None:
            for fly_id in fly_ids:
                bout_csv_files[fly_id].close()
            mask_path = os.path.join(
                output_dir, f'masks_bout{bout_idx}.npz')
            viz_mask_file = mask_path if os.path.exists(mask_path) else None
            bout_viz_dir = os.path.join(viz_base_dir, f'bout{bout_idx}')
            fut = viz_executor.submit(
                _viz_bout_worker,
                project_name, video_folder, calib_folder,
                dict(bout_csv_paths),
                bout_start, bout_len, viz_cameras,
                viz_mask_file, bout_viz_dir,
            )
            viz_futures.append((bout_idx, fut))

            def _viz_done(f, _bi=bout_idx):
                exc = f.exception()
                if exc is not None:
                    gpu_print(
                        f"{CLIColors.WARNING}Viz bout {_bi} crashed: "
                        f"{exc}{CLIColors.ENDC}", gpu_id=gpu_id)
            fut.add_done_callback(_viz_done)

            gpu_print(f"Submitted viz for bout {bout_idx + 1}/"
                      f"{len(resolved_bouts)} → {bout_viz_dir}",
                      gpu_id=gpu_id)

    # Safety release if lock was never released (e.g. all bouts skipped)
    if _sam3_lock_held:
        sam3_init_lock.release()
        _sam3_lock_held = False

    # Wait for any in-flight per-bout viz jobs to finish
    if viz_executor is not None:
        gpu_print(f"Waiting for {len(viz_futures)} viz job(s) to finish...",
                  gpu_id=gpu_id)
        for bi, fut in viz_futures:
            try:
                fut.result()
            except Exception as e:
                gpu_print(f"{CLIColors.WARNING}Viz bout {bi} failed: "
                          f"{e}{CLIColors.ENDC}", gpu_id=gpu_id)
        viz_executor.shutdown(wait=True)

    # Cleanup
    for cap in caps:
        cap.release()
    for fly_id in fly_ids:
        csvfiles[fly_id].close()

    # Save tracking info
    tracking_info = tracker.get_tracking_info()
    tracking_info['bouts'] = [
        {'start': s, 'end': e} for s, e in resolved_bouts
    ]
    with open(os.path.join(output_dir, 'tracking_info.json'), 'w') as f:
        json.dump(tracking_info, f, indent=2)

    gpu_print(f"\n{CLIColors.OKGREEN}Tracking summary:{CLIColors.ENDC}",
              gpu_id=gpu_id)
    gpu_print(f"  Frames processed: {tracking_info['frame_count']}",
              gpu_id=gpu_id)
    gpu_print(f"  Swap corrections: {tracking_info['swap_corrections']}",
              gpu_id=gpu_id)
    gpu_print(f"  Body sizes (EMA): {tracking_info['body_sizes']}",
              gpu_id=gpu_id)
    gpu_print(f"{CLIColors.OKGREEN}Completed: {output_dir}{CLIColors.ENDC}\n",
              gpu_id=gpu_id)

    # Collect mask file paths (one per bout)
    mask_files = {}
    for bi in range(len(resolved_bouts)):
        mp = os.path.join(output_dir, f'masks_bout{bi}.npz')
        if os.path.exists(mp):
            mask_files[bi] = mp

    return {
        "output_dir": output_dir,
        "data_csvs": {
            fly_id: os.path.join(output_dir, f'data3D_{fly_id}.csv')
            for fly_id in fly_ids
        },
        "mask_files": mask_files,
        "video_folder": video_folder,
        "calib_folder": calib_folder,
        "bouts": resolved_bouts,
        "total_bout_frames": total_bout_frames,
        "tracking_info": tracking_info,
        "viz_output_dir": viz_base_dir,
    }


def _gpu_worker(gpu_id, kwargs, sam3_init_lock=None):
    """Worker process for multi-GPU bout processing."""
    try:
        # Prevent HuggingFace tokenizer deadlock in spawned processes
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        # Give each worker its own torch.compile cache to avoid lock conflicts
        # (SAM3 uses torch.compile extensively; concurrent compilation with a
        # shared inductor cache causes one process to block)
        import tempfile
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(
            tempfile.gettempdir(), f'torchinductor_gpu{gpu_id}')
        # Make prints visible immediately (spawned processes use full buffering)
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

        torch.cuda.set_device(gpu_id)
        kwargs['gpu_id'] = gpu_id
        kwargs['sam3_device'] = f'cuda:{gpu_id}'
        kwargs['sam3_init_lock'] = sam3_init_lock
        run_prediction(**kwargs)
    except Exception as e:
        # Release lock on failure so other workers don't deadlock
        if sam3_init_lock is not None:
            try:
                sam3_init_lock.release()
            except ValueError:
                pass  # already released
        gpu_print(f"{CLIColors.FAIL}Worker failed: {e}{CLIColors.ENDC}",
                  gpu_id=gpu_id)
        import traceback
        traceback.print_exc()


def _detect_header_rows(csv_path):
    """Count header rows (non-numeric first field) in a CSV."""
    n_header = 0
    with open(csv_path, 'r') as f:
        for line in f:
            first_val = line.strip().split(',')[0]
            try:
                float(first_val)
                break
            except ValueError:
                n_header += 1
    return n_header


def _check_worker_identity_swap(prev_worker_dir, next_worker_dir, fly_ids,
                                n_header, n_compare=20):
    """Compare the end of one worker with the start of the next to detect
    identity swaps.  Uses 3D Antenna_Base position (cols 0-2) proximity.

    Returns True if next_worker's fly IDs should be swapped relative to
    prev_worker.
    """
    if len(fly_ids) != 2:
        return False

    def _load_endpoints(wdir, from_end, n):
        """Load xyz for Antenna_Base (cols 0-2) from first/last n data rows."""
        positions = {}
        for fly_id in fly_ids:
            csv_path = os.path.join(wdir, f'data3D_{fly_id}.csv')
            if not os.path.exists(csv_path):
                return None
            data = np.genfromtxt(csv_path, delimiter=',',
                                 skip_header=n_header)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if from_end:
                chunk = data[-n:]
            else:
                chunk = data[:n]
            # Average non-NaN positions
            pts = chunk[:, 0:3]  # Antenna_Base x,y,z
            valid = ~np.any(np.isnan(pts), axis=1)
            if valid.sum() == 0:
                positions[fly_id] = None
            else:
                positions[fly_id] = pts[valid].mean(axis=0)
        return positions

    prev_pos = _load_endpoints(prev_worker_dir, from_end=True, n=n_compare)
    next_pos = _load_endpoints(next_worker_dir, from_end=False, n=n_compare)

    if prev_pos is None or next_pos is None:
        return False
    if any(v is None for v in prev_pos.values()):
        return False
    if any(v is None for v in next_pos.values()):
        return False

    f0, f1 = fly_ids[0], fly_ids[1]
    cost_keep = (np.linalg.norm(prev_pos[f0] - next_pos[f0])
                 + np.linalg.norm(prev_pos[f1] - next_pos[f1]))
    cost_swap = (np.linalg.norm(prev_pos[f0] - next_pos[f1])
                 + np.linalg.norm(prev_pos[f1] - next_pos[f0]))

    return cost_swap < cost_keep


def _merge_gpu_outputs(main_output_dir, worker_dirs, all_bouts, fly_ids,
                       header_lines=None):
    """
    Merge per-worker CSV and mask outputs into the main output directory.

    Each worker produced CSVs and mask npz files for its subset of bouts.
    This function concatenates them in bout order, detecting and correcting
    identity swaps between workers using 3D position proximity.
    """
    # Detect header size from first worker
    first_worker_csv = os.path.join(worker_dirs[0],
                                    f'data3D_{fly_ids[0]}.csv')
    n_header = _detect_header_rows(first_worker_csv) if os.path.exists(
        first_worker_csv) else 0

    # Determine which workers need identity swapping.
    # Compare raw data of adjacent workers. Since the comparison uses raw
    # (uncorrected) files, XOR with the previous swap flag gives the
    # cumulative correction relative to worker 0.
    swap_flags = [False] * len(worker_dirs)
    for wi in range(1, len(worker_dirs)):
        raw_swap = _check_worker_identity_swap(
            worker_dirs[wi - 1], worker_dirs[wi], fly_ids, n_header)
        swap_flags[wi] = raw_swap ^ swap_flags[wi - 1]
        if swap_flags[wi]:
            print(f"{CLIColors.WARNING}  Worker {wi} ({worker_dirs[wi]}): "
                  f"identity swap detected — correcting{CLIColors.ENDC}")

    # Merge CSVs with swap correction
    # Open all output files
    out_files = {}
    for fly_id in fly_ids:
        main_csv = os.path.join(main_output_dir, f'data3D_{fly_id}.csv')
        out_files[fly_id] = open(main_csv, 'w')

    # Write header from first worker
    if os.path.exists(first_worker_csv):
        with open(first_worker_csv, 'r') as in_f:
            header_text = []
            for _ in range(n_header):
                header_text.append(in_f.readline())
        for fly_id in fly_ids:
            for h_line in header_text:
                out_files[fly_id].write(h_line)

    # Write data from all workers in order
    for wi, wdir in enumerate(worker_dirs):
        if swap_flags[wi] and len(fly_ids) == 2:
            # Swapped: read fly0 data → write to fly1 output and vice versa
            src_map = {fly_ids[0]: fly_ids[1], fly_ids[1]: fly_ids[0]}
        else:
            src_map = {fid: fid for fid in fly_ids}

        for out_fly_id in fly_ids:
            src_fly_id = src_map[out_fly_id]
            wcsv = os.path.join(wdir, f'data3D_{src_fly_id}.csv')
            if not os.path.exists(wcsv):
                continue
            with open(wcsv, 'r') as in_f:
                wlines = in_f.readlines()
            for line in wlines[n_header:]:
                out_files[out_fly_id].write(line)

    for f in out_files.values():
        f.close()

    # Merge mask files: renumber bout indices
    global_bout_idx = 0
    for wdir in worker_dirs:
        local_bout = 0
        while True:
            src = os.path.join(wdir, f'masks_bout{local_bout}.npz')
            if not os.path.exists(src):
                break
            dst = os.path.join(main_output_dir,
                               f'masks_bout{global_bout_idx}.npz')
            os.rename(src, dst)
            local_bout += 1
            global_bout_idx += 1

    # Merge tracking info
    merged_info = {
        'num_animals': len(fly_ids),
        'frame_count': 0,
        'swap_corrections': 0,
        'fly_ids': fly_ids,
        'body_sizes': {},
        'initialized': True,
        'bouts': [{'start': s, 'end': e} for s, e in all_bouts],
    }
    for wdir in worker_dirs:
        info_path = os.path.join(wdir, 'tracking_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                winfo = json.load(f)
            merged_info['frame_count'] += winfo.get('frame_count', 0)
            merged_info['swap_corrections'] += winfo.get(
                'swap_corrections', 0)
            merged_info['body_sizes'] = winfo.get('body_sizes',
                                                   merged_info['body_sizes'])

    with open(os.path.join(main_output_dir, 'tracking_info.json'), 'w') as f:
        json.dump(merged_info, f, indent=2)

    # Clean up worker directories
    import shutil
    for wdir in worker_dirs:
        shutil.rmtree(wdir, ignore_errors=True)


def run_prediction_multi_gpu(gpu_ids, common_kwargs, resolved_bouts,
                             output_name=None):
    """
    Dispatch bout processing across multiple GPUs.

    Splits bouts into groups (one per GPU), spawns a worker process per GPU,
    and merges the results. If output_name is provided, uses a deterministic
    directory name and auto-resumes if prior data exists.
    """
    import torch.multiprocessing as mp

    # Prevent tokenizer deadlock before spawning
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    n_gpus = len(gpu_ids)
    n_bouts = len(resolved_bouts)

    # Split bouts into contiguous blocks so merged CSV preserves bout order.
    # (Round-robin would interleave bouts across workers, breaking the
    # visualization's sequential csv_offset assumption after merge.)
    import math
    bouts_per_gpu = math.ceil(n_bouts / n_gpus)
    bout_groups = []
    for gi in range(n_gpus):
        start_idx = gi * bouts_per_gpu
        end_idx = min(start_idx + bouts_per_gpu, n_bouts)
        if start_idx < n_bouts:
            bout_groups.append(list(resolved_bouts[start_idx:end_idx]))
        else:
            bout_groups.append([])

    # Create output directory — save alongside the recording videos
    pred_base = common_kwargs['video_folder']
    if output_name:
        main_output_dir = os.path.join(
            pred_base, f"Predictions_3D_{output_name}")
    else:
        main_output_dir = os.path.join(
            pred_base,
            f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}',
        )
    os.makedirs(main_output_dir, exist_ok=True)

    # Resume check
    fly_ids = [f'fly{i}' for i in range(
        common_kwargs.get('num_animals', 2))]
    existing_csvs = [
        os.path.join(main_output_dir, f'data3D_{fid}.csv')
        for fid in fly_ids
    ]
    has_merged_data = all(os.path.isfile(c) for c in existing_csvs)

    if has_merged_data:
        # Previous run fully completed and merged — single-GPU resume on
        # the merged CSVs (remaining work is typically small).
        print(f"{CLIColors.OKCYAN}Merged data found in {main_output_dir}, "
              f"resuming in single-GPU mode...{CLIColors.ENDC}")
        resume_kwargs = dict(common_kwargs)
        resume_kwargs['resume_dir'] = main_output_dir
        resume_kwargs['output_name'] = None
        resume_kwargs['frame_ranges'] = resolved_bouts
        resume_kwargs.pop('frame_start', None)
        resume_kwargs.pop('frame_end', None)
        resume_kwargs['gpu_id'] = gpu_ids[0]
        return run_prediction(**resume_kwargs)

    # Detect per-worker progress from a previous run (preempted mid-work).
    # Workers that already finished all their bouts are skipped; partially-
    # done workers resume via the existing CSV-row-count logic.
    use_sam3 = common_kwargs.get('use_sam3_mask', False)
    resuming_workers = False

    # Build per-worker directories and detect progress
    worker_dirs = []
    worker_frames_done = {}  # gpu_id -> frames already written
    for i, gpu_id in enumerate(gpu_ids):
        if not bout_groups[i]:
            continue
        wdir = os.path.join(main_output_dir, f'_worker_gpu{gpu_id}')
        worker_dirs.append(wdir)
        # Check if this worker has prior progress
        first_csv = os.path.join(wdir, f'data3D_{fly_ids[0]}.csv')
        if os.path.isfile(first_csv):
            with open(first_csv, 'r') as f:
                total_lines = sum(1 for _ in f)
            # Detect header
            header_rows = 0
            try:
                with open(first_csv, 'r') as f:
                    first_line = f.readline()
                    if first_line and not first_line[0].isdigit() and first_line[0] != '-':
                        header_rows = 2
            except Exception:
                pass
            worker_frames_done[gpu_id] = max(0, total_lines - header_rows)
        else:
            worker_frames_done[gpu_id] = 0

    if any(v > 0 for v in worker_frames_done.values()):
        resuming_workers = True
        # Count expected frames per worker
        worker_expected = {}
        gi = 0
        for i, gpu_id in enumerate(gpu_ids):
            if not bout_groups[i]:
                continue
            worker_expected[gpu_id] = sum(
                e - s + 1 for s, e in bout_groups[i])
            gi += 1

        for gpu_id, done in worker_frames_done.items():
            expected = worker_expected.get(gpu_id, 0)
            if done >= expected:
                print(f"{CLIColors.OKCYAN}  GPU {gpu_id}: fully complete "
                      f"({done}/{expected} frames){CLIColors.ENDC}")
            elif done > 0:
                print(f"{CLIColors.OKCYAN}  GPU {gpu_id}: resuming "
                      f"({done}/{expected} frames done){CLIColors.ENDC}")
            else:
                print(f"  GPU {gpu_id}: starting fresh "
                      f"({expected} frames)")

    for wdir in worker_dirs:
        os.makedirs(wdir, exist_ok=True)

    # Create info file (only on first run)
    total_frames = sum(e - s + 1 for s, e in resolved_bouts)
    info_path = os.path.join(main_output_dir, 'info.cfg')
    if not os.path.isfile(info_path):
        create_info_file(
            main_output_dir,
            common_kwargs['video_folder'],
            common_kwargs['calib_folder'],
            resolved_bouts[0][0],
            total_frames,
            common_kwargs.get('num_animals', 2),
        )

    print(f"\n{CLIColors.BOLD}Multi-GPU dispatch: {n_bouts} bouts across "
          f"{n_gpus} GPUs {gpu_ids}"
          + (f" (resuming)" if resuming_workers else "")
          + f"{CLIColors.ENDC}")
    for i, gpu_id in enumerate(gpu_ids):
        if bout_groups[i]:
            print(f"  GPU {gpu_id}: {len(bout_groups[i])} bouts")

    # Spawn worker processes
    # Lock serializes the heavy SAM3 full-model load+detect phase so only
    # one worker holds the ~4 GB model in memory at a time (prevents OOM).
    mp.set_start_method('spawn', force=True)
    sam3_init_lock = mp.Lock() if use_sam3 else None
    processes = []
    active_worker_dirs = []
    for i, gpu_id in enumerate(gpu_ids):
        if not bout_groups[i]:
            continue

        wdir = os.path.join(main_output_dir, f'_worker_gpu{gpu_id}')
        done = worker_frames_done.get(gpu_id, 0)
        expected = sum(e - s + 1 for s, e in bout_groups[i])

        # Skip fully-completed workers
        if done >= expected and resuming_workers:
            print(f"{CLIColors.OKCYAN}Skipping GPU {gpu_id} — already "
                  f"complete{CLIColors.ENDC}")
            active_worker_dirs.append(wdir)
            continue

        kwargs = dict(common_kwargs)
        kwargs['frame_ranges'] = bout_groups[i]
        kwargs.pop('frame_start', None)
        kwargs.pop('frame_end', None)

        if done > 0 and resuming_workers:
            # Resume: pass resume_dir so run_prediction() picks up from
            # existing CSV progress (bout skip + partial frame offset)
            kwargs['resume_dir'] = wdir
        else:
            kwargs['output_dir'] = wdir

        active_worker_dirs.append(wdir)

        p = mp.Process(target=_gpu_worker,
                        args=(gpu_id, kwargs, sam3_init_lock))
        p.start()
        processes.append((gpu_id, p))

    # Wait for all workers
    failed = []
    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append(gpu_id)
            print(f"{CLIColors.FAIL}GPU {gpu_id} worker exited with "
                  f"code {p.exitcode}{CLIColors.ENDC}")

    if failed:
        print(f"{CLIColors.FAIL}Workers failed on GPUs: {failed}. "
              f"Partial results in {main_output_dir}{CLIColors.ENDC}")
        return None

    # Merge outputs
    fly_ids = [f'fly{i}' for i in range(
        common_kwargs.get('num_animals', 2))]
    print(f"\n{CLIColors.OKCYAN}Merging outputs from {len(active_worker_dirs)} "
          f"workers...{CLIColors.ENDC}")
    _merge_gpu_outputs(main_output_dir, active_worker_dirs,
                       resolved_bouts, fly_ids)

    # Collect mask files
    mask_files = {}
    for bi in range(len(resolved_bouts)):
        mp_path = os.path.join(main_output_dir, f'masks_bout{bi}.npz')
        if os.path.exists(mp_path):
            mask_files[bi] = mp_path

    print(f"{CLIColors.OKGREEN}Multi-GPU prediction complete: "
          f"{main_output_dir}{CLIColors.ENDC}\n")

    return {
        "output_dir": main_output_dir,
        "data_csvs": {
            fid: os.path.join(main_output_dir, f'data3D_{fid}.csv')
            for fid in fly_ids
        },
        "mask_files": mask_files,
        "video_folder": common_kwargs['video_folder'],
        "calib_folder": common_kwargs['calib_folder'],
        "bouts": resolved_bouts,
        "total_bout_frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Animal Batch 3D Prediction for JARVIS-HybridNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "-p", "--project", type=str, required=True,
        help="Name of the JARVIS project to use"
    )
    parser.add_argument(
        "-c", "--config", type=str,
        help="Path to JSON config file with batch jobs"
    )
    parser.add_argument(
        "-v", "--video_folder", type=str, action="append",
        help="Path to video folder (can be specified multiple times)"
    )
    parser.add_argument(
        "-k", "--calib_folder", type=str, action="append",
        help="Path to calibration folder (must match video_folder count)"
    )
    parser.add_argument(
        "--num_animals", type=int, default=2,
        help="Number of animals to track (default: 2)"
    )
    parser.add_argument(
        "--suppression_radius", type=int, default=15,
        help="NMS suppression radius in heatmap pixels (default: 15)"
    )
    parser.add_argument(
        "--mask_scale", type=float, default=1.5,
        help="Mask scale for mask-and-redetect (fraction of bbox_hw). "
             "1.5 masks 576x576px around detected center. Increase if "
             "same fly is detected twice; decrease if second fly is missed "
             "when flies are very close. (default: 1.5)"
    )
    parser.add_argument(
        "--frame_start", type=int, default=0,
        help="Starting frame (default: 0)"
    )
    parser.add_argument(
        "--frame_end", type=int, default=-1,
        help="Ending frame (default: -1 for all frames)"
    )
    parser.add_argument(
        "--frame_file", type=str,
        help="Path to a text file with frame ranges (one 'start end' pair "
             "per line). Frames outside the ranges are filled with NaN."
    )
    parser.add_argument(
        "--bouts_csv", type=str,
        help="Path to a bouts summary CSV with columns: fly_id, bout_idx, "
             "start_frame, end_frame. Only bout frames are predicted; "
             "gaps are filled with NaN. Overrides --frame_start/--frame_end."
    )
    parser.add_argument(
        "--weights_center", type=str, default="latest",
        help="Weights for CenterDetect (default: 'latest')"
    )
    parser.add_argument(
        "--weights_hybridnet", type=str, default="latest",
        help="Weights for HybridNet (default: 'latest')"
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+",
        help="Optional list of camera names to use"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=17,
        help="Number of parallel jobs for frame reading (default: 17)"
    )
    parser.add_argument(
        "--trt_mode", type=str, default="off", choices=["off", "on"],
        help="TensorRT mode (default: 'off')"
    )
    parser.add_argument(
        "--run_viz", action="store_true",
        help="Create visualization videos after prediction with per-fly colors"
    )
    parser.add_argument(
        "--viz_cameras", type=str, nargs="+",
        help="Camera names to create visualization videos for (default: all)"
    )
    parser.add_argument(
        "--use_sam3_mask", action="store_true",
        help="Enable SAM3 per-frame segmentation. Disabled by default; "
             "the multi-peak CenterDetect path is used instead."
    )
    parser.add_argument(
        "--sam3_device", type=str, default="cuda",
        help="Device for SAM3 model (default: 'cuda'). Use 'cuda:1' for "
             "second GPU if memory is tight."
    )
    parser.add_argument(
        "--sam3_text_prompt", type=str, default="insect",
        help="Text prompt for SAM3 segmentation (default: 'fly')"
    )
    parser.add_argument(
        "--no_sam3_constrain_keypoints", action="store_true",
        help="Disable SAM3 keypoint constraining (only use SAM3 for masking, "
             "not for constraining keypoint detection)"
    )
    parser.add_argument(
        "--sam3_chunk_size", type=int, default=1000,
        help="Max frames per SAM3 video chunk. Longer bouts are split into "
             "overlapping chunks that each get fresh detection + propagation. "
             "Set to 0 to disable chunking. (default: 1000)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing output directory to resume from. "
             "Skips already-completed bouts based on CSV row counts."
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Custom name for the output directory (e.g. SLURM job ID). "
             "Creates Predictions_3D_<output_name>. If the directory already "
             "exists, automatically resumes from where it left off."
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=None,
        help="GPU IDs for multi-GPU parallel processing. Each GPU processes "
             "a subset of bouts independently. Example: --gpus 0 1 for 2 GPUs, "
             "--gpus 0 1 2 3 4 5 6 7 for 8 GPUs on a cluster node. "
             "For full-video mode (no --bouts_csv), the video is auto-split "
             "into chunks, one per GPU."
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="Number of GPUs to use (shorthand for --gpus 0 1 ... N-1). "
             "Ignored if --gpus is also provided."
    )
    parser.add_argument(
        "--min_animal_separation_mm", type=float, default=0.0,
        help="Identity-collapse guard threshold in mm. When > 0, two "
             "triangulated animal centers closer than this are collapsed to "
             "one in multi_peak and the tracker so the output never contains "
             "two tracks on the same physical animal. ~1.0 mm is a reasonable "
             "floor for courtship Drosophila. 0 disables (default)."
    )

    args = parser.parse_args()

    # Resolve --num_gpus to --gpus if needed
    if args.gpus is None and args.num_gpus is not None:
        args.gpus = list(range(args.num_gpus))

    # Build list of jobs
    jobs = []

    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        raw_jobs = config.get("jobs", [])
        for job in raw_jobs:
            if "bouts_csv" in job and job["bouts_csv"]:
                job["frame_ranges"] = parse_bouts_csv(job["bouts_csv"])
                job.pop("frame_start", None)
                job.pop("frame_end", None)
                del job["bouts_csv"]
            elif "frame_file" in job and job["frame_file"]:
                job["frame_ranges"] = parse_frame_file(job["frame_file"])
                job.pop("frame_start", None)
                job.pop("frame_end", None)
                del job["frame_file"]
            jobs.append(job)
        print(f"Loaded {len(jobs)} jobs from config file")

    if args.video_folder and args.calib_folder:
        if len(args.video_folder) != len(args.calib_folder):
            print(f"{CLIColors.FAIL}Error: Number of video folders must "
                  f"match calibration folders{CLIColors.ENDC}")
            sys.exit(1)

        for video, calib in zip(args.video_folder, args.calib_folder):
            job_entry = {
                "video_folder": video,
                "calib_folder": calib,
            }
            if args.bouts_csv:
                job_entry["frame_ranges"] = parse_bouts_csv(args.bouts_csv)
            elif args.frame_file:
                job_entry["frame_ranges"] = parse_frame_file(args.frame_file)
            else:
                job_entry["frame_start"] = args.frame_start
                job_entry["frame_end"] = args.frame_end
            jobs.append(job_entry)

    if not jobs:
        print(f"{CLIColors.FAIL}Error: No jobs specified. Use --config or "
              f"--video_folder/--calib_folder{CLIColors.ENDC}")
        parser.print_help()
        sys.exit(1)

    # Process all jobs
    results = []
    total_jobs = len(jobs)

    print(f"\n{CLIColors.BOLD}Starting multi-animal batch prediction "
          f"with {total_jobs} job(s), {args.num_animals} animals{CLIColors.ENDC}\n")

    for i, job in enumerate(jobs):
        print(f"\n{CLIColors.BOLD}[Job {i + 1}/{total_jobs}]{CLIColors.ENDC}")

        common_kwargs = dict(
            project_name=args.project,
            video_folder=job["video_folder"],
            calib_folder=job["calib_folder"],
            num_animals=args.num_animals,
            suppression_radius=args.suppression_radius,
            mask_scale=args.mask_scale,
            cameras_to_use=args.cameras,
            weights_center_detect=args.weights_center,
            weights_hybridnet=args.weights_hybridnet,
            trt_mode=args.trt_mode,
            n_jobs=args.n_jobs,
            use_sam3_mask=args.use_sam3_mask,
            sam3_device=args.sam3_device,
            sam3_text_prompt=args.sam3_text_prompt,
            sam3_constrain_keypoints=not args.no_sam3_constrain_keypoints,
            sam3_chunk_size=args.sam3_chunk_size,
            output_name=args.output_name,
            min_animal_separation_mm=args.min_animal_separation_mm,
        )

        use_multi_gpu = args.gpus is not None and len(args.gpus) > 1
        frame_ranges = job.get("frame_ranges", None)

        if use_multi_gpu:
            # Resolve bouts for multi-GPU splitting
            if frame_ranges is not None and len(frame_ranges) > 0:
                resolved = list(frame_ranges)
            else:
                # Full-video mode: auto-split into chunks for parallel processing
                fs = job.get("frame_start", args.frame_start)
                fe = job.get("frame_end", args.frame_end)
                # Need total frame count to resolve -1
                video_paths_tmp = sorted(glob.glob(
                    os.path.join(job["video_folder"], "*.avi"))
                ) or sorted(glob.glob(
                    os.path.join(job["video_folder"], "*.mp4")))
                if video_paths_tmp:
                    tmp_cap = cv2.VideoCapture(video_paths_tmp[0])
                    total_f = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    tmp_cap.release()
                else:
                    total_f = 100000
                if fe == -1:
                    fe = total_f - 1
                fe = min(fe, total_f - 1)
                # Split into N chunks (one per GPU)
                total_len = fe - fs + 1
                chunk_size = max(1, total_len // len(args.gpus))
                resolved = []
                pos = fs
                for gi in range(len(args.gpus)):
                    chunk_end = min(pos + chunk_size - 1, fe)
                    if gi == len(args.gpus) - 1:
                        chunk_end = fe  # last GPU gets remainder
                    if pos <= fe:
                        resolved.append((pos, chunk_end))
                    pos = chunk_end + 1
                print(f"Full-video auto-split: {total_len} frames → "
                      f"{len(resolved)} chunks for {len(args.gpus)} GPUs")

            # Pop output_name: used by dispatcher, not individual workers
            mgpu_output_name = common_kwargs.pop('output_name', None)
            prediction_result = run_prediction_multi_gpu(
                gpu_ids=args.gpus,
                common_kwargs=common_kwargs,
                resolved_bouts=resolved,
                output_name=mgpu_output_name,
            )
        else:
            common_kwargs['frame_start'] = job.get(
                "frame_start", args.frame_start)
            common_kwargs['frame_end'] = job.get(
                "frame_end", args.frame_end)
            common_kwargs['frame_ranges'] = frame_ranges
            common_kwargs['resume_dir'] = args.resume
            if args.gpus and len(args.gpus) == 1:
                common_kwargs['gpu_id'] = args.gpus[0]
            # Run per-bout viz inline (overlaps with next bout's inference)
            common_kwargs['run_viz'] = (
                args.run_viz or job.get("run_viz", False)
            )
            common_kwargs['viz_cameras'] = (
                args.viz_cameras or job.get("viz_cameras", None)
            )
            prediction_result = run_prediction(**common_kwargs)

        viz_output_dir = None
        # If run_prediction already handled viz inline (per-bout parallel),
        # use that output dir and skip the post-prediction viz block.
        if prediction_result is not None and prediction_result.get(
            "viz_output_dir"
        ):
            viz_output_dir = prediction_result["viz_output_dir"]
        if prediction_result is not None and viz_output_dir is None and (
            args.run_viz or job.get("run_viz", False)
        ):
            viz_cameras = (args.viz_cameras
                           or job.get("viz_cameras", None))
            bouts = prediction_result.get("bouts", [(0, -1)])
            data_csvs = prediction_result["data_csvs"]
            mask_files = prediction_result.get("mask_files", {})

            print(f"\n{CLIColors.HEADER}Creating visualization videos..."
                  f"{CLIColors.ENDC}")

            if len(bouts) == 1:
                # Single contiguous range: visualize directly
                viz_output_dir = create_multi_animal_videos3D(
                    project_name=args.project,
                    recording_path=job["video_folder"],
                    data_csvs=data_csvs,
                    dataset_name=job["calib_folder"],
                    frame_start=bouts[0][0],
                    number_frames=bouts[0][1] - bouts[0][0] + 1,
                    video_cam_list=viz_cameras,
                    mask_file=mask_files.get(0),
                )
            else:
                # Multi-bout: create per-bout CSVs and visualize each
                # The concatenated CSVs have bout data back-to-back,
                # so we split them into per-bout temporary CSVs.
                viz_base_dir = os.path.join(
                    prediction_result["output_dir"],
                    f'visualization_multi_{time.strftime("%Y%m%d-%H%M%S")}',
                )
                os.makedirs(viz_base_dir, exist_ok=True)
                csv_offset = 0
                for bi, (bs, be) in enumerate(bouts):
                    bout_len = be - bs + 1
                    bout_csvs = {}
                    for fly_id, csv_path in data_csvs.items():
                        # Read the full CSV and extract this bout's rows
                        all_data = np.genfromtxt(csv_path, delimiter=',')
                        if all_data.ndim == 1:
                            continue
                        # Skip header rows
                        header_rows = 0
                        if np.isnan(all_data[0, 0]):
                            header_rows = 2
                        data_rows = all_data[header_rows:]
                        bout_rows = data_rows[csv_offset:csv_offset + bout_len]
                        bout_csv_path = os.path.join(
                            viz_base_dir,
                            f'data3D_{fly_id}_bout{bi}.csv'
                        )
                        # Write with header
                        header = all_data[:header_rows] if header_rows > 0 else None
                        with open(bout_csv_path, 'w') as bf:
                            if header is not None:
                                # Re-read original header lines (text)
                                with open(csv_path, 'r') as orig:
                                    for h in range(header_rows):
                                        bf.write(orig.readline())
                            for row in bout_rows:
                                bf.write(','.join(str(v) for v in row) + '\n')
                        bout_csvs[fly_id] = bout_csv_path

                    bout_viz_dir = os.path.join(viz_base_dir, f'bout{bi}')
                    print(f"  Bout {bi + 1}/{len(bouts)} "
                          f"[{bs}-{be}] ({bout_len} frames)")
                    create_multi_animal_videos3D(
                        project_name=args.project,
                        recording_path=job["video_folder"],
                        data_csvs=bout_csvs,
                        dataset_name=job["calib_folder"],
                        frame_start=bs,
                        number_frames=bout_len,
                        mask_file=mask_files.get(bi),
                        video_cam_list=viz_cameras,
                        output_dir=bout_viz_dir,
                    )
                    csv_offset += bout_len
                viz_output_dir = viz_base_dir

        results.append({
            "video_folder": job["video_folder"],
            "output_dir": (prediction_result["output_dir"]
                           if prediction_result else None),
            "viz_output_dir": viz_output_dir,
            "success": prediction_result is not None,
            "tracking_info": (prediction_result.get("tracking_info")
                              if prediction_result else None),
        })

    # Print summary
    print(f"\n{CLIColors.BOLD}{'=' * 60}{CLIColors.ENDC}")
    print(f"{CLIColors.BOLD}Multi-Animal Batch Processing Complete{CLIColors.ENDC}")
    print(f"{CLIColors.BOLD}{'=' * 60}{CLIColors.ENDC}")

    successful = sum(1 for r in results if r["success"])
    print(f"\nSuccessful: {successful}/{total_jobs}")

    for r in results:
        status = (f"{CLIColors.OKGREEN}ok{CLIColors.ENDC}"
                  if r["success"]
                  else f"{CLIColors.FAIL}FAIL{CLIColors.ENDC}")
        print(f"  [{status}] {r['video_folder']}")
        if r["output_dir"]:
            print(f"       Output: {r['output_dir']}")
        if r.get("viz_output_dir"):
            print(f"       Visualization: {r['viz_output_dir']}")
        if r.get("tracking_info"):
            ti = r["tracking_info"]
            print(f"       Swaps corrected: {ti['swap_corrections']}, "
                  f"Body sizes: {ti['body_sizes']}")


if __name__ == "__main__":
    main()

'''
python tools/predict3D_multianimal.py \
  --project red_data_unified \
  --session /data2/users/eabe/datasets/Johnson_lab/courtship/Session0/2025_10_20_13_20_04 \
  --bouts-csv /data2/users/eabe/datasets/Johnson_lab/courtship/Session0/2025_10_20_13_20_04/courtship_bouts_unified_summary.csv \
  --bouts 29 \
  --out /data2/users/eabe/datasets/Johnson_lab/courtship/Session0/2025_10_20_13_20_04/Predictions_3D_V4_phase4 \
  --num-animals 2 --sam3-gpu 1 \
  --sam3-version sam3.1 --sam3-compile \
  --save-masks --save-clips

'''