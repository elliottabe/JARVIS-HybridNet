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
import glob
import itertools
import json
import os
import sys
import time

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


def parse_frame_file(frame_file_path):
    """Parse a text file containing frame ranges (one per line)."""
    import re
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
    use_sam3_mask=True,
    sam3_device="cuda",
    sam3_text_prompt="fly",
    sam3_constrain_keypoints=True,
    resume_dir=None,
):
    """
    Run multi-animal 3D prediction on a single video folder.

    Returns:
        Dict with output info, or None on failure.
    """
    print(f"\n{CLIColors.HEADER}{'=' * 60}{CLIColors.ENDC}")
    print(f"{CLIColors.OKBLUE}Processing: {video_folder}{CLIColors.ENDC}")
    print(f"{CLIColors.OKCYAN}Calibration: {calib_folder}{CLIColors.ENDC}")
    print(f"{CLIColors.OKCYAN}Num animals: {num_animals}{CLIColors.ENDC}")
    print(f"{CLIColors.HEADER}{'=' * 60}{CLIColors.ENDC}\n")

    # Load project
    project = ProjectManager()
    if not project.load(project_name):
        print(f"{CLIColors.FAIL}Could not load project: "
              f"{project_name}!{CLIColors.ENDC}")
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
                print(f"{CLIColors.WARNING}Converting calibration files to "
                      f"JARVIS format...{CLIColors.ENDC}")
                os.makedirs(jarvis_calib_folder, exist_ok=True)
                convert2jarviscalib(calib_folder, jarvis_calib_folder)
            calib_folder = jarvis_calib_folder

    # Initialize multi-animal predictor
    # SAM3 masking is handled externally via SAM3StreamingTracker (video
    # propagation), not inside the predictor. The predictor receives
    # precomputed_masks per frame when SAM3 is enabled.
    jarvisPredictor = JarvisMultiAnimalPredictor3D(
        cfg,
        num_animals=num_animals,
        suppression_radius=suppression_radius,
        mask_scale=mask_scale,
        weights_center_detect=weights_center_detect,
        weights_hybridnet=weights_hybridnet,
        trt_mode=trt_mode,
        use_sam3_mask=False,  # SAM3 handled externally via streaming tracker
        sam3_constrain_keypoints=sam3_constrain_keypoints,
    )

    # Initialize identity tracker
    tracker = MultiAnimalTracker(
        keypoint_names=cfg.KEYPOINT_NAMES,
        num_animals=num_animals,
    )

    # Load reprojection tool
    reproTool = get_repro_tool(cfg, calib_folder, cameras_to_use=cameras_to_use)
    if reproTool is None:
        return None

    # Setup output directory
    if resume_dir is not None:
        output_dir = resume_dir
        if not os.path.isdir(output_dir):
            print(f"{CLIColors.FAIL}Resume directory does not exist: "
                  f"{output_dir}{CLIColors.ENDC}")
            return None
        print(f"{CLIColors.OKGREEN}Resuming from: {output_dir}{CLIColors.ENDC}")
    else:
        output_dir = os.path.join(
            project.parent_dir,
            cfg.PROJECTS_ROOT_PATH,
            project_name,
            "predictions",
            "predictions3D",
            f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}',
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"{CLIColors.OKGREEN}Output directory: {output_dir}{CLIColors.ENDC}")

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
        print(f"Bout mode: {len(resolved_bouts)} bouts, "
              f"{total_bout_frames} frames to predict")
    else:
        # Contiguous mode: single range
        if frame_end == -1:
            frame_end = total_frames - 1
        actual_start = frame_start
        actual_end = min(frame_end, total_frames - 1)
        resolved_bouts = [(actual_start, actual_end)]
        total_bout_frames = actual_end - actual_start + 1
        print(f"Processing frames {actual_start} to {actual_end} "
              f"({total_bout_frames} frames)")

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
            print(f"{CLIColors.OKCYAN}Found {frames_completed} frames "
                  f"already completed{CLIColors.ENDC}")

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
            print(f"{CLIColors.OKCYAN}Skipping {bouts_to_skip} completed "
                  f"bout(s){CLIColors.ENDC}")
        if resume_frame_offset > 0:
            print(f"{CLIColors.OKCYAN}Resuming bout {bouts_to_skip + 1} "
                  f"from frame offset {resume_frame_offset}{CLIColors.ENDC}")

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
    sam3_tracker = None
    if use_sam3_mask:
        gpu_id = 0
        if ':' in sam3_device:
            gpu_id = int(sam3_device.split(':')[1])
        from jarvis.prediction.sam3_video_tracker import SAM3StreamingTracker
        sam3_tracker = SAM3StreamingTracker(
            gpu_id=gpu_id,
            text_prompt=sam3_text_prompt,
        )

    # Process each bout
    for bout_idx, (bout_start, bout_end) in enumerate(resolved_bouts):
        bout_len = bout_end - bout_start + 1

        # Skip fully completed bouts on resume
        if bout_idx < bouts_to_skip:
            print(f"Bout {bout_idx + 1}/{len(resolved_bouts)} "
                  f"[{bout_start}-{bout_end}]: skipped (already complete)")
            continue

        # Determine start offset within this bout (for partial resume)
        start_offset = 0
        if bout_idx == bouts_to_skip and resume_frame_offset > 0:
            start_offset = resume_frame_offset

        desc = (f"Bout {bout_idx + 1}/{len(resolved_bouts)} "
                f"[{bout_start}-{bout_end}]")

        # Set up SAM3 streaming propagation for this bout
        bout_sam3 = None  # per-bout tracker reference (None = no SAM3)
        if sam3_tracker is not None:
            print(f"\n{CLIColors.OKCYAN}Starting SAM3 video propagation for "
                  f"bout {bout_idx + 1} ({bout_len} frames, "
                  f"{len(video_paths)} cameras)...{CLIColors.ENDC}")
            sam3_tracker.start_bout(video_paths, bout_start, bout_len)
            detection_ok = sam3_tracker.detect_frame0(
                num_animals=num_animals, max_retry_frames=10)
            if detection_ok:
                sam3_tracker.assign_identities(
                    reproTool, num_animals=num_animals)
                sam3_tracker.start_propagation()
                bout_sam3 = sam3_tracker
                print(f"{CLIColors.OKGREEN}SAM3 streaming propagation "
                      f"initialized (init frame: "
                      f"{sam3_tracker.init_frame}).{CLIColors.ENDC}")
            else:
                print(f"{CLIColors.WARNING}SAM3 detection failed for this "
                      f"bout — falling back to mask-and-redetect."
                      f"{CLIColors.ENDC}")
                sam3_tracker.close()

        # Seek all cameras to bout start (+ offset for partial resume)
        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(seek)(cap, bout_start + start_offset) for cap in caps
        )

        for frame_num in tqdm(range(start_offset, bout_len), desc=desc):
            # Read images in parallel
            Parallel(n_jobs=n_jobs, require="sharedmem")(
                delayed(read_images)(cap, slice_idx, imgs_orig)
                for slice_idx, cap in enumerate(caps)
            )

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
                if frame_num == bout_sam3.init_frame:
                    frame_masks = bout_sam3.get_frame0_masks(
                        num_animals=num_animals)
                elif frame_num > bout_sam3.init_frame:
                    frame_masks = bout_sam3.get_next_frame(
                        num_animals=num_animals)
                # Frames before init_frame have no masks (fallback path)

            # Run multi-animal prediction
            detections = jarvisPredictor(
                imgs, camera_matrices_cuda,
                precomputed_masks=frame_masks,
            )

            # Assign identities via tracker
            assignments = tracker.assign_identities(detections)

            # Write results to per-animal CSVs
            for fly_id in fly_ids:
                det = assignments.get(fly_id)
                if det is not None and det.get('points3D') is not None:
                    points3D = det['points3D']
                    confidences = det['confidences']
                    if points3D.dim() == 1:
                        for fid in fly_ids:
                            writers[fid].writerow(nan_row)
                        break
                    row = []
                    for point, conf in zip(
                        points3D.squeeze(),
                        confidences.squeeze().cpu().numpy()
                    ):
                        row = row + point.tolist() + [conf]
                    writers[fly_id].writerow(row)
                else:
                    writers[fly_id].writerow(nan_row)

        # Clean up streaming tracker for this bout
        if bout_sam3 is not None:
            bout_sam3.close()

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

    print(f"\n{CLIColors.OKGREEN}Tracking summary:{CLIColors.ENDC}")
    print(f"  Frames processed: {tracking_info['frame_count']}")
    print(f"  Swap corrections: {tracking_info['swap_corrections']}")
    print(f"  Body sizes (EMA): {tracking_info['body_sizes']}")
    print(f"{CLIColors.OKGREEN}Completed: {output_dir}{CLIColors.ENDC}\n")

    return {
        "output_dir": output_dir,
        "data_csvs": {
            fly_id: os.path.join(output_dir, f'data3D_{fly_id}.csv')
            for fly_id in fly_ids
        },
        "video_folder": video_folder,
        "calib_folder": calib_folder,
        "bouts": resolved_bouts,
        "total_bout_frames": total_bout_frames,
        "tracking_info": tracking_info,
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
        "--no_sam3_mask", action="store_true",
        help="Disable SAM3 per-frame segmentation and fall back to "
             "mask-and-redetect approach. SAM3 is enabled by default."
    )
    parser.add_argument(
        "--sam3_device", type=str, default="cuda",
        help="Device for SAM3 model (default: 'cuda'). Use 'cuda:1' for "
             "second GPU if memory is tight."
    )
    parser.add_argument(
        "--sam3_text_prompt", type=str, default="fly",
        help="Text prompt for SAM3 segmentation (default: 'fly')"
    )
    parser.add_argument(
        "--no_sam3_constrain_keypoints", action="store_true",
        help="Disable SAM3 keypoint constraining (only use SAM3 for masking, "
             "not for constraining keypoint detection)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing output directory to resume from. "
             "Skips already-completed bouts based on CSV row counts."
    )

    args = parser.parse_args()

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

        prediction_result = run_prediction(
            project_name=args.project,
            video_folder=job["video_folder"],
            calib_folder=job["calib_folder"],
            num_animals=args.num_animals,
            suppression_radius=args.suppression_radius,
            mask_scale=args.mask_scale,
            frame_start=job.get("frame_start", args.frame_start),
            frame_end=job.get("frame_end", args.frame_end),
            frame_ranges=job.get("frame_ranges", None),
            cameras_to_use=args.cameras,
            weights_center_detect=args.weights_center,
            weights_hybridnet=args.weights_hybridnet,
            trt_mode=args.trt_mode,
            n_jobs=args.n_jobs,
            use_sam3_mask=not args.no_sam3_mask,
            sam3_device=args.sam3_device,
            sam3_text_prompt=args.sam3_text_prompt,
            sam3_constrain_keypoints=not args.no_sam3_constrain_keypoints,
            resume_dir=args.resume,
        )

        viz_output_dir = None
        if prediction_result is not None and (
            args.run_viz or job.get("run_viz", False)
        ):
            viz_cameras = (args.viz_cameras
                           or job.get("viz_cameras", None))
            bouts = prediction_result.get("bouts", [(0, -1)])
            data_csvs = prediction_result["data_csvs"]

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
