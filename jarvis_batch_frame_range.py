#!/usr/bin/env python3
"""
Batch 3D Prediction Script for JARVIS-HybridNet

This script allows running JARVIS 3D predictions on multiple video folders
without using the interactive interface.

Usage:
    python jarvis_batch_predict3D.py --config batch_config.json --project rat24_2

    Or with individual folders:
    python jarvis_batch_predict3D.py --project rat24_2 \
        --video_folder /path/to/videos1 --calib_folder /path/to/calib1 \
        --video_folder /path/to/videos2 --calib_folder /path/to/calib2

Config file format (JSON):
{
    "jobs": [
        {
            "video_folder": "/path/to/videos",
            "calib_folder": "/path/to/calibration",
            "frame_start": 0,
            "frame_end": -1,
            "run_viz": false,
            "viz_cameras": ["Cam1", "Cam2"]
        },
        {
            "video_folder": "/path/to/other_videos",
            "calib_folder": "/path/to/other_calibration",
            "frame_file": "/path/to/frames.txt",
            "run_viz": false
        },
        ...
    ]
}

Frame file format (plain text, one range per line):
    # Comments start with '#'
    0 1000
    2500 4000
    6000 -1

    The output CSV will be continuous from frame 0 to the last frame,
    with NaN values for all frames not covered by any range.

Options:
    - frame_file: (optional) Path to a text file with frame ranges. Each line
                  has a start and end frame separated by whitespace or comma.
                  Use -1 as end frame to mean "until end of video". Overrides
                  frame_start/frame_end when provided.
    - run_viz: (optional, default: false) Create visualization videos after prediction
    - viz_cameras: (optional) List of camera names to create videos for. If not specified,
                   creates videos for all cameras.

Output is saved to the standard JARVIS locations:
    Predictions: JARVIS-HybridNet/projects/<project>/predictions/predictions3D/Predictions_3D_<timestamp>/
    Visualizations: JARVIS-HybridNet/projects/<project>/visualization/Videos_3D_<timestamp>/
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
import glob

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from ruamel.yaml import YAML
from tqdm import tqdm

from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis3D import JarvisPredictor3D
from jarvis.utils.paramClasses import Predict3DParams, CreateVideos3DParams
from jarvis.utils.reprojection import ReprojectionTool, load_reprojection_tools
from jarvis.visualization.create_videos3D import create_videos3D


class CLIColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def parse_frame_file(frame_file_path):
    """
    Parse a text file containing frame ranges (one per line).

    Each line should contain a start and end frame separated by whitespace
    or a comma. Lines starting with '#' are treated as comments.
    Empty lines are skipped. Use -1 as end frame to mean "until end of video".

    Example file contents:
        # Grooming bouts
        0 1000
        2500 4000
        6000 -1

    Args:
        frame_file_path: Path to the text file with frame ranges.

    Returns:
        List of [frame_start, frame_end] pairs, sorted by frame_start.
    """
    import re
    frame_ranges = []
    with open(frame_file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split on whitespace or comma
            parts = re.split(r'[\s,]+', line)
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    frame_ranges.append([start, end])
                except ValueError:
                    print(f"{CLIColors.WARNING}Warning: skipping invalid line "
                          f"{line_num} in {frame_file_path}: '{line}'{CLIColors.ENDC}")
            else:
                print(f"{CLIColors.WARNING}Warning: skipping invalid line "
                      f"{line_num} in {frame_file_path}: '{line}' "
                      f"(expected 2 values: start end){CLIColors.ENDC}")
    if not frame_ranges:
        print(f"{CLIColors.FAIL}Error: no valid frame ranges found in "
              f"{frame_file_path}{CLIColors.ENDC}")
    # Sort by start frame
    frame_ranges.sort(key=lambda x: x[0])
    return frame_ranges


def build_predict_set(frame_ranges, total_frames):
    """
    Build a set of frame numbers that should be predicted (not NaN-filled).

    Args:
        frame_ranges: List of [start, end] pairs. end=-1 means last frame.
        total_frames: Total number of frames in the video.

    Returns:
        (predict_set, global_start, global_end): A set of frame indices to
        predict, and the overall start/end of the continuous output.
    """
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
    """Load reprojection tool from a custom calibration folder."""
    reproTools = load_reprojection_tools(
        cfg, cameras_to_use=cameras_to_use, device=device
    )

    if calib_folder is not None and os.path.isdir(calib_folder):
        # Load calibration from the specified folder
        dataset_dir = os.path.join(
            cfg.PARENT_DIR,
            cfg.DATASET.DATASET_ROOT_DIR,
            cfg.DATASET.DATASET_3D,
        )
        dataset_json_path = os.path.join(dataset_dir, "annotations", "instances_val.json")

        with open(dataset_json_path) as dataset_json:
            data = json.load(dataset_json)

        calibPaths = {}
        calibParams = list(data["calibrations"].keys())[0]
        for cam in data["calibrations"][calibParams]:
            if cameras_to_use is None or cam in cameras_to_use:
                calibPaths[cam] = data["calibrations"][calibParams][cam].split("/")[-1]

        # Handle special camera naming for rat_pose projects
        if cfg["PROJECT_NAME"] in ["rat_pose", "rat24_2"]:
            ordered_serial = [
                "2002496", "2002483", "2002488", "2002480", "2002489",
                "2002485", "2002490", "2002492", "2002479", "2002494",
                "2002495", "2002482", "2002481", "2002491", "2002493",
                "2002484", "710038",
            ]
            calibPaths_new = {}
            for cam_name, _ in calibPaths.items():
                cam_order = int(cam_name[3:])
                cam_new_name = "Cam" + ordered_serial[cam_order]
                calibPaths_new[cam_new_name] = cam_new_name + ".yaml"
            calibPaths = calibPaths_new

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
        print(f"Converting: {input_file_name}")
        fs = cv2.FileStorage(input_file_name, cv2.FILE_STORAGE_READ)

        # Check if this is already in JARVIS format
        intrinsic_node = fs.getNode("intrinsicMatrix")
        if not intrinsic_node.empty():
            # Already in JARVIS format, just copy
            import shutil
            output_filename = os.path.join(output_folder, f"{cam_names[idx]}.yaml")
            shutil.copy(input_file_name, output_filename)
            continue

        # Convert from other format
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
    """Seek to a specific frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


def read_images(cap, slice_idx, imgs):
    """Read a frame from a video capture."""
    ret, img = cap.read()
    if ret and img is not None:
        imgs[slice_idx] = img.astype(np.uint8)


def create_header(writer, cfg):
    """Create CSV header with keypoint names (matching JARVIS format - no frame column)."""
    joints = list(
        itertools.chain.from_iterable(
            itertools.repeat(x, 4) for x in cfg.KEYPOINT_NAMES
        )
    )
    coords = ["x", "y", "z", "confidence"] * len(cfg.KEYPOINT_NAMES)
    writer.writerow(joints)
    writer.writerow(coords)


def create_info_file(output_dir, recording_path, dataset_name, frame_start, number_frames):
    """Create info.yaml file with prediction metadata (matching JARVIS format)."""
    with open(os.path.join(output_dir, "info.yaml"), "w") as file:
        yaml = YAML()
        yaml.dump(
            {
                "recording_path": recording_path,
                "dataset_name": dataset_name,
                "frame_start": frame_start,
                "number_frames": number_frames,
            },
            file,
        )


def run_prediction(
    project_name,
    video_folder,
    calib_folder,
    frame_start=0,
    frame_end=-1,
    frame_ranges=None,
    cameras_to_use=None,
    weights_center_detect="latest",
    weights_hybridnet="latest",
    trt_mode="off",
    n_jobs=17,
):
    """
    Run 3D prediction on a single video folder.

    Args:
        project_name: Name of the JARVIS project
        video_folder: Path to folder containing video files
        calib_folder: Path to folder containing calibration files
        frame_start: Starting frame number (used when frame_ranges is None)
        frame_end: Ending frame number, -1 for all (used when frame_ranges is None)
        frame_ranges: Optional list of [start, end] pairs from a frame file.
                      When provided, output is continuous from the earliest start
                      to the latest end, with NaN rows for frames outside the ranges.
        cameras_to_use: Optional list of camera names to use
        weights_center_detect: Path to center detection weights or 'latest'
        weights_hybridnet: Path to hybridnet weights or 'latest'
        trt_mode: TensorRT mode ('off', 'on')
        n_jobs: Number of parallel jobs for reading frames

    Returns:
        Dict with output info, or None on failure.
    """
    print(f"\n{CLIColors.HEADER}{'='*60}{CLIColors.ENDC}")
    print(f"{CLIColors.OKBLUE}Processing: {video_folder}{CLIColors.ENDC}")
    print(f"{CLIColors.OKCYAN}Calibration: {calib_folder}{CLIColors.ENDC}")
    print(f"{CLIColors.HEADER}{'='*60}{CLIColors.ENDC}\n")

    # Load project
    project = ProjectManager()
    if not project.load(project_name):
        print(f"{CLIColors.FAIL}Could not load project: {project_name}!{CLIColors.ENDC}")
        return None
    cfg = project.cfg

    if cameras_to_use is not None:
        cfg.HYBRIDNET.NUM_CAMERAS = len(cameras_to_use)

    # Check if calibration needs conversion
    jarvis_calib_folder = os.path.join(
        os.path.dirname(calib_folder),
        os.path.basename(calib_folder) + "_jarvis"
    )

    # Check if calibration files are in JARVIS format
    calib_files = glob.glob(os.path.join(calib_folder, "*.yaml"))
    if calib_files:
        fs = cv2.FileStorage(calib_files[0], cv2.FILE_STORAGE_READ)
        intrinsic_node = fs.getNode("intrinsicMatrix")
        projection_node = fs.getNode("projectionMatrix")
        fs.release()

        if intrinsic_node.empty() and projection_node.empty():
            # Need to convert calibration files
            if not os.path.isdir(jarvis_calib_folder):
                print(f"{CLIColors.WARNING}Converting calibration files to JARVIS format...{CLIColors.ENDC}")
                os.makedirs(jarvis_calib_folder, exist_ok=True)
                convert2jarviscalib(calib_folder, jarvis_calib_folder)
            calib_folder = jarvis_calib_folder

    # Initialize predictor
    jarvisPredictor = JarvisPredictor3D(
        cfg,
        weights_center_detect,
        weights_hybridnet,
        trt_mode,
    )

    # Load reprojection tool
    reproTool = get_repro_tool(cfg, calib_folder, cameras_to_use=cameras_to_use)
    if reproTool is None:
        return None

    # Setup output directory in standard JARVIS location:
    # JARVIS-HybridNet/projects/<project_name>/predictions/predictions3D/Predictions_3D_<timestamp>/
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
    caps, img_size = create_video_reader(video_paths, frame_start)

    # Determine frame range
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_ranges is not None and len(frame_ranges) > 0:
        # --- Frame-file mode: predict only specified ranges, NaN for gaps ---
        predict_set, actual_frame_start, actual_frame_end = build_predict_set(
            frame_ranges, total_frames
        )
        num_frames = actual_frame_end - actual_frame_start + 1
        num_predict = len(predict_set)
        num_nan = num_frames - num_predict
        print(f"Frame-file mode: frames {actual_frame_start} to {actual_frame_end} "
              f"({num_frames} total, {num_predict} predicted, {num_nan} NaN-filled)")
        for s, e in frame_ranges:
            e_resolved = total_frames - 1 if e == -1 else min(e, total_frames - 1)
            print(f"  Range: {s} - {e_resolved}")
    else:
        # --- Standard mode: single contiguous range ---
        if frame_end == -1:
            frame_end = total_frames - 1

        actual_frame_start = frame_start
        actual_frame_end = min(frame_end, total_frames - 1)
        num_frames = actual_frame_end - actual_frame_start + 1
        predict_set = None  # means predict everything
        print(f"Processing frames {actual_frame_start} to {actual_frame_end} ({num_frames} frames)")

    # Create info file (matching JARVIS format)
    create_info_file(output_dir, video_folder, calib_folder, actual_frame_start, num_frames)

    # Open CSV file
    csvfile = open(os.path.join(output_dir, "data3D.csv"), "w", newline="")
    writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write header if keypoint names are defined
    if len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS:
        create_header(writer, cfg)

    # Pre-allocate NaN row for gap frames
    nan_row = ["NaN"] * (cfg.KEYPOINTDETECT.NUM_JOINTS * 4)

    # Pre-allocate image buffer
    imgs_orig = np.zeros((len(caps), img_size[1], img_size[0], 3)).astype(np.uint8)

    # Seek to start frame
    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(seek)(cap, actual_frame_start) for cap in caps
    )

    # Process frames
    for frame_num in tqdm(range(actual_frame_start, actual_frame_end + 1), desc="Predicting"):
        # Check if this frame should be predicted or NaN-filled
        if predict_set is not None and frame_num not in predict_set:
            # Write NaN row and advance video readers by one frame
            writer.writerow(nan_row)
            Parallel(n_jobs=n_jobs, require="sharedmem")(
                delayed(read_images)(cap, slice_idx, imgs_orig)
                for slice_idx, cap in enumerate(caps)
            )
            continue

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

        # Run prediction
        points3D_net, confidences = jarvisPredictor(
            imgs,
            reproTool.cameraMatrices.cuda(),
        )

        # Write results (no frame column - matches JARVIS format)
        if points3D_net is not None:
            row = []
            for point, conf in zip(
                points3D_net.squeeze(), confidences.squeeze().cpu().numpy()
            ):
                row = row + point.tolist() + [conf]
            writer.writerow(row)
        else:
            row = []
            for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS * 4):
                row = row + ["NaN"]
            writer.writerow(row)

    # Cleanup
    for cap in caps:
        cap.release()
    csvfile.close()

    print(f"{CLIColors.OKGREEN}Completed: {output_dir}{CLIColors.ENDC}\n")

    # Return info needed for visualization
    return {
        "output_dir": output_dir,
        "data_csv": os.path.join(output_dir, "data3D.csv"),
        "video_folder": video_folder,
        "calib_folder": calib_folder,
        "frame_start": actual_frame_start,
        "number_frames": num_frames,
    }


def run_visualization(
    project_name,
    video_folder,
    calib_folder,
    data_csv,
    frame_start=0,
    number_frames=-1,
    viz_cameras=None,
):
    """
    Create 3D visualization videos.

    Args:
        project_name: Name of the JARVIS project
        video_folder: Path to folder containing video files
        calib_folder: Path to folder containing calibration files
        data_csv: Path to the predictions CSV file
        frame_start: Starting frame number
        number_frames: Number of frames to visualize (-1 for all)
        viz_cameras: List of camera names to create videos for (None for all)

    Returns:
        Path to visualization output directory
    """
    print(f"\n{CLIColors.HEADER}Creating visualization videos...{CLIColors.ENDC}")

    # Load project to get camera list if viz_cameras not specified
    project = ProjectManager()
    if not project.load(project_name):
        print(f"{CLIColors.FAIL}Could not load project: {project_name}!{CLIColors.ENDC}")
        return None
    cfg = project.cfg

    # If no specific cameras specified, we need to get the camera list from the repro tool
    if viz_cameras is None or len(viz_cameras) == 0:
        # Get all available cameras from the video folder
        videos = os.listdir(video_folder)
        viz_cameras = [v.split(".")[0] for v in videos if v.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"Creating videos for all cameras: {viz_cameras}")

    # Create params for visualization
    params = CreateVideos3DParams(
        project_name=project_name,
        recording_path=video_folder,
        data_csv=data_csv,
        frame_start=frame_start,
        number_frames=number_frames,
        video_cam_list=viz_cameras,
    )
    params.dataset_name = calib_folder

    # Run visualization
    create_videos3D(params)

    print(f"{CLIColors.OKGREEN}Visualization completed: {params.output_dir}{CLIColors.ENDC}\n")
    return params.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Batch 3D Prediction for JARVIS-HybridNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "-p", "--project",
        type=str,
        required=True,
        help="Name of the JARVIS project to use"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to JSON config file with batch jobs"
    )

    parser.add_argument(
        "-v", "--video_folder",
        type=str,
        action="append",
        help="Path to video folder (can be specified multiple times)"
    )

    parser.add_argument(
        "-k", "--calib_folder",
        type=str,
        action="append",
        help="Path to calibration folder (must match video_folder count)"
    )

    parser.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="Starting frame (default: 0)"
    )

    parser.add_argument(
        "--frame_end",
        type=int,
        default=-1,
        help="Ending frame (default: -1 for all frames)"
    )

    parser.add_argument(
        "--frame_file",
        type=str,
        help="Path to a text file with frame ranges (one 'start end' pair per "
             "line). Frames outside the ranges are filled with NaN in the output. "
             "Overrides --frame_start/--frame_end when provided."
    )

    parser.add_argument(
        "--weights_center",
        type=str,
        default="latest",
        help="Weights for CenterDetect (default: 'latest')"
    )

    parser.add_argument(
        "--weights_hybridnet",
        type=str,
        default="latest",
        help="Weights for HybridNet (default: 'latest')"
    )

    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        help="Optional list of camera names to use"
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=17,
        help="Number of parallel jobs for frame reading (default: 17)"
    )

    parser.add_argument(
        "--trt_mode",
        type=str,
        default="off",
        choices=["off", "on"],
        help="TensorRT mode (default: 'off')"
    )

    parser.add_argument(
        "--run_viz",
        action="store_true",
        help="Create visualization videos after prediction (for command line jobs)"
    )

    parser.add_argument(
        "--viz_cameras",
        type=str,
        nargs="+",
        help="Camera names to create visualization videos for (default: all cameras)"
    )

    args = parser.parse_args()

    # Build list of jobs
    jobs = []

    if args.config:
        # Load from config file
        with open(args.config, "r") as f:
            config = json.load(f)
        raw_jobs = config.get("jobs", [])

        # Resolve frame_file entries: parse the file and attach ranges to the job
        for job in raw_jobs:
            if "frame_file" in job and job["frame_file"]:
                job["frame_ranges"] = parse_frame_file(job["frame_file"])
                # Remove frame_start/frame_end since frame_ranges takes priority
                job.pop("frame_start", None)
                job.pop("frame_end", None)
                del job["frame_file"]
            jobs.append(job)

        print(f"Loaded {len(jobs)} jobs from config file")

    if args.video_folder and args.calib_folder:
        # Add jobs from command line arguments
        if len(args.video_folder) != len(args.calib_folder):
            print(f"{CLIColors.FAIL}Error: Number of video folders must match calibration folders{CLIColors.ENDC}")
            sys.exit(1)

        for video, calib in zip(args.video_folder, args.calib_folder):
            job_entry = {
                "video_folder": video,
                "calib_folder": calib,
                "run_viz": args.run_viz,
                "viz_cameras": args.viz_cameras,
            }
            if args.frame_file:
                job_entry["frame_ranges"] = parse_frame_file(args.frame_file)
            else:
                job_entry["frame_start"] = args.frame_start
                job_entry["frame_end"] = args.frame_end
            jobs.append(job_entry)

    if not jobs:
        print(f"{CLIColors.FAIL}Error: No jobs specified. Use --config or --video_folder/--calib_folder{CLIColors.ENDC}")
        parser.print_help()
        sys.exit(1)

    # Process all jobs
    results = []
    total_jobs = len(jobs)

    print(f"\n{CLIColors.BOLD}Starting batch prediction with {total_jobs} job(s){CLIColors.ENDC}\n")

    for i, job in enumerate(jobs):
        print(f"\n{CLIColors.BOLD}[Job {i+1}/{total_jobs}]{CLIColors.ENDC}")

        prediction_result = run_prediction(
            project_name=args.project,
            video_folder=job["video_folder"],
            calib_folder=job["calib_folder"],
            frame_start=job.get("frame_start", args.frame_start),
            frame_end=job.get("frame_end", args.frame_end),
            frame_ranges=job.get("frame_ranges", None),
            cameras_to_use=args.cameras,
            weights_center_detect=args.weights_center,
            weights_hybridnet=args.weights_hybridnet,
            trt_mode=args.trt_mode,
            n_jobs=args.n_jobs,
        )

        viz_output_dir = None

        # Run visualization if requested
        if prediction_result is not None and job.get("run_viz", False):
            viz_cameras = job.get("viz_cameras", None)
            viz_output_dir = run_visualization(
                project_name=args.project,
                video_folder=prediction_result["video_folder"],
                calib_folder=prediction_result["calib_folder"],
                data_csv=prediction_result["data_csv"],
                frame_start=prediction_result["frame_start"],
                number_frames=prediction_result["number_frames"],
                viz_cameras=viz_cameras,
            )

        results.append({
            "video_folder": job["video_folder"],
            "output_dir": prediction_result["output_dir"] if prediction_result else None,
            "viz_output_dir": viz_output_dir,
            "success": prediction_result is not None,
        })

    # Print summary
    print(f"\n{CLIColors.BOLD}{'='*60}{CLIColors.ENDC}")
    print(f"{CLIColors.BOLD}Batch Processing Complete{CLIColors.ENDC}")
    print(f"{CLIColors.BOLD}{'='*60}{CLIColors.ENDC}")

    successful = sum(1 for r in results if r["success"])
    print(f"\nSuccessful: {successful}/{total_jobs}")

    for r in results:
        status = f"{CLIColors.OKGREEN}✓{CLIColors.ENDC}" if r["success"] else f"{CLIColors.FAIL}✗{CLIColors.ENDC}"
        print(f"  {status} {r['video_folder']}")
        if r["output_dir"]:
            print(f"      Predictions: {r['output_dir']}")
        if r.get("viz_output_dir"):
            print(f"      Visualization: {r['viz_output_dir']}")


if __name__ == "__main__":
    main()
