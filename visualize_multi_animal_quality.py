#!/usr/bin/env python3
"""Visualize multi-animal prediction quality and identify good bouts.

Scans prediction CSVs for segments where both flies are on the ground with
good tracking, generates preview images and a summary CSV.

Usage:
    # Single recording with explicit paths:
    python visualize_multi_animal_quality.py \
        --pred_dir /path/to/Predictions_3D_XXXX \
        --video_dir /path/to/video_folder

    # Batch mode via JSON config:
    python visualize_multi_animal_quality.py --config batch_config.json

Config JSON format:
    {
        "jobs": [
            {
                "pred_dir": "/path/to/Predictions_3D_XXXX",
                "video_dir": "/path/to/video_folder"
            },
            ...
        ],
        "cameras": ["Cam2012630", "Cam2012855"],
        "window_size": 500,
        "min_both_pct": 0.8,
        "min_conf": 0.3,
        "max_z_std": 5.0,
        "conf_threshold": 0.1,
        "top_k": 5
    }
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np


# -- Keypoint / skeleton definitions ------------------------------------------

KEYPOINT_NAMES = [
    'Antenna_Base', 'EyeL', 'EyeR', 'Scutellum', 'Abd_A4', 'Abd_tip',
    'WingL_base', 'WingL_V12', 'WingL_V13',
    'T1L_ThxCx', 'T1L_Tro', 'T1L_FeTi', 'T1L_TiTa', 'T1L_TaT1',
    'T1L_TaT3', 'T1L_TaTip',
    'T2L_Tro', 'T2L_FeTi', 'T2L_TiTa', 'T2L_TaT1', 'T2L_TaT3',
    'T2L_TaTip',
    'T3L_Tro', 'T3L_FeTi', 'T3L_TiTa', 'T3L_TaT1', 'T3L_TaT3',
    'T3L_TaTip',
    'WingR_base', 'WingR_V12', 'WingR_V13',
    'T1R_ThxCx', 'T1R_Tro', 'T1R_FeTi', 'T1R_TiTa', 'T1R_TaT1',
    'T1R_TaT3', 'T1R_TaTip',
    'T2R_Tro', 'T2R_FeTi', 'T2R_TiTa', 'T2R_TaT1', 'T2R_TaT3',
    'T2R_TaTip',
    'T3R_Tro', 'T3R_FeTi', 'T3R_TiTa', 'T3R_TaT1', 'T3R_TaT3',
    'T3R_TaTip',
]

SKELETON = [
    ('Antenna_Base', 'Scutellum'), ('Scutellum', 'Abd_A4'),
    ('Abd_A4', 'Abd_tip'),
    ('Antenna_Base', 'EyeL'), ('Antenna_Base', 'EyeR'),
    ('Scutellum', 'WingL_base'), ('WingL_base', 'WingL_V12'),
    ('WingL_V12', 'WingL_V13'),
    ('Scutellum', 'WingR_base'), ('WingR_base', 'WingR_V12'),
    ('WingR_V12', 'WingR_V13'),
    ('T1L_ThxCx', 'T1L_Tro'), ('T1L_Tro', 'T1L_FeTi'),
    ('T1L_FeTi', 'T1L_TiTa'), ('T1L_TiTa', 'T1L_TaT1'),
    ('T1L_TaT1', 'T1L_TaT3'), ('T1L_TaT3', 'T1L_TaTip'),
    ('T2L_Tro', 'T2L_FeTi'), ('T2L_FeTi', 'T2L_TiTa'),
    ('T2L_TiTa', 'T2L_TaT1'), ('T2L_TaT1', 'T2L_TaT3'),
    ('T2L_TaT3', 'T2L_TaTip'),
    ('T3L_Tro', 'T3L_FeTi'), ('T3L_FeTi', 'T3L_TiTa'),
    ('T3L_TiTa', 'T3L_TaT1'), ('T3L_TaT1', 'T3L_TaT3'),
    ('T3L_TaT3', 'T3L_TaTip'),
    ('T1R_ThxCx', 'T1R_Tro'), ('T1R_Tro', 'T1R_FeTi'),
    ('T1R_FeTi', 'T1R_TiTa'), ('T1R_TiTa', 'T1R_TaT1'),
    ('T1R_TaT1', 'T1R_TaT3'), ('T1R_TaT3', 'T1R_TaTip'),
    ('T2R_Tro', 'T2R_FeTi'), ('T2R_FeTi', 'T2R_TiTa'),
    ('T2R_TiTa', 'T2R_TaT1'), ('T2R_TaT1', 'T2R_TaT3'),
    ('T2R_TaT3', 'T2R_TaTip'),
    ('T3R_Tro', 'T3R_FeTi'), ('T3R_FeTi', 'T3R_TiTa'),
    ('T3R_TiTa', 'T3R_TaT1'), ('T3R_TaT1', 'T3R_TaT3'),
    ('T3R_TaT3', 'T3R_TaTip'),
]

KP_IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
SKEL_IDX = [(KP_IDX[a], KP_IDX[b]) for a, b in SKELETON
            if a in KP_IDX and b in KP_IDX]

# Indices for key columns (each kp = 4 cols: x, y, z, conf)
AB_XYZ = slice(0, 3)       # Antenna_Base xyz
AB_CONF = 3                # Antenna_Base confidence
SCUT_Z = 4 * 3 + 2        # Scutellum z (kp idx 3, col 14)
AT_XYZ = slice(4 * 5, 4 * 5 + 3)  # Abd_tip xyz (kp idx 5, cols 20-22)

FLY_COLORS = {
    'fly0': (0, 100, 255),   # orange (BGR)
    'fly1': (255, 100, 0),   # blue  (BGR)
}


# -- I/O helpers ---------------------------------------------------------------

def load_dlt_matrix(calib_dir, cam_name):
    yaml_path = os.path.join(calib_dir, f'{cam_name}.yaml')
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    P = fs.getNode('projectionMatrix').mat()
    fs.release()
    return P


def project_3d_to_2d(pts3d, P):
    N = pts3d.shape[0]
    pts_h = np.hstack([pts3d, np.ones((N, 1))])
    proj = (P @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def load_csv_columns(csv_path, n_header=2):
    """Load full CSV data (all columns). Returns (n_frames, n_cols) array."""
    return np.genfromtxt(csv_path, delimiter=',', skip_header=n_header)


def load_csv_rows(csv_path, start_row, n_rows, n_header=2):
    """Load a specific row range from a CSV. Returns (n_rows, n_kp, 4) array."""
    rows = []
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i < n_header:
                continue
            data_idx = i - n_header
            if data_idx >= start_row + n_rows:
                break
            if data_idx >= start_row:
                vals = line.strip().split(',')
                row = []
                for v in vals:
                    try:
                        row.append(float(v) if v != 'NaN' else float('nan'))
                    except ValueError:
                        row.append(float('nan'))
                rows.append(row)
    if not rows:
        return np.zeros((0, len(KEYPOINT_NAMES), 4))
    arr = np.array(rows)
    n_kp = arr.shape[1] // 4
    return arr.reshape(len(rows), n_kp, 4)


def detect_cameras(video_dir):
    """Detect available camera names from video files."""
    cams = []
    for f in sorted(os.listdir(video_dir)):
        if f.endswith(('.mp4', '.avi')) and not f.startswith('.'):
            cam = os.path.splitext(f)[0]
            cams.append(cam)
    return cams


def get_video_ext(video_dir):
    """Return the video extension used in this folder."""
    for f in os.listdir(video_dir):
        if f.endswith('.mp4'):
            return '.mp4'
        if f.endswith('.avi'):
            return '.avi'
    return '.mp4'


# -- Quality scoring -----------------------------------------------------------

def find_good_segments(pred_dir, window=500, step=500,
                       min_both_pct=0.8, min_conf=0.3, max_z_std=5.0):
    """Scan prediction CSVs and score windows for quality.

    A "good" segment has:
      - Both flies detected in >=min_both_pct of frames
      - Mean confidence >= min_conf for both flies
      - Low z-variance (both on ground, not flying): z_std <= max_z_std

    Returns list of dicts sorted by quality score (best first).
    """
    f0_path = os.path.join(pred_dir, 'data3D_fly0.csv')
    f1_path = os.path.join(pred_dir, 'data3D_fly1.csv')

    print("  Loading CSVs for quality scan...")
    data_f0 = load_csv_columns(f0_path)
    data_f1 = load_csv_columns(f1_path)
    n_frames = len(data_f0)
    print(f"  {n_frames} frames loaded")

    segments = []
    for start in range(0, n_frames - window, step):
        end = start + window
        w0 = data_f0[start:end]
        w1 = data_f1[start:end]

        # Both flies valid (Antenna_Base not NaN)
        f0_valid = ~np.isnan(w0[:, 0])
        f1_valid = ~np.isnan(w1[:, 0])
        both_valid = f0_valid & f1_valid
        pct_both = both_valid.sum() / window

        if pct_both < min_both_pct:
            continue

        # Mean confidence (Antenna_Base conf = col 3)
        f0_conf = np.nanmean(w0[f0_valid, AB_CONF])
        f1_conf = np.nanmean(w1[f1_valid, AB_CONF])

        if f0_conf < min_conf or f1_conf < min_conf:
            continue

        # Z stability (Scutellum z)
        f0_z = w0[f0_valid, SCUT_Z]
        f1_z = w1[f1_valid, SCUT_Z]
        f0_z_std = np.nanstd(f0_z)
        f1_z_std = np.nanstd(f1_z)

        if f0_z_std > max_z_std or f1_z_std > max_z_std:
            continue

        # Body length (Antenna_Base to Abd_tip)
        f0_bl = np.nanmean(np.linalg.norm(
            w0[f0_valid, AB_XYZ] - w0[f0_valid, AT_XYZ], axis=1))
        f1_bl = np.nanmean(np.linalg.norm(
            w1[f1_valid, AB_XYZ] - w1[f1_valid, AT_XYZ], axis=1))

        # Inter-fly distance
        dist = np.nanmean(np.linalg.norm(
            w0[both_valid, AB_XYZ] - w1[both_valid, AB_XYZ], axis=1))

        # Quality score: higher = better
        score = pct_both * (f0_conf + f1_conf) / (1 + f0_z_std + f1_z_std)

        segments.append({
            'start': start,
            'end': end,
            'pct_both': pct_both,
            'f0_z_std': f0_z_std,
            'f1_z_std': f1_z_std,
            'f0_conf': f0_conf,
            'f1_conf': f1_conf,
            'f0_bl': f0_bl,
            'f1_bl': f1_bl,
            'dist': dist,
            'score': score,
        })

    segments.sort(key=lambda x: x['score'], reverse=True)
    return segments


def merge_adjacent_segments(segments, window, gap_tolerance=1):
    """Merge overlapping or adjacent good segments into contiguous bouts."""
    if not segments:
        return []
    # Sort by start frame
    by_start = sorted(segments, key=lambda x: x['start'])
    bouts = []
    cur_start = by_start[0]['start']
    cur_end = by_start[0]['end']
    cur_scores = [by_start[0]['score']]

    for seg in by_start[1:]:
        if seg['start'] <= cur_end + gap_tolerance * window:
            cur_end = max(cur_end, seg['end'])
            cur_scores.append(seg['score'])
        else:
            bouts.append({
                'start': cur_start,
                'end': cur_end,
                'n_frames': cur_end - cur_start,
                'mean_score': np.mean(cur_scores),
            })
            cur_start = seg['start']
            cur_end = seg['end']
            cur_scores = [seg['score']]

    bouts.append({
        'start': cur_start,
        'end': cur_end,
        'n_frames': cur_end - cur_start,
        'mean_score': np.mean(cur_scores),
    })
    return bouts


# -- Drawing -------------------------------------------------------------------

def draw_skeleton(img, pts_2d, confs, color, conf_thresh=0.1):
    h, w = img.shape[:2]
    for ia, ib in SKEL_IDX:
        if confs[ia] > conf_thresh and confs[ib] > conf_thresh:
            xa, ya = int(pts_2d[ia, 0]), int(pts_2d[ia, 1])
            xb, yb = int(pts_2d[ib, 0]), int(pts_2d[ib, 1])
            if 0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h:
                cv2.line(img, (xa, ya), (xb, yb), color, 1, cv2.LINE_AA)
    for i in range(len(pts_2d)):
        if confs[i] > conf_thresh:
            x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)


def render_frame(frame, fly_data, frame_idx, P, conf_threshold=0.1):
    """Draw both flies' skeletons onto a video frame."""
    for fly_id in ['fly0', 'fly1']:
        kp = fly_data[fly_id][frame_idx]  # (n_kp, 4)
        pts3d = kp[:, :3]
        confs = kp[:, 3]
        valid = ~np.any(np.isnan(pts3d), axis=1)
        if valid.sum() < 2:
            continue
        pts2d = np.zeros((len(pts3d), 2))
        pts2d[valid] = project_3d_to_2d(pts3d[valid], P)
        draw_skeleton(frame, pts2d, confs * valid, FLY_COLORS[fly_id],
                      conf_threshold)
        # Label near Antenna_Base
        if valid[0] and confs[0] > conf_threshold:
            x, y = int(pts2d[0, 0]), int(pts2d[0, 1])
            cv2.putText(frame, fly_id, (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        FLY_COLORS[fly_id], 1, cv2.LINE_AA)
    return frame


# -- Main per-recording processing ---------------------------------------------

def process_recording(pred_dir, video_dir, cameras=None,
                      window=500, min_both_pct=0.8, min_conf=0.3,
                      max_z_std=5.0, conf_threshold=0.1, top_k=5):
    """Process one recording: find good segments, render previews, save summary.

    Returns dict with results for the summary CSV.
    """
    rec_name = os.path.basename(pred_dir)
    print(f"\n{'=' * 60}")
    print(f"Processing: {rec_name}")
    print(f"  Predictions: {pred_dir}")
    print(f"  Videos:      {video_dir}")

    # Load tracking info
    info_path = os.path.join(pred_dir, 'tracking_info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    # Find good segments
    all_segments = find_good_segments(
        pred_dir, window=window, min_both_pct=min_both_pct,
        min_conf=min_conf, max_z_std=max_z_std)

    # Merge into contiguous bouts
    good_bouts = merge_adjacent_segments(all_segments, window)

    n_good = len(good_bouts)
    total_good_frames = sum(b['n_frames'] for b in good_bouts)
    total_frames = info.get('frame_count', 0)
    pct_good = total_good_frames / total_frames * 100 if total_frames > 0 else 0

    print(f"  Found {len(all_segments)} good windows -> {n_good} contiguous bouts")
    print(f"  Good frames: {total_good_frames}/{total_frames} ({pct_good:.1f}%)")

    # Output directory: save alongside the videos
    out_dir = os.path.join(video_dir, 'quality_viz')
    os.makedirs(out_dir, exist_ok=True)

    # Detect cameras
    calib_dir = os.path.join(video_dir, 'calibration')
    if cameras is None:
        available = detect_cameras(video_dir)
        # Pick up to 3 cameras for preview images
        cameras = available[:3]
    vid_ext = get_video_ext(video_dir)

    has_video = len(cameras) > 0 and os.path.exists(
        os.path.join(video_dir, cameras[0] + vid_ext))

    # Load DLT matrices
    proj_matrices = {}
    if has_video and os.path.isdir(calib_dir):
        for cam in cameras:
            try:
                proj_matrices[cam] = load_dlt_matrix(calib_dir, cam)
            except Exception as e:
                print(f"  Warning: could not load DLT for {cam}: {e}")

    # Render preview images for top-K segments
    top_segments = all_segments[:top_k]
    rendered_previews = []

    if has_video and proj_matrices:
        print(f"  Rendering {len(top_segments)} preview images...")
        for si, seg in enumerate(top_segments):
            mid_frame = (seg['start'] + seg['end']) // 2

            # Load keypoint data for the middle frame
            fly_data = {}
            for fid in ['fly0', 'fly1']:
                fly_data[fid] = load_csv_rows(
                    os.path.join(pred_dir, f'data3D_{fid}.csv'),
                    mid_frame, 1)

            panels = []
            for cam in cameras:
                if cam not in proj_matrices:
                    continue
                cap = cv2.VideoCapture(
                    os.path.join(video_dir, cam + vid_ext))
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    continue

                render_frame(frame, fly_data, 0, proj_matrices[cam],
                             conf_threshold)

                # Annotations
                cv2.putText(frame, cam, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    f'Frame {mid_frame}  score={seg["score"]:.2f}  '
                    f'dist={seg["dist"]:.0f}mm',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
                panels.append(frame)

            if panels:
                composite = np.vstack(panels)
                img_name = f'{rec_name}_top{si}_f{mid_frame}.png'
                img_path = os.path.join(out_dir, img_name)
                cv2.imwrite(img_path, composite)
                rendered_previews.append(img_path)
                print(f"    {img_name}")
    else:
        print("  No video files found — skipping image rendering")

    # Save per-recording bout summary CSV
    bout_csv_path = os.path.join(out_dir, f'{rec_name}_good_bouts.csv')
    with open(bout_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bout_idx', 'start_frame', 'end_frame', 'n_frames', 'mean_score'
        ])
        for bi, bout in enumerate(good_bouts):
            writer.writerow([
                bi, bout['start'], bout['end'], bout['n_frames'],
                f"{bout['mean_score']:.4f}"
            ])
    print(f"  Bouts CSV: {bout_csv_path}")

    return {
        'recording': rec_name,
        'pred_dir': pred_dir,
        'video_dir': video_dir,
        'total_frames': total_frames,
        'n_good_bouts': n_good,
        'good_frames': total_good_frames,
        'pct_good': pct_good,
        'body_sizes': info.get('body_sizes', {}),
        'good_bouts': good_bouts,
        'previews': rendered_previews,
    }


# -- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-animal prediction quality and find good bouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--pred_dir', type=str,
        help='Path to a single Predictions_3D_* directory')
    parser.add_argument(
        '--video_dir', type=str,
        help='Path to the video folder (with calibration/ subfolder)')
    parser.add_argument(
        '--config', type=str,
        help='Path to JSON config for batch processing')
    parser.add_argument(
        '--cameras', type=str, nargs='+', default=None,
        help='Camera names to visualize (default: auto-detect up to 3)')
    parser.add_argument(
        '--window', type=int, default=500,
        help='Window size in frames for quality scoring (default: 500)')
    parser.add_argument(
        '--min_both_pct', type=float, default=0.8,
        help='Min fraction of frames with both flies detected (default: 0.8)')
    parser.add_argument(
        '--min_conf', type=float, default=0.3,
        help='Min mean Antenna_Base confidence per fly (default: 0.3)')
    parser.add_argument(
        '--max_z_std', type=float, default=5.0,
        help='Max Scutellum z std dev (filters out flying) (default: 5.0)')
    parser.add_argument(
        '--conf_threshold', type=float, default=0.1,
        help='Min confidence to draw a keypoint (default: 0.1)')
    parser.add_argument(
        '--top_k', type=int, default=5,
        help='Number of top segments to render previews for (default: 5)')
    parser.add_argument(
        '--output_csv', type=str, default=None,
        help='Path for the overall summary CSV (default: auto)')

    args = parser.parse_args()

    # Build job list
    jobs = []
    cfg = {}

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for job in cfg.get('jobs', []):
            jobs.append(job)
        print(f"Loaded {len(jobs)} jobs from {args.config}")

    if args.pred_dir and args.video_dir:
        jobs.append({
            'pred_dir': args.pred_dir,
            'video_dir': args.video_dir,
        })

    if not jobs:
        parser.print_help()
        sys.exit(1)

    # Override defaults with config values
    window = cfg.get('window_size', args.window)
    min_both_pct = cfg.get('min_both_pct', args.min_both_pct)
    min_conf = cfg.get('min_conf', args.min_conf)
    max_z_std = cfg.get('max_z_std', args.max_z_std)
    conf_threshold = cfg.get('conf_threshold', args.conf_threshold)
    top_k = cfg.get('top_k', args.top_k)
    cameras = cfg.get('cameras', args.cameras)

    # Process all recordings
    all_results = []
    for job in jobs:
        result = process_recording(
            pred_dir=job['pred_dir'],
            video_dir=job['video_dir'],
            cameras=cameras,
            window=window,
            min_both_pct=min_both_pct,
            min_conf=min_conf,
            max_z_std=max_z_std,
            conf_threshold=conf_threshold,
            top_k=top_k,
        )
        all_results.append(result)

    # Write overall summary CSV
    if args.output_csv:
        summary_path = args.output_csv
    elif args.config:
        summary_path = os.path.splitext(args.config)[0] + '_quality_summary.csv'
    else:
        summary_path = os.path.join(
            os.path.dirname(jobs[0]['pred_dir']), 'quality_summary.csv')

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'recording', 'total_frames', 'n_good_bouts', 'good_frames',
            'pct_good', 'fly0_body_size', 'fly1_body_size',
        ])
        for r in all_results:
            bs = r.get('body_sizes', {})
            writer.writerow([
                r['recording'],
                r['total_frames'],
                r['n_good_bouts'],
                r['good_frames'],
                f"{r['pct_good']:.1f}",
                f"{bs.get('fly0', 0):.1f}",
                f"{bs.get('fly1', 0):.1f}",
            ])

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Recording':<35} {'Frames':>8} {'Good bouts':>10} "
          f"{'Good frames':>11} {'%':>6}")
    print('-' * 75)
    total_bouts = 0
    total_good = 0
    total_all = 0
    for r in all_results:
        print(f"{r['recording']:<35} {r['total_frames']:>8} "
              f"{r['n_good_bouts']:>10} {r['good_frames']:>11} "
              f"{r['pct_good']:>5.1f}%")
        total_bouts += r['n_good_bouts']
        total_good += r['good_frames']
        total_all += r['total_frames']

    pct_total = total_good / total_all * 100 if total_all > 0 else 0
    print('-' * 75)
    print(f"{'TOTAL':<35} {total_all:>8} {total_bouts:>10} "
          f"{total_good:>11} {pct_total:>5.1f}%")
    print(f"\nSummary CSV: {summary_path}")


if __name__ == '__main__':
    main()
