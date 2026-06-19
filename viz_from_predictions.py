"""Standalone multi-bout multi-animal visualization from an existing
Predictions_3D_* directory (no re-prediction).

Reads info.yaml + tracking_info.json from the prediction dir, slices the
concatenated per-fly CSVs into per-bout CSVs, and renders an annotated video
per bout under <pred_dir>/visualization_bout<bi>/.

Usage:
    python viz_from_predictions.py \
        --pred_dir /path/to/Predictions_3D_36196238 \
        --project unified_V2_masked

    # only a few bouts (e.g. to spot-check model quality):
    python viz_from_predictions.py --pred_dir ... --project ... --bouts 0 1 2

    # restrict to specific cameras:
    python viz_from_predictions.py --pred_dir ... --project ... --cameras Cam... Cam...
"""
import argparse
import json
import os

import numpy as np
import yaml

from jarvis.visualization.create_multi_animal_videos3D import (
    create_multi_animal_videos3D,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True,
                    help="Predictions_3D_* directory to visualize")
    ap.add_argument("--project", required=True,
                    help="JARVIS project name (for skeleton/keypoints)")
    ap.add_argument("--bouts", type=int, nargs="+", default=None,
                    help="Subset of bout indices to render (default: all)")
    ap.add_argument("--cameras", type=str, nargs="+", default=None,
                    help="Camera names to render (default: all)")
    args = ap.parse_args()

    pred_dir = args.pred_dir

    with open(os.path.join(pred_dir, "info.yaml")) as f:
        info = yaml.safe_load(f)
    recording_path = info["recording_path"].strip()
    dataset_name = info["dataset_name"].strip()
    frame_start = info["frame_start"]
    number_frames = info["number_frames"]

    # Discover per-fly CSVs
    data_csvs = {}
    for f_name in sorted(os.listdir(pred_dir)):
        if f_name.startswith("data3D_fly") and f_name.endswith(".csv") \
                and "_bout" not in f_name:
            fly_id = f_name.replace("data3D_", "").replace(".csv", "")
            data_csvs[fly_id] = os.path.join(pred_dir, f_name)
    if not data_csvs:
        raise SystemExit(f"No data3D_fly*.csv found in {pred_dir}")

    # Bouts + optional mask files
    bouts = []
    mask_files = {}
    ti_path = os.path.join(pred_dir, "tracking_info.json")
    if os.path.exists(ti_path):
        with open(ti_path) as f:
            ti = json.load(f)
        for bout in ti.get("bouts", []):
            bouts.append((bout["start"], bout["end"]))
    for f_name in os.listdir(pred_dir):
        if f_name.startswith("masks_bout") and f_name.endswith(".npz"):
            bi = int(f_name.replace("masks_bout", "").replace(".npz", ""))
            mask_files[bi] = os.path.join(pred_dir, f_name)

    if not bouts:
        bouts = [(frame_start, frame_start + number_frames - 1)]

    # Preload each fly's full data once (avoid re-reading per bout)
    full_data = {}
    header_text = {}
    for fly_id, csv_path in data_csvs.items():
        all_data = np.genfromtxt(csv_path, delimiter=",")
        header_rows = 2 if np.isnan(all_data[0, 0]) else 0
        full_data[fly_id] = all_data[header_rows:]
        with open(csv_path) as orig:
            header_text[fly_id] = [orig.readline() for _ in range(header_rows)]

    want = set(args.bouts) if args.bouts is not None else None

    csv_offset = 0
    for bi, (bs, be) in enumerate(bouts):
        bout_len = be - bs + 1
        if want is not None and bi not in want:
            csv_offset += bout_len
            continue

        bout_csvs = {}
        for fly_id in data_csvs:
            bout_rows = full_data[fly_id][csv_offset:csv_offset + bout_len]
            bp = os.path.join(pred_dir, f"data3D_{fly_id}_bout{bi}.csv")
            with open(bp, "w") as bf:
                for h in header_text[fly_id]:
                    bf.write(h)
                for row in bout_rows:
                    bf.write(",".join(str(v) for v in row) + "\n")
            bout_csvs[fly_id] = bp

        bout_viz_dir = os.path.join(pred_dir, f"visualization_bout{bi}")
        print(f"Bout {bi} [{bs}-{be}] ({bout_len} frames) -> {bout_viz_dir}",
              flush=True)
        create_multi_animal_videos3D(
            project_name=args.project,
            recording_path=recording_path,
            data_csvs=bout_csvs,
            dataset_name=dataset_name,
            frame_start=bs,
            number_frames=bout_len,
            video_cam_list=args.cameras,
            mask_file=mask_files.get(bi),
            output_dir=bout_viz_dir,
        )
        csv_offset += bout_len

    print("Visualization complete.", flush=True)


if __name__ == "__main__":
    main()
