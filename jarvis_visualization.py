"""Standalone visualization script for multi-bout multi-animal predictions."""
import os
import yaml
import numpy as np
from jarvis.visualization.create_multi_animal_videos3D import create_multi_animal_videos3D

pred_dir = '/home/eabe/Research/Github/JARVIS-HybridNet/projects/merge_courtship_V3/predictions/predictions3D/Predictions_3D_20260403-150304'

# Load prediction info
with open(os.path.join(pred_dir, 'info.yaml')) as f:
    info = yaml.safe_load(f)

recording_path = info['recording_path'].strip()
dataset_name = info['dataset_name'].strip()
frame_start = info['frame_start']
number_frames = info['number_frames']

# Discover fly CSVs
data_csvs = {}
for f_name in sorted(os.listdir(pred_dir)):
    if f_name.startswith('data3D_fly') and f_name.endswith('.csv'):
        fly_id = f_name.replace('data3D_', '').replace('.csv', '')
        data_csvs[fly_id] = os.path.join(pred_dir, f_name)

# Discover mask files and bout boundaries from tracking_info if available
tracking_info_path = os.path.join(pred_dir, 'tracking_info.json')
mask_files = {}
bouts = []

if os.path.exists(tracking_info_path):
    import json
    with open(tracking_info_path) as f:
        tracking_info = json.load(f)
    for bi, bout in enumerate(tracking_info.get('bouts', [])):
        bouts.append((bout['start'], bout['end']))
else:
    # Discover bouts from mask files
    for f_name in sorted(os.listdir(pred_dir)):
        if f_name.startswith('masks_bout') and f_name.endswith('.npz'):
            bi = int(f_name.replace('masks_bout', '').replace('.npz', ''))
            mask_files[bi] = os.path.join(pred_dir, f_name)

# If no bout info found, visualize per mask file
if not bouts and mask_files:
    # Load first fly's CSV to determine total data rows
    first_csv = list(data_csvs.values())[0]
    all_data = np.genfromtxt(first_csv, delimiter=',')
    header_rows = 2 if np.isnan(all_data[0, 0]) else 0
    total_data_rows = len(all_data) - header_rows

    # If only one mask file, visualize just that bout's frames
    if len(mask_files) == 1 and 0 in mask_files:
        mask_data = np.load(mask_files[0], allow_pickle=True)
        # Count unique frame indices in the mask file
        frame_keys = sorted(k for k in mask_data.files
                            if k.startswith('f') and '_' in k)
        frame_nums = sorted(set(int(k.split('_')[0][1:]) for k in frame_keys))
        bout0_len = max(frame_nums) + 1 if frame_nums else total_data_rows
        bouts = [(frame_start, frame_start + bout0_len - 1)]
    else:
        bouts = [(frame_start, frame_start + total_data_rows - 1)]

if len(bouts) <= 1:
    # Single bout: visualize directly
    bout_start, bout_end = bouts[0] if bouts else (frame_start, frame_start + number_frames - 1)
    bout_len = bout_end - bout_start + 1
    create_multi_animal_videos3D(
        project_name='merge_courtship_V3',
        recording_path=recording_path,
        data_csvs=data_csvs,
        dataset_name=dataset_name,
        frame_start=bout_start,
        number_frames=bout_len,
        mask_file=mask_files.get(0),
    )
else:
    # Multi-bout: per-bout visualization
    csv_offset = 0
    for bi, (bs, be) in enumerate(bouts):
        bout_len = be - bs + 1
        # Slice CSVs for this bout
        bout_csvs = {}
        for fly_id, csv_path in data_csvs.items():
            all_data = np.genfromtxt(csv_path, delimiter=',')
            header_rows = 2 if np.isnan(all_data[0, 0]) else 0
            data_rows = all_data[header_rows:]
            bout_rows = data_rows[csv_offset:csv_offset + bout_len]
            bout_csv_path = os.path.join(pred_dir, f'data3D_{fly_id}_bout{bi}.csv')
            with open(bout_csv_path, 'w') as bf:
                with open(csv_path, 'r') as orig:
                    for h in range(header_rows):
                        bf.write(orig.readline())
                for row in bout_rows:
                    bf.write(','.join(str(v) for v in row) + '\n')
            bout_csvs[fly_id] = bout_csv_path

        bout_viz_dir = os.path.join(pred_dir, f'visualization_bout{bi}')
        print(f"Bout {bi} [{bs}-{be}] ({bout_len} frames)")
        create_multi_animal_videos3D(
            project_name='fly50_V6',
            recording_path=recording_path,
            data_csvs=bout_csvs,
            dataset_name=dataset_name,
            frame_start=bs,
            number_frames=bout_len,
            mask_file=mask_files.get(bi),
            output_dir=bout_viz_dir,
        )
        csv_offset += bout_len
