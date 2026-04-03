"""
Generate a multi-fly CenterDetect training dataset from SAM3 segmentation h5 files.

Extracts video frames from courtship recordings and creates COCO-format annotations
with 2 bounding boxes per image (male + female fly), suitable for training a
multi-peak CenterDetect model.

Optionally merges with existing single-fly data to create a combined dataset.

Usage:
    python scripts/generate_multi_fly_dataset.py
    python scripts/generate_multi_fly_dataset.py --sample-every 3 --min-area 500
    python scripts/generate_multi_fly_dataset.py --no-merge
"""

import argparse
import json
import os
import glob
import shutil

import cv2
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-fly CenterDetect training dataset"
    )
    parser.add_argument(
        "--h5-dir",
        default="/data2/users/eabe/datasets/Johnson_lab/courtship/"
                "Predictions_3D_20260123-093310/sam3_segmentation_3d",
        help="Directory containing bout_*_segmentation_3d.h5 files",
    )
    parser.add_argument(
        "--video-dir",
        default="/data2/users/eabe/datasets/Johnson_lab/courtship/"
                "Predictions_3D_20260123-093310/videos",
        help="Directory containing Cam*.mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        default="/data2/users/eabe/datasets/Johnson_lab/red_data/"
                "courtship_2fly_center",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--single-fly-dataset",
        default="/data2/users/eabe/datasets/Johnson_lab/red_data/merge_fly50_V6",
        help="Existing single-fly dataset to merge with (set --no-merge to skip)",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip merging with single-fly dataset",
    )
    parser.add_argument(
        "--sample-every", type=int, default=5,
        help="Sample every Nth frame from each bout (default: 5)",
    )
    parser.add_argument(
        "--min-area", type=float, default=500,
        help="Minimum mask area to consider a valid detection (default: 500)",
    )
    parser.add_argument(
        "--min-com-distance", type=float, default=20,
        help="Minimum pixel distance between male/female COMs (default: 20)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Fraction of frames for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--session-name", default="2025_10_20_13_20_04",
        help="Session name for image directory structure",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split",
    )
    return parser.parse_args()


def load_bout_data(h5_path, cameras):
    """Load COM and area data from a bout h5 file."""
    with h5py.File(h5_path, "r") as f:
        bout_idx = int(f["bout_idx"][()])
        n_frames = int(f["n_frames"][()])
        start_frame = int(f["start_frame"][()])

        cam_data = {}
        for cam in cameras:
            key_prefix = cam
            if f"{key_prefix}_male_com" not in f:
                continue
            cam_data[cam] = {
                "male_com": f[f"{key_prefix}_male_com"][:],
                "female_com": f[f"{key_prefix}_female_com"][:],
                "male_areas": f[f"{key_prefix}_male_areas"][:],
                "female_areas": f[f"{key_prefix}_female_areas"][:],
            }

    return {
        "bout_idx": bout_idx,
        "n_frames": n_frames,
        "start_frame": start_frame,
        "cam_data": cam_data,
    }


def extract_frames(video_path, frame_indices):
    """Extract specific frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = {}
    sorted_indices = sorted(frame_indices)

    for idx in sorted_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = frame
        else:
            print(f"  Warning: could not read frame {idx} from {video_path}")

    cap.release()
    return frames


def com_to_bbox(com, area, img_width, img_height):
    """Convert center-of-mass + mask area to COCO bbox [x, y, w, h]."""
    cx, cy = com
    side = np.sqrt(area) * 1.5
    x = max(0, cx - side / 2)
    y = max(0, cy - side / 2)
    w = min(side, img_width - x)
    h = min(side, img_height - y)
    return [float(x), float(y), float(w), float(h)]


def generate_courtship_dataset(args):
    """Generate the 2-fly courtship dataset."""
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, "bout_*_segmentation_3d.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No h5 files found in {args.h5_dir}")
    print(f"Found {len(h5_files)} bout files")

    # Discover cameras from first h5 file
    with h5py.File(h5_files[0], "r") as f:
        cameras = sorted(set(
            k.rsplit("_male_com", 1)[0]
            for k in f.keys()
            if k.endswith("_male_com")
        ))
    print(f"Cameras: {cameras}")

    # Verify videos exist
    for cam in cameras:
        vpath = os.path.join(args.video_dir, f"{cam}.mp4")
        if not os.path.exists(vpath):
            raise FileNotFoundError(f"Video not found: {vpath}")

    # Get video dimensions
    cap = cv2.VideoCapture(os.path.join(args.video_dir, f"{cameras[0]}.mp4"))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Image size: {img_width}x{img_height}")

    # Collect all valid (bout, frame_idx, camera) tuples
    # Filter per-camera: each image is independently valid if both flies
    # have sufficient area and separation in that camera view.
    all_bout_data = []
    valid_frame_keys = []  # (bout_data_idx, local_frame_idx, cam)

    for h5_path in h5_files:
        bout_data = load_bout_data(h5_path, cameras)
        bout_data_idx = len(all_bout_data)
        all_bout_data.append(bout_data)

        for fi in range(0, bout_data["n_frames"], args.sample_every):
            for cam in cameras:
                if cam not in bout_data["cam_data"]:
                    continue
                cd = bout_data["cam_data"][cam]
                if (cd["male_areas"][fi] < args.min_area
                        or cd["female_areas"][fi] < args.min_area):
                    continue
                dist = np.linalg.norm(cd["male_com"][fi] - cd["female_com"][fi])
                if dist < args.min_com_distance:
                    continue
                valid_frame_keys.append((bout_data_idx, fi, cam))

    print(f"Valid (frame, camera) pairs (sampled every {args.sample_every}): "
          f"{len(valid_frame_keys)}")

    # Train/val split at the image level
    rng = np.random.RandomState(args.seed)
    n_val = max(1, int(len(valid_frame_keys) * args.val_fraction))
    indices = rng.permutation(len(valid_frame_keys))
    val_set = set(indices[:n_val].tolist())

    # Create output directories
    for split in ["train", "val"]:
        for cam in cameras:
            os.makedirs(
                os.path.join(args.output_dir, split, args.session_name, cam),
                exist_ok=True,
            )
    os.makedirs(os.path.join(args.output_dir, "annotations"), exist_ok=True)

    # Generate dataset
    train_images, train_annotations = [], []
    val_images, val_annotations = [], []
    train_img_id, val_img_id = 0, 0
    train_ann_id, val_ann_id = 0, 0

    # Group by (bout, camera) for efficient video reading
    bout_cam_frames = {}  # (bout_data_idx, cam) -> list of (local_fi, global_idx)
    for gi, (bi, fi, cam) in enumerate(valid_frame_keys):
        bout_cam_frames.setdefault((bi, cam), []).append((fi, gi))

    for (bi, cam), frame_list in bout_cam_frames.items():
        bout_data = all_bout_data[bi]
        bout_idx = bout_data["bout_idx"]
        start_frame = bout_data["start_frame"]
        local_fis = sorted(set(fl[0] for fl in frame_list))
        global_frame_indices = [start_frame + fi for fi in local_fis]

        print(f"Processing bout {bout_idx} {cam}: {len(local_fis)} frames")

        video_path = os.path.join(args.video_dir, f"{cam}.mp4")
        extracted = extract_frames(video_path, global_frame_indices)

        for local_fi, global_idx in frame_list:
            abs_frame = start_frame + local_fi
            if abs_frame not in extracted:
                continue

            frame = extracted[abs_frame]
            is_val = global_idx in val_set
            split = "val" if is_val else "train"

            # Save image
            fname = f"{args.session_name}/{cam}/Frame_{abs_frame}.jpg"
            img_path = os.path.join(args.output_dir, split, fname)
            cv2.imwrite(img_path, frame)

            # Get COM data -- h5 stores as (row, col) = (y, x), swap to (x, y)
            cd = bout_data["cam_data"][cam]
            male_com_yx = cd["male_com"][local_fi]
            female_com_yx = cd["female_com"][local_fi]
            male_com = np.array([male_com_yx[1], male_com_yx[0]])
            female_com = np.array([female_com_yx[1], female_com_yx[0]])
            male_area = cd["male_areas"][local_fi]
            female_area = cd["female_areas"][local_fi]

            male_bbox = com_to_bbox(male_com, male_area, img_width, img_height)
            female_bbox = com_to_bbox(female_com, female_area,
                                      img_width, img_height)

            # Create dummy keypoints (50 joints, all zeros except center)
            dummy_kpts = [0] * (50 * 3)
            # Set first keypoint to center for compatibility
            male_kpts = list(dummy_kpts)
            male_kpts[0] = int(male_com[0])
            male_kpts[1] = int(male_com[1])
            male_kpts[2] = 1
            female_kpts = list(dummy_kpts)
            female_kpts[0] = int(female_com[0])
            female_kpts[1] = int(female_com[1])
            female_kpts[2] = 1

            img_entry = {
                "coco_url": "",
                "date_captured": "",
                "file_name": fname,
                "flickr_url": "",
                "height": img_height,
                "id": train_img_id if not is_val else val_img_id,
                "width": img_width,
            }

            male_ann = {
                "bbox": male_bbox,
                "category_id": 1,
                "id": train_ann_id if not is_val else val_ann_id,
                "image_id": img_entry["id"],
                "iscrowd": 0,
                "keypoints": male_kpts,
                "num_keypoints": 50,
                "segmentation": [],
            }

            if is_val:
                val_ann_id += 1
            else:
                train_ann_id += 1

            female_ann = {
                "bbox": female_bbox,
                "category_id": 1,
                "id": train_ann_id if not is_val else val_ann_id,
                "image_id": img_entry["id"],
                "iscrowd": 0,
                "keypoints": female_kpts,
                "num_keypoints": 50,
                "segmentation": [],
            }

            if is_val:
                val_images.append(img_entry)
                val_annotations.append(male_ann)
                val_annotations.append(female_ann)
                val_img_id += 1
                val_ann_id += 1
            else:
                train_images.append(img_entry)
                train_annotations.append(male_ann)
                train_annotations.append(female_ann)
                train_img_id += 1
                train_ann_id += 1

    return cameras, train_images, train_annotations, val_images, val_annotations


def build_coco_json(images, annotations, keypoint_names, skeleton, categories):
    """Build a COCO-format JSON dict."""
    return {
        "keypoint_names": keypoint_names,
        "skeleton": skeleton,
        "categories": categories,
        "calibrations": [],
        "images": images,
        "annotations": annotations,
        "framesets": [],
    }


def merge_with_single_fly(args, train_images, train_annotations,
                           val_images, val_annotations):
    """Merge courtship 2-fly data with existing single-fly dataset."""
    sf_dir = args.single_fly_dataset

    for split, imgs, anns in [("train", train_images, train_annotations),
                               ("val", val_images, val_annotations)]:
        sf_json_path = os.path.join(sf_dir, "annotations", f"instances_{split}.json")
        with open(sf_json_path) as f:
            sf_data = json.load(f)

        # Remap single-fly IDs to avoid collisions
        img_id_offset = max((im["id"] for im in imgs), default=-1) + 1
        ann_id_offset = max((a["id"] for a in anns), default=-1) + 1

        img_id_map = {}
        for sf_img in sf_data["images"]:
            old_id = sf_img["id"]
            new_id = old_id + img_id_offset
            img_id_map[old_id] = new_id
            sf_img["id"] = new_id
            # Prefix file_name to indicate it lives in the single-fly dataset
            sf_img["_source_dir"] = sf_dir
            imgs.append(sf_img)

        for sf_ann in sf_data["annotations"]:
            sf_ann["id"] = sf_ann["id"] + ann_id_offset
            sf_ann["image_id"] = img_id_map[sf_ann["image_id"]]
            anns.append(sf_ann)

        # Create symlinks for single-fly images
        for sf_img in sf_data["images"]:
            src = os.path.join(sf_dir, split, sf_img["file_name"])
            dst = os.path.join(args.output_dir, split, sf_img["file_name"])
            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.exists(dst):
                if os.path.exists(src):
                    os.symlink(src, dst)

        print(f"  {split}: merged {len(sf_data['images'])} single-fly images, "
              f"total now {len(imgs)} images, {len(anns)} annotations")

    return imgs, anns


def main():
    args = parse_args()

    print("=== Generating Multi-Fly Courtship Dataset ===\n")

    # Load keypoint names and skeleton from existing dataset
    sf_json_path = os.path.join(
        args.single_fly_dataset, "annotations", "instances_train.json"
    )
    with open(sf_json_path) as f:
        sf_data = json.load(f)
    keypoint_names = sf_data["keypoint_names"]
    skeleton = sf_data["skeleton"]
    categories = sf_data["categories"]

    # Generate courtship dataset
    cameras, train_imgs, train_anns, val_imgs, val_anns = (
        generate_courtship_dataset(args)
    )
    print(f"\nCourtship data: {len(train_imgs)} train images, "
          f"{len(val_imgs)} val images")
    print(f"  Train annotations: {len(train_anns)} "
          f"({len(train_anns)//len(train_imgs) if train_imgs else 0} per image)")
    print(f"  Val annotations: {len(val_anns)} "
          f"({len(val_anns)//len(val_imgs) if val_imgs else 0} per image)")

    # Merge with single-fly data
    if not args.no_merge:
        print(f"\nMerging with single-fly dataset: {args.single_fly_dataset}")
        merge_with_single_fly(args, train_imgs, train_anns, val_imgs, val_anns)

    # Save COCO JSON annotations
    for split, imgs, anns in [("train", train_imgs, train_anns),
                               ("val", val_imgs, val_anns)]:
        coco = build_coco_json(imgs, anns, keypoint_names, skeleton, categories)
        out_path = os.path.join(args.output_dir, "annotations",
                                f"instances_{split}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f)
        print(f"Saved {out_path}: {len(imgs)} images, {len(anns)} annotations")

    # Copy calibration if available
    calib_src = os.path.join(args.single_fly_dataset, "calib_params")
    calib_dst = os.path.join(args.output_dir, "calib_params")
    if os.path.exists(calib_src) and not os.path.exists(calib_dst):
        shutil.copytree(calib_src, calib_dst)
        print(f"Copied calibration from {calib_src}")

    print("\nDone!")


if __name__ == "__main__":
    main()
