"""
Test multi-peak extraction on courtship frames using existing CenterDetect model.

Runs the single-fly-trained CenterDetect on courtship video frames and
extracts 2 peaks via NMS to see how well it captures both flies without
retraining.

Saves visualization images to an output directory.

Usage:
    python scripts/test_multi_peak_courtship.py
    python scripts/test_multi_peak_courtship.py --num-frames 20 --save-dir /tmp/multi_peak_test
"""

import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis.config.project_manager import ProjectManager
from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.prediction.multi_peak import extract_top_k_peaks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", default="merge_courtship_V3",
        help="JARVIS project name with trained CenterDetect",
    )
    parser.add_argument(
        "--h5-dir",
        default="/data2/users/eabe/datasets/Johnson_lab/courtship/"
                "Predictions_3D_20260123-093310/sam3_segmentation_3d",
    )
    parser.add_argument(
        "--video-dir",
        default="/data2/users/eabe/datasets/Johnson_lab/courtship/"
                "Predictions_3D_20260123-093310/videos",
    )
    parser.add_argument("--bout", type=int, default=1, help="Bout index to test")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=50,
                        help="Sample every Nth frame from the bout")
    parser.add_argument("--suppression-radius", type=int, default=15)
    parser.add_argument("--save-dir", default="/tmp/multi_peak_test")
    parser.add_argument("--camera", default=None,
                        help="Test specific camera (e.g. Cam2012630). Default: all")
    return parser.parse_args()


def load_model(project_name):
    """Load CenterDetect model."""
    project = ProjectManager()
    project.load(project_name)
    cfg = project.cfg

    center_detect = EfficientTrack(
        mode='CenterDetectInference', cfg=cfg, weights='latest'
    ).model

    transform_mean = torch.tensor(cfg.DATASET.MEAN, device='cuda').view(3, 1, 1)
    transform_std = torch.tensor(cfg.DATASET.STD, device='cuda').view(3, 1, 1)

    return center_detect, cfg, transform_mean, transform_std


def run_center_detect(model, frame_bgr, cfg, transform_mean, transform_std):
    """Run CenterDetect on a single frame, return raw heatmap."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

    img_size = cfg.CENTERDETECT.IMAGE_SIZE
    img_resized = transforms.functional.resize(img_tensor, [img_size, img_size])
    img_norm = (img_resized.cuda() - transform_mean) / transform_std

    with torch.no_grad():
        outputs = model(img_norm.unsqueeze(0))

    # outputs[1] is the half-resolution heatmap (used for peak finding)
    return outputs[1]  # (1, 1, H, W)


def visualize_frame(frame_bgr, peaks_img, maxvals, gt_male_com, gt_female_com,
                    title=""):
    """Draw detected peaks and ground truth on frame."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    # Draw GT male COM (green circle)
    if gt_male_com is not None:
        cx, cy = int(gt_male_com[0]), int(gt_male_com[1])
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(vis, (cx, cy), 12, (0, 255, 0), 2)
            cv2.putText(vis, "GT_M", (cx + 15, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw GT female COM (blue circle)
    if gt_female_com is not None:
        cx, cy = int(gt_female_com[0]), int(gt_female_com[1])
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(vis, (cx, cy), 12, (255, 0, 0), 2)
            cv2.putText(vis, "GT_F", (cx + 15, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw detected peaks
    colors = [(0, 0, 255), (0, 165, 255)]  # Red, Orange
    for pi in range(len(peaks_img)):
        cx, cy = int(peaks_img[pi][0]), int(peaks_img[pi][1])
        conf = maxvals[pi]
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(vis, (cx, cy), 8, colors[pi % 2], -1)
            cv2.putText(vis, f"P{pi+1}:{conf:.0f}", (cx + 12, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[pi % 2], 2)

    if title:
        cv2.putText(vis, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def visualize_heatmap(heatmap_np, frame_bgr, img_size):
    """Create a heatmap overlay on the resized frame."""
    frame_resized = cv2.resize(frame_bgr, (img_size, img_size))
    # heatmap_np shape: (H, W) where H = img_size/2
    hm_resized = cv2.resize(heatmap_np, (img_size, img_size))
    hm_norm = np.clip(hm_resized / max(hm_resized.max(), 1) * 255, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_resized, 0.6, hm_color, 0.4, 0)
    return overlay


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading CenterDetect model...")
    model, cfg, transform_mean, transform_std = load_model(args.project)
    img_size = cfg.CENTERDETECT.IMAGE_SIZE

    # Load bout data
    h5_path = os.path.join(args.h5_dir,
                           f"bout_{args.bout:03d}_segmentation_3d.h5")
    print(f"Loading bout data from {h5_path}")
    with h5py.File(h5_path, "r") as f:
        start_frame = int(f["start_frame"][()])
        n_frames = int(f["n_frames"][()])
        cameras = sorted(set(
            k.rsplit("_male_com", 1)[0] for k in f.keys()
            if k.endswith("_male_com")
        ))
        cam_data = {}
        for cam in cameras:
            cam_data[cam] = {
                "male_com": f[f"{cam}_male_com"][:],
                "female_com": f[f"{cam}_female_com"][:],
                "male_areas": f[f"{cam}_male_areas"][:],
                "female_areas": f[f"{cam}_female_areas"][:],
            }

    if args.camera:
        cameras = [c for c in cameras if c == args.camera]
    print(f"Cameras: {cameras}")
    print(f"Bout {args.bout}: {n_frames} frames, start={start_frame}")

    # Select frames to test
    frame_indices = list(range(0, n_frames, args.sample_every))[:args.num_frames]
    print(f"Testing {len(frame_indices)} frames: {frame_indices[:5]}...")

    # Stats tracking
    stats = {
        "peak1_conf": [], "peak2_conf": [],
        "peak1_dist_to_nearest_gt": [], "peak2_dist_to_nearest_gt": [],
        "both_detected": 0, "total": 0,
    }

    for cam in cameras:
        video_path = os.path.join(args.video_dir, f"{cam}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Cannot open {video_path}, skipping")
            continue

        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n{cam}: {img_w}x{img_h}")

        cam_save_dir = os.path.join(args.save_dir, cam)
        os.makedirs(cam_save_dir, exist_ok=True)

        for fi in frame_indices:
            abs_frame = start_frame + fi
            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"  Cannot read frame {abs_frame}")
                continue

            # Run CenterDetect
            heatmap = run_center_detect(
                model, frame, cfg, transform_mean, transform_std
            )  # (1, 1, H_hm, W_hm)

            # Extract 2 peaks
            peaks, maxvals = extract_top_k_peaks(
                heatmap, k=2, suppression_radius=args.suppression_radius
            )
            # peaks: (2, 1, 2), maxvals: (2, 1, 1)

            # Scale peaks to original image coords
            hm_h, hm_w = heatmap.shape[2], heatmap.shape[3]
            scale_x = img_w / hm_w
            scale_y = img_h / hm_h
            peaks_img = peaks[:, 0, :].cpu().numpy()  # (2, 2)
            peaks_img[:, 0] *= scale_x
            peaks_img[:, 1] *= scale_y
            mvals = maxvals[:, 0, 0].cpu().numpy()  # (2,)

            # GT centers -- h5 stores COM as (row, col) = (y, x), swap to (x, y)
            gt_male_yx = cam_data[cam]["male_com"][fi]
            gt_female_yx = cam_data[cam]["female_com"][fi]
            gt_male = np.array([gt_male_yx[1], gt_male_yx[0]])
            gt_female = np.array([gt_female_yx[1], gt_female_yx[0]])

            # Compute distance from each peak to nearest GT
            for pi in range(2):
                d_male = np.linalg.norm(peaks_img[pi] - gt_male)
                d_female = np.linalg.norm(peaks_img[pi] - gt_female)
                d_nearest = min(d_male, d_female)
                if pi == 0:
                    stats["peak1_dist_to_nearest_gt"].append(d_nearest)
                    stats["peak1_conf"].append(mvals[pi])
                else:
                    stats["peak2_dist_to_nearest_gt"].append(d_nearest)
                    stats["peak2_conf"].append(mvals[pi])

            # Check if both GT centers are "captured" (within 50px)
            threshold = 50
            gt_centers = [gt_male, gt_female]
            captured = [False, False]
            for gi, gt in enumerate(gt_centers):
                for pi in range(2):
                    if np.linalg.norm(peaks_img[pi] - gt) < threshold:
                        captured[gi] = True
            stats["total"] += 1
            if all(captured):
                stats["both_detected"] += 1

            # Visualize
            title = (f"F{abs_frame} P1:{mvals[0]:.0f} P2:{mvals[1]:.0f} "
                     f"ratio:{mvals[1]/max(mvals[0],1):.2f}")
            vis = visualize_frame(
                frame, peaks_img, mvals, gt_male, gt_female, title=title
            )

            # Heatmap overlay
            hm_np = heatmap[0, 0].cpu().numpy()
            hm_overlay = visualize_heatmap(hm_np, frame, img_size)

            # Save side by side
            vis_small = cv2.resize(vis, (img_size * 2, int(img_h * img_size * 2 / img_w)))
            combined = np.vstack([vis_small, cv2.resize(hm_overlay, (vis_small.shape[1], vis_small.shape[0]))])

            out_path = os.path.join(cam_save_dir, f"frame_{abs_frame:06d}.jpg")
            cv2.imwrite(out_path, combined)
            print(f"  Frame {abs_frame}: P1={mvals[0]:.0f} P2={mvals[1]:.0f} "
                  f"ratio={mvals[1]/max(mvals[0],1):.2f} "
                  f"both_captured={all(captured)}")

        cap.release()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    if stats["total"] > 0:
        print(f"Frames tested: {stats['total']}")
        print(f"Both flies detected (<50px): {stats['both_detected']}/{stats['total']} "
              f"({100*stats['both_detected']/stats['total']:.1f}%)")
        print(f"\nPeak 1 (strongest):")
        print(f"  Confidence: mean={np.mean(stats['peak1_conf']):.1f}, "
              f"median={np.median(stats['peak1_conf']):.1f}")
        print(f"  Distance to nearest GT: mean={np.mean(stats['peak1_dist_to_nearest_gt']):.1f}px, "
              f"median={np.median(stats['peak1_dist_to_nearest_gt']):.1f}px")
        print(f"\nPeak 2 (second strongest):")
        print(f"  Confidence: mean={np.mean(stats['peak2_conf']):.1f}, "
              f"median={np.median(stats['peak2_conf']):.1f}")
        print(f"  Distance to nearest GT: mean={np.mean(stats['peak2_dist_to_nearest_gt']):.1f}px, "
              f"median={np.median(stats['peak2_dist_to_nearest_gt']):.1f}px")
        print(f"\nConfidence ratio (P2/P1): "
              f"mean={np.mean(np.array(stats['peak2_conf'])/np.maximum(stats['peak1_conf'],1)):.3f}")
    print(f"\nVisualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
