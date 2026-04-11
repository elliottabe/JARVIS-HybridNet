"""
Multi-peak extraction utilities for multi-animal tracking.

Extracts multiple peaks from CenterDetect heatmaps using iterative
non-maximum suppression, and assigns peaks consistently across cameras
using multi-view geometry.
"""

import os
import torch

# Debug: set JARVIS_PEAK_DEBUG=1 to print per-frame peak/threshold info.
# Set JARVIS_PEAK_DUMP_DIR=/path/to/dir to also save raw heatmaps as .npz
# (one file per frame, every JARVIS_PEAK_DUMP_EVERY frames; default 1).
PEAK_DEBUG = os.environ.get("JARVIS_PEAK_DEBUG", "0") == "1"
PEAK_DUMP_DIR = os.environ.get("JARVIS_PEAK_DUMP_DIR", "")
PEAK_DUMP_EVERY = int(os.environ.get("JARVIS_PEAK_DUMP_EVERY", "1"))
_DEBUG_FRAME_COUNT = 0
if PEAK_DUMP_DIR:
    os.makedirs(PEAK_DUMP_DIR, exist_ok=True)


def extract_top_k_peaks(heatmaps, k=2, suppression_radius=6):
    """
    Extract the top-k peaks from CenterDetect heatmaps using iterative NMS.

    Args:
        heatmaps: tensor of shape (num_cameras, 1, H, W) from CenterDetect
        k: number of peaks to extract
        suppression_radius: radius (in heatmap pixels) to suppress around
            each detected peak before finding the next one

    Returns:
        peaks: tensor of shape (k, num_cameras, 2) with (x, y) coordinates
            in heatmap space
        maxvals: tensor of shape (k, num_cameras, 1) with peak confidence values
    """
    num_cameras = heatmaps.shape[0]
    H, W = heatmaps.shape[2], heatmaps.shape[3]
    device = heatmaps.device

    # Optional: dump raw heatmaps for offline inspection.
    if PEAK_DUMP_DIR and (_DEBUG_FRAME_COUNT % PEAK_DUMP_EVERY == 0):
        try:
            import numpy as _np
            _np.savez_compressed(
                os.path.join(PEAK_DUMP_DIR, f"heatmap_{_DEBUG_FRAME_COUNT:08d}.npz"),
                heatmaps=heatmaps.detach().cpu().numpy(),
            )
        except Exception as _e:
            print(f"[peak_debug] heatmap dump failed: {_e}", flush=True)

    working = heatmaps.clone()

    # Precompute coordinate grids for circular suppression mask
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    all_peaks = []
    all_maxvals = []

    for _ in range(k):
        flat = working.view(num_cameras, 1, -1)
        m = flat.argmax(2)  # (num_cameras, 1)
        peak_x = (m % W).float()
        peak_y = (m // W).float()
        peak_vals = flat.gather(2, m.unsqueeze(2)).squeeze(2)  # (num_cameras, 1)

        peaks_xy = torch.stack([peak_x.squeeze(1), peak_y.squeeze(1)], dim=1)
        all_peaks.append(peaks_xy)
        all_maxvals.append(peak_vals)

        # Suppress circular region around each detected peak
        for cam in range(num_cameras):
            cx = peak_x[cam, 0]
            cy = peak_y[cam, 0]
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= suppression_radius ** 2
            working[cam, 0][mask] = 0

    # Stack: (k, num_cameras, 2) and (k, num_cameras, 1)
    return torch.stack(all_peaks), torch.stack(all_maxvals)


def assign_peaks_across_cameras(peaks, maxvals, reproTool, downsampling_scale,
                                confidence_threshold=5,
                                min_animal_separation_mm=0.0):
    """
    Assign multi-peak detections consistently across cameras using multi-view
    geometry. Handles the case where peak[0] in camera A might correspond to
    peak[1] in camera B.

    Args:
        peaks: tensor (k, num_cameras, 2) in heatmap coordinates
        maxvals: tensor (k, num_cameras, 1) confidence values
        reproTool: ReprojectionTool instance with cameraMatrices set
        downsampling_scale: tensor (2,) mapping heatmap coords to image coords
        confidence_threshold: minimum confidence to consider a detection valid
        min_animal_separation_mm: if > 0, reject any pair of returned animal
            centers whose 3D distance is below this threshold. When a
            collapse is detected, the lower-confidence member of the pair
            is dropped so the frame is downgraded to fewer-than-k
            detections. The tracker's single-detection path then holds out
            the other animal rather than locking two tracks onto the same
            physical animal.

    Returns:
        list of dicts, one per animal, each containing:
            'center3D': (3,) tensor of 3D position
            'points2D': (num_cameras, 2) tensor of image-space 2D points
            'maxvals': (num_cameras, 1) tensor of confidences
            'num_cams_detect': int, number of cameras with valid detection
        Animals with fewer than 2 camera detections are excluded.
        May return fewer than ``k`` entries when
        ``min_animal_separation_mm`` triggers a collapse drop.
    """
    k = peaks.shape[0]
    num_cameras = peaks.shape[1]

    # Scale peaks from heatmap coordinates to image coordinates
    # The heatmap is at half resolution, and argmax gives coords in that space
    scaled_peaks = peaks * (downsampling_scale * 2).unsqueeze(0).unsqueeze(0)

    if k == 1:
        # Single animal case
        points = scaled_peaks[0]  # (num_cameras, 2)
        mvals = maxvals[0]  # (num_cameras, 1)
        num_detect = torch.sum(mvals.squeeze() > confidence_threshold).item()
        if num_detect < 2:
            return []
        # reconstructPoint expects maxvals as (num_cameras, 1, 1) for broadcasting
        center3D = reproTool.reconstructPoint(
            points.transpose(0, 1), mvals.unsqueeze(1)
        )
        return [{
            'center3D': center3D,
            'points2D': points,
            'maxvals': mvals,
            'num_cams_detect': num_detect
        }]

    # Step 1: Initial triangulation using naive peak ordering
    global _DEBUG_FRAME_COUNT
    if PEAK_DEBUG:
        per_animal_max = [maxvals[ai].squeeze().tolist() for ai in range(k)]
        per_animal_ndet = [
            int(torch.sum(maxvals[ai].squeeze() > confidence_threshold).item())
            for ai in range(k)
        ]
        print(
            f"[peak_debug f={_DEBUG_FRAME_COUNT}] thr={confidence_threshold} "
            f"k={k} ncams={num_cameras} per_animal_ndet={per_animal_ndet} "
            f"maxvals={per_animal_max}",
            flush=True,
        )
        _DEBUG_FRAME_COUNT += 1

    initial_results = []
    for animal_idx in range(k):
        points = scaled_peaks[animal_idx]  # (num_cameras, 2)
        mvals = maxvals[animal_idx]  # (num_cameras, 1)
        num_detect = torch.sum(mvals.squeeze() > confidence_threshold).item()
        if num_detect >= 2:
            # reconstructPoint expects maxvals as (num_cameras, 1, 1)
            center3D = reproTool.reconstructPoint(
                points.transpose(0, 1), mvals.unsqueeze(1)
            )
            initial_results.append({
                'center3D': center3D,
                'points': points,
                'maxvals': mvals,
            })

    if len(initial_results) < 2:
        # Could only triangulate one or zero animals
        results = []
        for r in initial_results:
            num_detect = torch.sum(
                r['maxvals'].squeeze() > confidence_threshold
            ).item()
            results.append({
                'center3D': r['center3D'],
                'points2D': r['points'],
                'maxvals': r['maxvals'],
                'num_cams_detect': num_detect,
            })
        return results

    # Step 2: Reproject initial 3D centers to all cameras
    reprojected = []
    for r in initial_results:
        reproj = reproTool.reprojectPoint(r['center3D'].unsqueeze(0))
        reprojected.append(reproj)  # each is (num_cameras, 2)

    # Step 3: Reassign peaks to nearest reprojected center per camera
    n_animals = len(initial_results)
    final_points = [torch.zeros(num_cameras, 2, device=peaks.device)
                    for _ in range(n_animals)]
    final_maxvals = [torch.zeros(num_cameras, 1, device=peaks.device)
                     for _ in range(n_animals)]

    for cam in range(num_cameras):
        # Gather all candidate peaks for this camera
        cam_peaks = scaled_peaks[:, cam, :]  # (k, 2)
        cam_vals = maxvals[:, cam, :]  # (k, 1)

        # Compute distance from each peak to each reprojected center
        # cost_matrix[peak_idx, animal_idx]
        cost = torch.zeros(k, n_animals, device=peaks.device)
        for pi in range(k):
            for ai in range(n_animals):
                cost[pi, ai] = torch.norm(cam_peaks[pi] - reprojected[ai][cam])

        # Greedy assignment (for k=2 this is optimal)
        assigned_peaks = set()
        assigned_animals = set()
        assignments = {}

        flat_costs = []
        for pi in range(k):
            for ai in range(n_animals):
                flat_costs.append((cost[pi, ai].item(), pi, ai))
        flat_costs.sort()

        for _, pi, ai in flat_costs:
            if pi not in assigned_peaks and ai not in assigned_animals:
                assignments[ai] = pi
                assigned_peaks.add(pi)
                assigned_animals.add(ai)

        for ai in range(n_animals):
            if ai in assignments:
                pi = assignments[ai]
                final_points[ai][cam] = cam_peaks[pi]
                final_maxvals[ai][cam] = cam_vals[pi]
            else:
                # No peak assigned, use the best available
                final_maxvals[ai][cam] = 0

    # Step 4: Re-triangulate with corrected assignments
    results = []
    for ai in range(n_animals):
        mvals = final_maxvals[ai]
        num_detect = torch.sum(mvals.squeeze() > confidence_threshold).item()
        if num_detect >= 2:
            # reconstructPoint expects maxvals as (num_cameras, 1, 1)
            center3D = reproTool.reconstructPoint(
                final_points[ai].transpose(0, 1), mvals.unsqueeze(1)
            )
            results.append({
                'center3D': center3D,
                'points2D': final_points[ai],
                'maxvals': mvals,
                'num_cams_detect': int(num_detect),
            })

    # Step 5: Identity-collapse guard. Step 1's naive peak ordering can leave
    # two animal entries triangulated onto the *same* physical animal when
    # one fly is occluded (the "second-brightest" peak in every camera is on
    # the first fly). Steps 2-4 then lock that collapse in. Drop any pair of
    # results whose 3D centers are closer than ``min_animal_separation_mm``,
    # keeping the more confident one. The tracker's single-detection path
    # (tracker._single_detection_assignment) then assigns the survivor to
    # the nearest fly and holds out the other, which is strictly better
    # than two phantom tracks on the same animal.
    if min_animal_separation_mm > 0 and len(results) >= 2:
        def _score(r):
            # Higher is better: more cameras first, then higher mean maxval
            # over the cameras that passed threshold.
            mv = r['maxvals'].squeeze()
            valid = mv > confidence_threshold
            mean_conf = float(mv[valid].mean().item()) if valid.any() else 0.0
            return (int(r['num_cams_detect']), mean_conf)

        # Iteratively drop the lower-scoring member of any collapsed pair
        # until no pair is within the cap. O(n^2) but n is tiny (2-3).
        dropped_any = True
        while dropped_any and len(results) >= 2:
            dropped_any = False
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    d_ij = torch.norm(
                        results[i]['center3D'] - results[j]['center3D']
                    ).item()
                    if d_ij < min_animal_separation_mm:
                        drop = j if _score(results[i]) >= _score(results[j]) else i
                        if PEAK_DEBUG or os.environ.get("JARVIS_COLLAPSE_DEBUG", "0") == "1":
                            print(
                                f"[collapse_guard] dropping animal {drop}: "
                                f"3D distance {d_ij:.3f} mm < "
                                f"{min_animal_separation_mm:.3f} mm "
                                f"(scores {_score(results[i])} vs {_score(results[j])})",
                                flush=True,
                            )
                        results.pop(drop)
                        dropped_any = True
                        break
                if dropped_any:
                    break

    return results
