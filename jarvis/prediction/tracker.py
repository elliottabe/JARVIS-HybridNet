"""
Multi-animal identity tracker for JARVIS-HybridNet.

Maintains persistent identity assignments across frames using:
- Hungarian algorithm for optimal frame-to-frame assignment
- Body size (Antenna_Base to Abd_tip distance) for initial identity assignment
  and swap verification
- Exponential moving average of body features for robust identity maintenance
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class MultiAnimalTracker:
    """
    Tracks N animals across frames with persistent identity assignment.

    Identity is initially assigned by body size (largest = fly0, smallest = flyN-1).
    Subsequent frames use Hungarian matching on 3D center positions with body-size
    verification to detect and correct swaps.

    Args:
        keypoint_names: list of keypoint name strings (used to find body landmarks)
        num_animals: number of animals to track
        max_jump_mm: maximum plausible inter-frame displacement in mm
        ema_alpha: exponential moving average decay for body size tracking
        swap_check_frames: number of initial frames to wait before enabling
            body-size-based swap correction (allows EMA to stabilize)
    """

    def __init__(self, keypoint_names, num_animals=2, max_jump_mm=5.0,
                 ema_alpha=0.05, swap_check_frames=50,
                 disable_swap_check=False):
        self.keypoint_names = list(keypoint_names)
        self.num_animals = num_animals
        self.max_jump_mm = max_jump_mm
        self.ema_alpha = ema_alpha
        self.swap_check_frames = swap_check_frames
        self.disable_swap_check = disable_swap_check

        # Keypoint indices for body size computation
        self.antenna_base_idx = self.keypoint_names.index('Antenna_Base')
        self.abd_tip_idx = self.keypoint_names.index('Abd_tip')

        # Tracking state
        self.prev_centers = {}      # fly_id -> 3D center tensor
        self.body_sizes = {}        # fly_id -> EMA of body length
        self.frame_count = 0
        self.initialized = False
        self.swap_count = 0
        self.fly_ids = [f'fly{i}' for i in range(num_animals)]

    def compute_body_length(self, points3D):
        """Euclidean distance from Antenna_Base to Abd_tip in mm."""
        if points3D is None:
            return 0.0
        p1 = points3D[self.antenna_base_idx]
        p2 = points3D[self.abd_tip_idx]
        return torch.norm(p1 - p2).item()

    def assign_identities(self, detections):
        """
        Assign persistent identities to current-frame detections.

        Args:
            detections: list of dicts, each with keys:
                'center3D': (3,) tensor
                'points3D': (num_joints, 3) tensor
                'confidences': (num_joints,) tensor

        Returns:
            dict mapping fly_id -> detection dict (or None if that fly
            was not detected this frame)
        """
        if len(detections) == 0:
            self.frame_count += 1
            return {fid: None for fid in self.fly_ids}

        if not self.initialized:
            return self._initial_assignment(detections)

        if len(detections) == 1:
            return self._single_detection_assignment(detections[0])

        return self._hungarian_assignment(detections)

    def _initial_assignment(self, detections):
        """
        First valid frame: assign identities by body size (largest = fly0).
        Waits until all N animals are detected.
        """
        if len(detections) < self.num_animals:
            # Not enough detections to initialize -- return sorted by body size
            self.frame_count += 1
            result = {fid: None for fid in self.fly_ids}
            sizes = [(self.compute_body_length(d['points3D']), i)
                     for i, d in enumerate(detections)]
            sizes.sort(reverse=True)
            for rank, (_, det_idx) in enumerate(sizes):
                if rank < len(self.fly_ids):
                    result[self.fly_ids[rank]] = detections[det_idx]
            return result

        # Sort by body length descending (largest first = fly0)
        sizes = [(self.compute_body_length(d['points3D']), i)
                 for i, d in enumerate(detections)]
        sizes.sort(reverse=True)

        assignment = {}
        for rank, (body_len, det_idx) in enumerate(sizes[:self.num_animals]):
            fly_id = self.fly_ids[rank]
            det = detections[det_idx]
            assignment[fly_id] = det
            self.prev_centers[fly_id] = det['center3D'].clone()
            self.body_sizes[fly_id] = body_len

        self.initialized = True
        self.frame_count += 1
        return assignment

    def _single_detection_assignment(self, detection):
        """
        Only one animal detected. Assign to nearest previous position.
        """
        center = detection['center3D']
        best_id = None
        best_dist = float('inf')

        for fly_id, prev_center in self.prev_centers.items():
            dist = torch.norm(center - prev_center).item()
            if dist < best_dist:
                best_dist = dist
                best_id = fly_id

        assignment = {fid: None for fid in self.fly_ids}
        if best_id is not None and best_dist < self.max_jump_mm:
            assignment[best_id] = detection
            self.prev_centers[best_id] = center.clone()
            bl = self.compute_body_length(detection['points3D'])
            self.body_sizes[best_id] = (
                (1 - self.ema_alpha) * self.body_sizes[best_id]
                + self.ema_alpha * bl
            )
        else:
            # Detection is too far -- don't update position to avoid teleporting
            pass

        self.frame_count += 1
        return assignment

    def _hungarian_assignment(self, detections):
        """
        Use Hungarian algorithm on 3D center distances, then verify with body size.
        """
        fly_ids = list(self.prev_centers.keys())
        n_flies = len(fly_ids)
        n_dets = len(detections)

        # Build cost matrix: distance from each previous center to each detection
        cost_matrix = np.zeros((n_flies, n_dets))
        for i, fly_id in enumerate(fly_ids):
            prev = self.prev_centers[fly_id]
            for j, det in enumerate(detections):
                cost_matrix[i, j] = torch.norm(prev - det['center3D']).item()

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignment = {fid: None for fid in self.fly_ids}
        matched = {}
        for r, c in zip(row_ind, col_ind):
            matched[fly_ids[r]] = detections[c]

        # Body size swap verification (only after EMA has stabilized,
        # disabled when SAM3 streaming already provides identity tracking)
        if (not self.disable_swap_check
                and self.frame_count > self.swap_check_frames
                and len(matched) == self.num_animals
                and self.num_animals == 2):
            matched = self._verify_swap_two_animals(matched)

        # Update state
        for fly_id, det in matched.items():
            assignment[fly_id] = det
            if det is not None:
                self.prev_centers[fly_id] = det['center3D'].clone()
                bl = self.compute_body_length(det['points3D'])
                self.body_sizes[fly_id] = (
                    (1 - self.ema_alpha) * self.body_sizes[fly_id]
                    + self.ema_alpha * bl
                )

        self.frame_count += 1
        return assignment

    def _verify_swap_two_animals(self, matched):
        """
        For 2 animals, check if body sizes are more consistent with swapped
        identities. If so, swap and increment swap counter.
        """
        ids = list(matched.keys())
        if len(ids) != 2:
            return matched

        id_a, id_b = ids[0], ids[1]
        det_a, det_b = matched[id_a], matched[id_b]

        if det_a is None or det_b is None:
            return matched

        bl_a = self.compute_body_length(det_a['points3D'])
        bl_b = self.compute_body_length(det_b['points3D'])

        ref_a = self.body_sizes[id_a]
        ref_b = self.body_sizes[id_b]

        # Cost of current assignment vs swapped assignment
        cost_current = abs(bl_a - ref_a) + abs(bl_b - ref_b)
        cost_swapped = abs(bl_a - ref_b) + abs(bl_b - ref_a)

        # Only swap if it's significantly better (threshold prevents noise-driven swaps)
        if cost_swapped < cost_current * 0.7:
            matched[id_a], matched[id_b] = det_b, det_a
            self.swap_count += 1

        return matched

    def get_tracking_info(self):
        """Return tracking statistics for logging."""
        return {
            'num_animals': self.num_animals,
            'frame_count': self.frame_count,
            'swap_corrections': self.swap_count,
            'fly_ids': self.fly_ids,
            'body_sizes': {k: round(v, 3) for k, v in self.body_sizes.items()},
            'initialized': self.initialized,
        }
