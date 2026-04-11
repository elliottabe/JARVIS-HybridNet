"""
Multi-animal identity tracker for JARVIS-HybridNet.

Maintains persistent identity assignments across frames using:
- Velocity-aware Hungarian assignment (cost is on the *predicted* next
  position, not the previous position) so a brief teleport-style detection
  is rejected by the ``max_jump_mm`` plausibility cap.
- Body-length consistency in the Hungarian cost itself plus a generalised
  permutation swap check that works for any N >= 2 (not just two animals).
- Exponential moving averages of velocity and body length for robust
  identity maintenance.
- Per-fly hold-out: when no observation lies within the plausibility cap of
  a fly's predicted position, that fly is reported as missing for the frame
  rather than being teleported to the nearest noisy detection.
"""

from itertools import permutations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class MultiAnimalTracker:
    """
    Tracks N animals across frames with persistent identity assignment.

    Identity is initially assigned by body size (largest = fly0, smallest = flyN-1).
    Subsequent frames use velocity-aware Hungarian matching on 3D center
    positions with body-length-aware cost and a generalised permutation check
    that catches swaps for any N >= 2.

    Args:
        keypoint_names: list of keypoint name strings (used to find body landmarks)
        num_animals: number of animals to track
        max_jump_mm: maximum plausible inter-frame *prediction residual* in mm.
            Detections farther than this from any fly's predicted position are
            treated as missing for that fly.
        ema_alpha: exponential moving average decay for body size tracking
        swap_check_frames: number of initial frames to wait before enabling
            body-length-based swap correction (allows EMAs to stabilize)
        velocity_alpha: EMA weight for velocity update (1 = no smoothing)
        cost_size_weight: weight (mm per mm) of the body-length consistency
            term in the Hungarian cost matrix. 0 disables it.
        disable_velocity_pred: if True, fall back to using the previous
            position as the predicted next position (legacy behaviour).
    """

    def __init__(self, keypoint_names, num_animals=2, max_jump_mm=5.0,
                 ema_alpha=0.05, swap_check_frames=50,
                 disable_swap_check=False,
                 velocity_alpha=0.5,
                 cost_size_weight=0.5,
                 disable_velocity_pred=False,
                 min_animal_separation_mm=0.0):
        self.keypoint_names = list(keypoint_names)
        self.num_animals = num_animals
        self.max_jump_mm = max_jump_mm
        self.ema_alpha = ema_alpha
        self.swap_check_frames = swap_check_frames
        self.disable_swap_check = disable_swap_check
        self.velocity_alpha = velocity_alpha
        self.cost_size_weight = cost_size_weight
        self.disable_velocity_pred = disable_velocity_pred
        # Belt-and-suspenders identity-collapse guard: if two incoming
        # detections' 3D centers are closer than this threshold, drop the
        # less-confident one before running Hungarian, so we don't lock
        # two tracks onto the same physical animal. 0 disables. A matching
        # check lives upstream in multi_peak.assign_peaks_across_cameras.
        self.min_animal_separation_mm = min_animal_separation_mm

        # Keypoint indices for body size computation
        self.antenna_base_idx = self.keypoint_names.index('Antenna_Base')
        self.abd_tip_idx = self.keypoint_names.index('Abd_tip')

        # Tracking state
        self.prev_centers = {}      # fly_id -> 3D center tensor
        self.prev_velocities = {}   # fly_id -> 3D velocity tensor (EMA)
        self.body_sizes = {}        # fly_id -> EMA of body length
        self.frame_count = 0
        self.initialized = False
        self.swap_count = 0
        self.held_out_count = 0
        self.fly_ids = [f'fly{i}' for i in range(num_animals)]

    def reset(self):
        """Clear per-bout state but keep config and cumulative stats."""
        self.prev_centers.clear()
        self.prev_velocities.clear()
        self.body_sizes.clear()
        self.frame_count = 0
        self.initialized = False

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

        # Belt-and-suspenders collapse guard: if upstream multi_peak didn't
        # filter collapsed duplicates (e.g. its threshold is lower than
        # ours, or it ran without the guard), drop them here before they
        # reach Hungarian matching.
        detections = self._filter_collapsed_detections(detections)

        if len(detections) == 0:
            self.frame_count += 1
            return {fid: None for fid in self.fly_ids}

        if not self.initialized:
            return self._initial_assignment(detections)

        if len(detections) == 1:
            return self._single_detection_assignment(detections[0])

        return self._hungarian_assignment(detections)

    def _filter_collapsed_detections(self, detections):
        """Drop members of any pair of detections whose 3D centers are
        closer than ``self.min_animal_separation_mm``.

        When two detections collide, the one with more cameras (tiebreak:
        higher mean keypoint confidence) is kept; the other is dropped.
        Returns the pruned list. No-op when the guard is disabled.
        """
        if self.min_animal_separation_mm <= 0 or len(detections) < 2:
            return detections

        def _score(det):
            # Prefer more camera support; fall back to mean kp confidence.
            n_cams = int(det.get('num_cams_detect', 0))
            confs = det.get('confidences')
            if confs is None:
                mean_conf = 0.0
            else:
                mean_conf = float(torch.mean(confs).item())
            return (n_cams, mean_conf)

        pruned = list(detections)
        changed = True
        while changed and len(pruned) >= 2:
            changed = False
            for i in range(len(pruned)):
                for j in range(i + 1, len(pruned)):
                    d_ij = torch.norm(
                        pruned[i]['center3D'] - pruned[j]['center3D']
                    ).item()
                    if d_ij < self.min_animal_separation_mm:
                        drop = j if _score(pruned[i]) >= _score(pruned[j]) else i
                        pruned.pop(drop)
                        self.held_out_count += 1
                        changed = True
                        break
                if changed:
                    break
        return pruned

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
            self.prev_velocities[fly_id] = torch.zeros_like(det['center3D'])
            self.body_sizes[fly_id] = body_len

        self.initialized = True
        self.frame_count += 1
        return assignment

    def _predicted_center(self, fly_id):
        prev = self.prev_centers[fly_id]
        if self.disable_velocity_pred:
            return prev
        vel = self.prev_velocities.get(fly_id)
        if vel is None:
            return prev
        return prev + vel

    def _update_state(self, fly_id, det):
        new_center = det['center3D'].clone()
        old_center = self.prev_centers[fly_id]
        instant_vel = new_center - old_center
        old_vel = self.prev_velocities.get(fly_id)
        if old_vel is None:
            self.prev_velocities[fly_id] = instant_vel
        else:
            self.prev_velocities[fly_id] = (
                (1 - self.velocity_alpha) * old_vel
                + self.velocity_alpha * instant_vel
            )
        self.prev_centers[fly_id] = new_center
        bl = self.compute_body_length(det['points3D'])
        self.body_sizes[fly_id] = (
            (1 - self.ema_alpha) * self.body_sizes[fly_id]
            + self.ema_alpha * bl
        )

    def _single_detection_assignment(self, detection):
        """
        Only one animal detected. Assign to nearest predicted position.
        """
        center = detection['center3D']
        best_id = None
        best_dist = float('inf')

        for fly_id in self.prev_centers:
            pred = self._predicted_center(fly_id)
            dist = torch.norm(center - pred).item()
            if dist < best_dist:
                best_dist = dist
                best_id = fly_id

        assignment = {fid: None for fid in self.fly_ids}
        if best_id is not None and best_dist < self.max_jump_mm:
            assignment[best_id] = detection
            self._update_state(best_id, detection)
        else:
            self.held_out_count += 1

        self.frame_count += 1
        return assignment

    def _hungarian_assignment(self, detections):
        """
        Velocity-aware Hungarian assignment with body-length cost,
        plausibility-cap hold-out, and N-way permutation swap check.
        """
        fly_ids = list(self.prev_centers.keys())
        n_flies = len(fly_ids)
        n_dets = len(detections)

        # Precompute predicted positions and detection body lengths
        predicted = [self._predicted_center(fid) for fid in fly_ids]
        det_body_lens = [self.compute_body_length(d['points3D']) for d in detections]

        # Distance (residual w.r.t. predicted) and full cost (incl. body length)
        BIG = 1e6
        dist_matrix = np.full((n_flies, n_dets), BIG)
        cost_matrix = np.full((n_flies, n_dets), BIG)
        for i, fly_id in enumerate(fly_ids):
            ref_bl = self.body_sizes.get(fly_id, 0.0)
            for j, det in enumerate(detections):
                d = torch.norm(predicted[i] - det['center3D']).item()
                dist_matrix[i, j] = d
                if d <= self.max_jump_mm:
                    cost_matrix[i, j] = (
                        d + self.cost_size_weight * abs(det_body_lens[j] - ref_bl)
                    )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignment = {fid: None for fid in self.fly_ids}
        matched = {}
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= self.max_jump_mm:
                matched[fly_ids[r]] = detections[c]
            else:
                self.held_out_count += 1

        # Generalised body-size swap verification (any N >= 2)
        if (not self.disable_swap_check
                and self.frame_count > self.swap_check_frames
                and len(matched) >= 2):
            matched = self._verify_swap_general(matched)

        for fly_id, det in matched.items():
            assignment[fly_id] = det
            if det is not None:
                self._update_state(fly_id, det)

        self.frame_count += 1
        return assignment

    def _verify_swap_general(self, matched):
        """
        Generalised swap check: pick the permutation of matched detections to
        fly IDs that minimises total |body_len - ref_body_len|. Only swap if the
        best permutation is meaningfully better than the current assignment.
        Works for any N >= 2; N! enumeration is fine for the small N we use.
        """
        ids = [fid for fid, det in matched.items() if det is not None]
        if len(ids) < 2:
            return matched

        dets = [matched[fid] for fid in ids]
        body_lens = [self.compute_body_length(d['points3D']) for d in dets]
        refs = [self.body_sizes[fid] for fid in ids]

        def perm_cost(perm):
            return sum(abs(body_lens[perm[i]] - refs[i]) for i in range(len(ids)))

        identity = tuple(range(len(ids)))
        cost_current = perm_cost(identity)

        best_perm = identity
        best_cost = cost_current
        for perm in permutations(range(len(ids))):
            if perm == identity:
                continue
            c = perm_cost(perm)
            if c < best_cost:
                best_cost = c
                best_perm = perm

        if best_perm != identity and best_cost < cost_current * 0.7:
            new_matched = dict(matched)
            for i, fid in enumerate(ids):
                new_matched[fid] = dets[best_perm[i]]
            self.swap_count += 1
            return new_matched

        return matched

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
            'held_out': self.held_out_count,
            'fly_ids': self.fly_ids,
            'body_sizes': {k: round(v, 3) for k, v in self.body_sizes.items()},
            'initialized': self.initialized,
        }
