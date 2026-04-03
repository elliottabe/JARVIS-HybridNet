"""
SAM3 Video Propagation Tracker for multi-animal tracking.

Pre-computes per-camera segmentation masks for entire bouts using SAM3's
video propagation mode. Maintains persistent fly identities across frames
within each camera, then uses multi-view triangulation to establish
cross-camera identity correspondence.
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import torch
from tqdm import tqdm

# SAM3 is an external dependency
SAM3_PATH = "/home/eabe/Research/Github/sam3"
if SAM3_PATH not in sys.path:
    sys.path.insert(0, SAM3_PATH)

from sam3.model_builder import build_sam3_video_predictor


class BoutMasks:
    """
    Stores pre-computed SAM3 masks for an entire bout across all cameras.

    Structure:
        masks[cam_idx][frame_idx] = {
            obj_id: {
                'mask': (H, W) bool ndarray,
                'centroid': (2,) float ndarray [cx, cy],
                'score': float,
            }
        }

    After assign_identities() is called, identity_map maps
    SAM3 per-camera obj_ids to global fly_ids (0, 1, ...).
    """

    def __init__(self, num_cameras, num_frames):
        self.num_cameras = num_cameras
        self.num_frames = num_frames
        # masks[cam][frame] = {obj_id: {'mask': ..., 'centroid': ..., 'score': ...}}
        self.masks = [[{} for _ in range(num_frames)] for _ in range(num_cameras)]
        # identity_map[cam][sam3_obj_id] = global_fly_idx
        self.identity_map = [None] * num_cameras

    def set_camera_masks(self, cam_idx, frame_idx, obj_ids, binary_masks,
                         scores):
        """Store masks for one camera and one frame."""
        H, W = binary_masks.shape[1], binary_masks.shape[2]
        for i, obj_id in enumerate(obj_ids):
            mask = binary_masks[i]  # (H, W) bool
            ys, xs = np.where(mask)
            if len(xs) > 0:
                centroid = np.array([xs.mean(), ys.mean()])
            else:
                centroid = np.array([0.0, 0.0])
            self.masks[cam_idx][frame_idx][int(obj_id)] = {
                'mask': mask,
                'centroid': centroid,
                'score': float(scores[i]) if i < len(scores) else 0.0,
            }

    def assign_identities(self, repro_tool, num_animals=2):
        """
        Establish cross-camera identity using multi-view triangulation
        on the first frame where all cameras have detections.

        SAM3 assigns independent obj_ids per camera. This method finds
        which obj_id in each camera corresponds to the same physical fly.
        """
        from jarvis.prediction.multi_peak import assign_peaks_across_cameras

        # Find first frame where most cameras have >= num_animals detections
        best_frame = 0
        best_count = 0
        for f in range(min(self.num_frames, 30)):  # check first 30 frames
            count = sum(
                1 for cam in range(self.num_cameras)
                if len(self.masks[cam][f]) >= num_animals
            )
            if count > best_count:
                best_count = count
                best_frame = f
            if count == self.num_cameras:
                break

        # Build peak tensors from SAM3 centroids at best_frame
        k = num_animals
        peaks = torch.zeros(k, self.num_cameras, 2)
        maxvals = torch.zeros(k, self.num_cameras, 1)

        # Track which obj_ids correspond to which peak index per camera
        cam_obj_ids = [[] for _ in range(self.num_cameras)]

        for cam in range(self.num_cameras):
            frame_data = self.masks[cam][best_frame]
            if not frame_data:
                continue
            # Sort by score descending
            sorted_objs = sorted(
                frame_data.items(),
                key=lambda x: x[1]['score'],
                reverse=True,
            )[:k]
            for pi, (obj_id, data) in enumerate(sorted_objs):
                peaks[pi, cam] = torch.tensor(data['centroid'])
                maxvals[pi, cam, 0] = data['score'] * 255
                cam_obj_ids[cam].append(obj_id)

        # Use existing multi-peak assignment
        # downsampling_scale=0.5 so *2 scaling is identity (centroids in pixels)
        downsampling_scale = torch.tensor([0.5, 0.5])
        assignments = assign_peaks_across_cameras(
            peaks, maxvals, repro_tool, downsampling_scale,
            confidence_threshold=0.1,
        )

        # Build identity map: for each camera, map SAM3 obj_id → global fly idx
        for cam in range(self.num_cameras):
            self.identity_map[cam] = {}

        for fly_idx, animal in enumerate(assignments):
            assigned_2d = animal['points2D']  # (num_cameras, 2)
            for cam in range(self.num_cameras):
                frame_data = self.masks[cam][best_frame]
                if not frame_data:
                    continue
                # Find which obj_id is closest to the assigned 2D point
                best_obj = None
                best_dist = float('inf')
                for obj_id, data in frame_data.items():
                    dist = np.linalg.norm(
                        data['centroid'] - assigned_2d[cam].numpy()
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_obj = obj_id
                if best_obj is not None:
                    self.identity_map[cam][best_obj] = fly_idx

    def get_frame(self, frame_idx, num_animals=2):
        """
        Get pre-computed masks for a single frame, organized by global
        fly identity.

        Returns:
            list of dicts (one per fly), each with:
                'masks': (num_cameras, H, W) bool tensor — SAM3 mask per camera
                'centroids': (num_cameras, 2) float tensor — centroid per camera
                'valid': (num_cameras,) bool tensor — whether detection exists
            Returns None if no identity map is available.
        """
        if self.identity_map[0] is None:
            return None

        # Determine image dimensions from first available mask
        H, W = 0, 0
        for cam in range(self.num_cameras):
            if self.masks[cam][frame_idx]:
                first_data = next(iter(self.masks[cam][frame_idx].values()))
                H, W = first_data['mask'].shape
                break
        if H == 0:
            return None

        results = []
        for fly_idx in range(num_animals):
            fly_masks = np.zeros((self.num_cameras, H, W), dtype=bool)
            fly_centroids = np.zeros((self.num_cameras, 2), dtype=np.float32)
            fly_valid = np.zeros(self.num_cameras, dtype=bool)

            for cam in range(self.num_cameras):
                id_map = self.identity_map[cam]
                frame_data = self.masks[cam][frame_idx]
                # Find which obj_id maps to this fly_idx
                for obj_id, mapped_fly in id_map.items():
                    if mapped_fly == fly_idx and obj_id in frame_data:
                        data = frame_data[obj_id]
                        fly_masks[cam] = data['mask']
                        fly_centroids[cam] = data['centroid']
                        fly_valid[cam] = True
                        break

            results.append({
                'masks': torch.from_numpy(fly_masks),
                'centroids': torch.from_numpy(fly_centroids),
                'valid': torch.from_numpy(fly_valid),
            })

        return results


class SAM3VideoTracker:
    """
    Pre-computes SAM3 segmentation masks for entire bouts using video
    propagation mode. Runs on a separate GPU from the JARVIS pipeline.

    Args:
        gpu_id: GPU index for SAM3 (default: 1 to keep JARVIS on GPU 0)
        text_prompt: text description for detection
        apply_temporal_disambiguation: whether to use hotstart delay
    """

    def __init__(self, gpu_id=1, text_prompt='fly',
                 apply_temporal_disambiguation=False):
        self.gpu_id = gpu_id
        self.text_prompt = text_prompt

        print(f"  Loading SAM3 video predictor on cuda:{gpu_id}...")
        self.predictor = build_sam3_video_predictor(
            gpus_to_use=[gpu_id],
            apply_temporal_disambiguation=apply_temporal_disambiguation,
        )
        print(f"  SAM3 video predictor loaded.")

    def _extract_bout_frames(self, video_path, frame_start, num_frames,
                             output_dir):
        """Extract bout frames from video to JPEG folder for SAM3."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(
                os.path.join(output_dir, f'{i:06d}.jpg'),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
        cap.release()

    def process_bout(self, video_paths, frame_start, num_frames,
                     num_animals=2):
        """
        Run SAM3 video propagation on all cameras for one bout.

        Args:
            video_paths: list of camera video file paths
            frame_start: first frame number in the original video
            num_frames: number of frames in the bout
            num_animals: expected number of animals

        Returns:
            BoutMasks object with per-camera, per-frame, per-fly masks
        """
        num_cameras = len(video_paths)
        bout_masks = BoutMasks(num_cameras, num_frames)

        # Create temp directory for frame extraction
        tmp_base = tempfile.mkdtemp(prefix='sam3_bout_')

        try:
            for cam_idx, video_path in enumerate(video_paths):
                cam_name = os.path.splitext(
                    os.path.basename(video_path)
                )[0]
                cam_dir = os.path.join(tmp_base, cam_name)

                # Extract bout frames to JPEG folder
                print(f"    Extracting frames for {cam_name}...")
                self._extract_bout_frames(
                    video_path, frame_start, num_frames, cam_dir
                )

                # Start SAM3 session
                response = self.predictor.handle_request({
                    'type': 'start_session',
                    'resource_path': cam_dir,
                })
                session_id = response['session_id']

                # Detect flies on frame 0
                response = self.predictor.handle_request({
                    'type': 'add_prompt',
                    'session_id': session_id,
                    'frame_index': 0,
                    'text': self.text_prompt,
                })
                outputs = response['outputs']
                if outputs and 'out_obj_ids' in outputs:
                    bout_masks.set_camera_masks(
                        cam_idx, 0,
                        outputs['out_obj_ids'],
                        outputs['out_binary_masks'],
                        outputs.get('out_probs', np.array([])),
                    )
                    n_det = len(outputs['out_obj_ids'])
                    print(f"    {cam_name}: {n_det} flies detected on frame 0")
                else:
                    print(f"    {cam_name}: no detections on frame 0")

                # Propagate through all frames
                print(f"    {cam_name}: propagating masks...")
                for response in self.predictor.handle_stream_request({
                    'type': 'propagate_in_video',
                    'session_id': session_id,
                    'propagation_direction': 'forward',
                }):
                    fi = response['frame_index']
                    out = response['outputs']
                    if out and 'out_obj_ids' in out:
                        bout_masks.set_camera_masks(
                            cam_idx, fi,
                            out['out_obj_ids'],
                            out['out_binary_masks'],
                            out.get('out_probs', np.array([])),
                        )

                # Close session
                self.predictor.handle_request({
                    'type': 'close_session',
                    'session_id': session_id,
                })
                print(f"    {cam_name}: done")

        finally:
            # Clean up temp frames
            shutil.rmtree(tmp_base, ignore_errors=True)

        return bout_masks


class SAM3StreamingTracker:
    """
    Streaming SAM3 video propagation tracker for multi-animal tracking.

    Unlike SAM3VideoTracker which pre-computes all masks for an entire bout,
    this class keeps SAM3 video sessions open and yields masks one frame at
    a time. This is much more memory-efficient for long bouts.

    Usage:
        tracker = SAM3StreamingTracker(gpu_id=1)
        tracker.start_bout(video_paths, frame_start, num_frames)
        tracker.detect_frame0()  # text-grounded detection on frame 0
        tracker.assign_identities(repro_tool, num_animals=2)

        # In frame loop:
        for frame_idx in range(num_frames):
            masks = tracker.get_next_frame(num_animals=2)
            # masks is a list of per-fly dicts (same format as BoutMasks.get_frame)
            predictor(imgs, camera_matrices, precomputed_masks=masks)

        tracker.close()
    """

    def __init__(self, gpu_id=1, text_prompt='fly',
                 apply_temporal_disambiguation=False):
        self.gpu_id = gpu_id
        self.text_prompt = text_prompt

        print(f"  Loading SAM3 video predictor on cuda:{gpu_id}...")
        self.predictor = build_sam3_video_predictor(
            gpus_to_use=[gpu_id],
            apply_temporal_disambiguation=apply_temporal_disambiguation,
        )
        print(f"  SAM3 video predictor loaded.")

        self._session_ids = []
        self._propagation_generators = []
        self._tmp_dir = None
        self._num_cameras = 0
        self._img_size = None  # (H, W)
        # frame0_masks[cam] = {obj_id: {'mask': ..., 'centroid': ..., 'score': ...}}
        self._frame0_masks = []
        # identity_map[cam][obj_id] = global_fly_idx
        self._identity_map = []
        self._init_frame = 0  # frame where detection succeeded

    def start_bout(self, video_paths, frame_start, num_frames):
        """
        Extract bout frames and start SAM3 video sessions for all cameras.

        Args:
            video_paths: list of camera video file paths
            frame_start: first frame number in the original video
            num_frames: number of frames in the bout
        """
        self._num_cameras = len(video_paths)
        self._tmp_dir = tempfile.mkdtemp(prefix='sam3_stream_')
        self._session_ids = []
        self._frame0_masks = [{} for _ in range(self._num_cameras)]
        self._identity_map = [{} for _ in range(self._num_cameras)]
        self._init_frame = 0

        for cam_idx, video_path in enumerate(video_paths):
            cam_name = os.path.splitext(os.path.basename(video_path))[0]
            cam_dir = os.path.join(self._tmp_dir, cam_name)

            # Extract bout frames to JPEG folder
            os.makedirs(cam_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if self._img_size is None:
                    self._img_size = (frame.shape[0], frame.shape[1])
                cv2.imwrite(
                    os.path.join(cam_dir, f'{i:06d}.jpg'),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
            cap.release()

            # Start SAM3 session
            response = self.predictor.handle_request({
                'type': 'start_session',
                'resource_path': cam_dir,
            })
            self._session_ids.append(response['session_id'])
            print(f"    {cam_name}: session started ({num_frames} frames)")

    def detection_count(self):
        """Return the median number of detections across cameras.

        Useful for checking whether enough flies were found. Uses median
        so that one bad camera doesn't dominate the count.
        """
        counts = [len(cam_masks) for cam_masks in self._frame0_masks]
        if not counts:
            return 0
        return int(np.median(counts))

    def _detect_on_frame(self, frame_index):
        """Run text-grounded detection on a specific frame for all cameras.

        Returns per-camera mask dicts in the same format as _frame0_masks.
        """
        per_cam = [{} for _ in range(self._num_cameras)]
        for cam_idx, session_id in enumerate(self._session_ids):
            response = self.predictor.handle_request({
                'type': 'add_prompt',
                'session_id': session_id,
                'frame_index': frame_index,
                'text': self.text_prompt,
            })
            outputs = response['outputs']
            if outputs and 'out_obj_ids' in outputs:
                obj_ids = outputs['out_obj_ids']
                masks = outputs['out_binary_masks']  # (N, H, W)
                scores = outputs.get('out_probs', np.array([]))
                for i, obj_id in enumerate(obj_ids):
                    mask = masks[i]
                    ys, xs = np.where(mask)
                    centroid = (np.array([xs.mean(), ys.mean()])
                                if len(xs) > 0 else np.array([0.0, 0.0]))
                    per_cam[cam_idx][int(obj_id)] = {
                        'mask': mask,
                        'centroid': centroid,
                        'score': float(scores[i]) if i < len(scores) else 0.0,
                    }
        return per_cam

    def detect_frame0(self, num_animals=2, max_retry_frames=10):
        """
        Run text-grounded detection, retrying on later frames if needed.

        Tries frame 0 first. If fewer than `num_animals` are detected
        (median across cameras), retries on frames 1..max_retry_frames.
        Keeps the best detection (highest median count).

        Args:
            num_animals: expected number of animals to find
            max_retry_frames: max frames to try before giving up

        Returns:
            True if enough flies detected, False otherwise.
        """
        best_masks = None
        best_count = 0
        best_frame = 0

        for frame_idx in range(max_retry_frames):
            per_cam = self._detect_on_frame(frame_idx)
            counts = [len(cm) for cm in per_cam]
            median_count = int(np.median(counts))

            if median_count > best_count:
                best_count = median_count
                best_masks = per_cam
                best_frame = frame_idx

            if median_count >= num_animals:
                break

            if frame_idx == 0:
                print(f"    Frame 0: median {median_count}/{num_animals} "
                      f"detections, retrying...")

        # Use the best detection found
        self._frame0_masks = best_masks
        self._init_frame = best_frame

        for cam_idx in range(self._num_cameras):
            n = len(self._frame0_masks[cam_idx])
            print(f"    Camera {cam_idx}: {n} flies detected "
                  f"(frame {best_frame})")

        if best_count < num_animals:
            print(f"    WARNING: Only {best_count}/{num_animals} flies "
                  f"detected after trying {max_retry_frames} frames!")
            return False

        return True

    def assign_identities(self, repro_tool, num_animals=2):
        """
        Establish cross-camera identity using multi-view triangulation
        on frame 0 detections. Same logic as BoutMasks.assign_identities.
        """
        from jarvis.prediction.multi_peak import assign_peaks_across_cameras

        k = num_animals
        peaks = torch.zeros(k, self._num_cameras, 2)
        maxvals = torch.zeros(k, self._num_cameras, 1)
        cam_obj_ids = [[] for _ in range(self._num_cameras)]

        for cam in range(self._num_cameras):
            frame_data = self._frame0_masks[cam]
            if not frame_data:
                continue
            sorted_objs = sorted(
                frame_data.items(),
                key=lambda x: x[1]['score'],
                reverse=True,
            )[:k]
            for pi, (obj_id, data) in enumerate(sorted_objs):
                peaks[pi, cam] = torch.tensor(data['centroid'])
                maxvals[pi, cam, 0] = data['score'] * 255
                cam_obj_ids[cam].append(obj_id)

        downsampling_scale = torch.tensor([0.5, 0.5])
        assignments = assign_peaks_across_cameras(
            peaks, maxvals, repro_tool, downsampling_scale,
            confidence_threshold=0.1,
        )

        for fly_idx, animal in enumerate(assignments):
            assigned_2d = animal['points2D']
            for cam in range(self._num_cameras):
                frame_data = self._frame0_masks[cam]
                if not frame_data:
                    continue
                best_obj = None
                best_dist = float('inf')
                for obj_id, data in frame_data.items():
                    dist = np.linalg.norm(
                        data['centroid'] - assigned_2d[cam].numpy()
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_obj = obj_id
                if best_obj is not None:
                    self._identity_map[cam][best_obj] = fly_idx

    def start_propagation(self):
        """
        Start the propagation generators for all cameras.
        Must be called after detect_frame0() and assign_identities().
        """
        self._propagation_generators = []
        for session_id in self._session_ids:
            gen = self.predictor.handle_stream_request({
                'type': 'propagate_in_video',
                'session_id': session_id,
                'propagation_direction': 'forward',
            })
            self._propagation_generators.append(gen)

    @property
    def init_frame(self):
        """Frame index where detection succeeded (0 if frame 0 worked)."""
        return self._init_frame

    def get_frame0_masks(self, num_animals=2):
        """Get masks for the init frame (from detection, not propagation).

        If retry was needed, this returns masks from the frame where
        detection succeeded (self.init_frame), not necessarily frame 0.
        """
        return self._build_frame_output(self._frame0_masks, num_animals)

    def get_next_frame(self, num_animals=2):
        """
        Advance all camera propagation generators by one frame and return
        masks in the same format as BoutMasks.get_frame().

        Returns:
            list of per-fly dicts, each with:
                'masks': (num_cameras, H, W) bool tensor
                'centroids': (num_cameras, 2) float tensor
                'valid': (num_cameras,) bool tensor
            Returns None if propagation is exhausted.
        """
        current_frame_masks = [{} for _ in range(self._num_cameras)]

        for cam_idx, gen in enumerate(self._propagation_generators):
            try:
                response = next(gen)
            except StopIteration:
                return None
            out = response['outputs']
            if out and 'out_obj_ids' in out:
                obj_ids = out['out_obj_ids']
                masks = out['out_binary_masks']
                scores = out.get('out_probs', np.array([]))
                for i, obj_id in enumerate(obj_ids):
                    mask = masks[i]
                    ys, xs = np.where(mask)
                    centroid = (np.array([xs.mean(), ys.mean()])
                                if len(xs) > 0 else np.array([0.0, 0.0]))
                    current_frame_masks[cam_idx][int(obj_id)] = {
                        'mask': mask,
                        'centroid': centroid,
                        'score': float(scores[i]) if i < len(scores) else 0.0,
                    }

        return self._build_frame_output(current_frame_masks, num_animals)

    def _build_frame_output(self, frame_masks, num_animals):
        """Convert per-camera mask dicts into the per-fly format expected
        by _forward_with_precomputed_masks."""
        H, W = self._img_size if self._img_size else (0, 0)
        if H == 0:
            return None

        results = []
        for fly_idx in range(num_animals):
            fly_masks = np.zeros((self._num_cameras, H, W), dtype=bool)
            fly_centroids = np.zeros((self._num_cameras, 2), dtype=np.float32)
            fly_valid = np.zeros(self._num_cameras, dtype=bool)

            for cam in range(self._num_cameras):
                id_map = self._identity_map[cam]
                cam_masks = frame_masks[cam]
                for obj_id, mapped_fly in id_map.items():
                    if mapped_fly == fly_idx and obj_id in cam_masks:
                        data = cam_masks[obj_id]
                        fly_masks[cam] = data['mask']
                        fly_centroids[cam] = data['centroid']
                        fly_valid[cam] = True
                        break

            results.append({
                'masks': torch.from_numpy(fly_masks),
                'centroids': torch.from_numpy(fly_centroids),
                'valid': torch.from_numpy(fly_valid),
            })

        return results

    def close(self):
        """Close all sessions and clean up temp files."""
        for session_id in self._session_ids:
            try:
                self.predictor.handle_request({
                    'type': 'close_session',
                    'session_id': session_id,
                })
            except Exception:
                pass
        self._session_ids = []
        self._propagation_generators = []
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
