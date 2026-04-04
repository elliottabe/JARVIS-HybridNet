"""
SAM3 Video Propagation Tracker for multi-animal tracking.

Pre-computes per-camera segmentation masks for entire bouts using SAM3's
video propagation mode. Maintains persistent fly identities across frames
within each camera, then uses multi-view triangulation to establish
cross-camera identity correspondence.
"""

import os
import shutil
import subprocess
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

import gc

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
                        data['centroid'] - assigned_2d[cam].cpu().numpy()
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

    Opens all camera sessions simultaneously (img_batch stays on CPU via
    patched SAM3, so GPU usage is ~2GB/session instead of ~9GB). Masks are
    produced frame-by-frame via propagation generators.

    Usage:
        tracker = SAM3StreamingTracker(gpu_id=0)
        tracker.start_bout(video_paths, frame_start, num_frames)
        ok = tracker.detect_frame0(num_animals=2)
        if ok:
            tracker.assign_identities(repro_tool, num_animals=2)
            tracker.start_propagation()
            # frame loop:
            masks = tracker.get_frame0_masks(num_animals=2)  # init frame
            for _ in range(1, num_frames):
                masks = tracker.get_next_frame(num_animals=2)
        tracker.close()
    """

    def __init__(self, gpu_id=0, text_prompt='insect',
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
        self._frame0_masks = []
        self._identity_map = []
        self._init_frame = 0

    def start_bout(self, video_paths, frame_start, num_frames):
        """
        Extract bout clips and open SAM3 video sessions for all cameras.
        img_batch stays on CPU (patched SAM3), so this won't OOM.
        """
        self._num_cameras = len(video_paths)
        self._tmp_dir = tempfile.mkdtemp(prefix='sam3_stream_')
        self._session_ids = []
        self._frame0_masks = [{} for _ in range(self._num_cameras)]
        self._identity_map = [{} for _ in range(self._num_cameras)]
        self._init_frame = 0

        for cam_idx, video_path in enumerate(video_paths):
            cam_name = os.path.splitext(os.path.basename(video_path))[0]

            # Get video metadata
            if self._img_size is None:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    self._img_size = (h, w)
                    self._fps = fps
                cap.release()

            # Extract bout as temp MP4 clip via ffmpeg
            clip_path = os.path.join(self._tmp_dir, f'{cam_name}.mp4')
            fps = getattr(self, '_fps', 30.0)
            start_sec = frame_start / fps
            duration_sec = num_frames / fps
            subprocess.run(
                ['ffmpeg', '-y', '-ss', f'{start_sec:.4f}',
                 '-i', video_path,
                 '-t', f'{duration_sec:.4f}',
                 '-c:v', 'libx264', '-preset', 'ultrafast',
                 '-pix_fmt', 'yuv420p',
                 '-loglevel', 'error',
                 clip_path],
                check=True,
            )

            # Open SAM3 session on the clip
            response = self.predictor.handle_request({
                'type': 'start_session',
                'resource_path': clip_path,
            })
            self._session_ids.append(response['session_id'])
            print(f"    {cam_name}: session started ({num_frames} frames)")

    def _detect_on_frame(self, frame_index):
        """Run text-grounded detection on a specific frame for all cameras."""
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
                masks = outputs['out_binary_masks']
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

        if best_masks is not None:
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
        """Assign cross-camera identities using multi-view triangulation."""
        from jarvis.prediction.multi_peak import assign_peaks_across_cameras

        k = num_animals
        peaks = torch.zeros(k, self._num_cameras, 2)
        maxvals = torch.zeros(k, self._num_cameras, 1)

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

        device = repro_tool.cameraMatrices.device
        downsampling_scale = torch.tensor([0.5, 0.5], device=device)
        assignments = assign_peaks_across_cameras(
            peaks.to(device), maxvals.to(device), repro_tool,
            downsampling_scale, confidence_threshold=0.1,
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
                        data['centroid'] - assigned_2d[cam].cpu().numpy()
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_obj = obj_id
                if best_obj is not None:
                    self._identity_map[cam][best_obj] = fly_idx

    def start_propagation(self):
        """Start propagation generators for all cameras."""
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
        """Frame index where detection succeeded."""
        return self._init_frame

    def get_frame0_masks(self, num_animals=2):
        """Get masks for the init frame (from detection, not propagation)."""
        return self._build_frame_output(self._frame0_masks, num_animals)

    def get_next_frame(self, num_animals=2):
        """Advance all propagation generators by one frame."""
        current_frame_masks = [{} for _ in range(self._num_cameras)]
        all_stopped = True

        for cam_idx, gen in enumerate(self._propagation_generators):
            if gen is None:
                continue
            try:
                response = next(gen)
                all_stopped = False
            except StopIteration:
                self._propagation_generators[cam_idx] = None
                continue
            except RuntimeError as e:
                print(f"    SAM3 propagation error on camera {cam_idx}: {e}")
                self._propagation_generators[cam_idx] = None
                continue
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

        if all_stopped:
            return None
        return self._build_frame_output(current_frame_masks, num_animals)

    def _build_frame_output(self, frame_masks, num_animals):
        """Convert per-camera mask dicts into per-fly format."""
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


class SAM3LowLatencyTracker:
    """
    Low-latency streaming SAM3 tracker for multi-animal tracking.

    Processes frames one at a time instead of loading entire videos into
    GPU memory.  Supports multiple objects (e.g. 2 flies) tracked
    simultaneously with automatic memory trimming for long bouts.

    Detection on frame 0 uses the full SAM3 video model (text-grounded).
    After detection the heavyweight detector is released and only the
    lightweight tracker + backbone are kept for streaming propagation,
    reducing steady-state GPU usage from ~9 GB to ~300-500 MB per camera.

    Usage:
        tracker = SAM3LowLatencyTracker(gpu_id=0)
        tracker.start_bout(video_paths, frame_start, num_frames)
        ok = tracker.detect_frame0(num_animals=2)
        if ok:
            tracker.assign_identities(repro_tool, num_animals=2)
            tracker.start_propagation()
            masks = tracker.get_frame0_masks(num_animals=2)
            for fi in range(1, num_frames):
                masks = tracker.get_next_frame(num_animals=2)
        tracker.close()
    """

    # SAM3 model constants
    _IMAGE_SIZE = 1008
    _IMAGE_MEAN = 0.5
    _IMAGE_STD = 0.5

    def __init__(self, gpu_id=0, text_prompt='insect',
                 apply_temporal_disambiguation=False):
        self.gpu_id = gpu_id
        self.text_prompt = text_prompt

        print(f"  Loading SAM3 video predictor on cuda:{gpu_id}...")
        self._full_predictor = build_sam3_video_predictor(
            gpus_to_use=[gpu_id],
            apply_temporal_disambiguation=apply_temporal_disambiguation,
        )
        print(f"  SAM3 video predictor loaded.")

        self.tracker = None
        self._inference_states = []
        self._video_caps = []
        self._video_paths = []
        self._tmp_dir = None
        self._num_cameras = 0
        self._img_size = None
        self._frame0_masks = []
        self._identity_map = []
        self._init_frame = 0
        self._frame_start = 0
        self._num_frames = 0
        self._current_frame_idx = 0
        self._prev_centroids = None  # per-cam, per-obj centroids from last frame

    # ── Frame preprocessing ──────────────────────────────────────────

    def _preprocess_frame(self, bgr_frame):
        """Convert an OpenCV BGR frame to a SAM3 tracker input tensor.

        Returns:
            (3, 1008, 1008) float32 tensor, normalised with mean/std = 0.5.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._IMAGE_SIZE, self._IMAGE_SIZE))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self._IMAGE_MEAN) / self._IMAGE_STD
        return tensor

    # ── Bout lifecycle ───────────────────────────────────────────────

    def start_bout(self, video_paths, frame_start, num_frames):
        """Open video captures for all cameras (no video extraction)."""
        self._num_cameras = len(video_paths)
        self._frame_start = frame_start
        self._num_frames = num_frames
        self._video_paths = list(video_paths)
        self._video_caps = []
        self._frame0_masks = [{} for _ in range(self._num_cameras)]
        self._identity_map = [{} for _ in range(self._num_cameras)]
        self._init_frame = 0

        for video_path in video_paths:
            if self._img_size is None:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self._img_size = (h, w)
                cap.release()

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            self._video_caps.append(cap)
            cam_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"    {cam_name}: video opened ({num_frames} frames)")

    # ── Detection (uses full model, temporary) ───────────────────────

    def _extract_detection_frames(self, max_frames=10):
        """Extract a small number of frames as temp JPEGs for detection."""
        self._tmp_dir = tempfile.mkdtemp(prefix='sam3_detect_')
        session_ids = []

        for cam_idx, video_path in enumerate(self._video_paths):
            cam_name = os.path.splitext(os.path.basename(video_path))[0]
            cam_dir = os.path.join(self._tmp_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_start)
            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(
                    os.path.join(cam_dir, f'{i:06d}.jpg'),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
            cap.release()

            response = self._full_predictor.handle_request({
                'type': 'start_session',
                'resource_path': cam_dir,
            })
            session_ids.append(response['session_id'])

        return session_ids

    def _detect_on_frame(self, session_ids, frame_index):
        """Run text-grounded detection on a specific frame for all cameras."""
        per_cam = [{} for _ in range(self._num_cameras)]
        for cam_idx, session_id in enumerate(session_ids):
            response = self._full_predictor.handle_request({
                'type': 'add_prompt',
                'session_id': session_id,
                'frame_index': frame_index,
                'text': self.text_prompt,
            })
            outputs = response['outputs']
            if outputs and 'out_obj_ids' in outputs:
                obj_ids = outputs['out_obj_ids']
                masks = outputs['out_binary_masks']
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
        Detect animals using text-grounded detection, then switch to
        streaming tracker mode.

        Returns:
            True if enough animals detected, False otherwise.
        """
        # Phase 1: detect using full model on a few temp frames
        session_ids = self._extract_detection_frames(
            max_frames=max_retry_frames,
        )

        best_masks = None
        best_count = 0
        best_frame = 0

        for frame_idx in range(max_retry_frames):
            per_cam = self._detect_on_frame(session_ids, frame_idx)
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

        if best_masks is not None:
            self._frame0_masks = best_masks
        self._init_frame = best_frame

        for cam_idx in range(self._num_cameras):
            n = len(self._frame0_masks[cam_idx])
            print(f"    Camera {cam_idx}: {n} flies detected "
                  f"(frame {best_frame})")

        # Close detection sessions
        for session_id in session_ids:
            try:
                self._full_predictor.handle_request({
                    'type': 'close_session',
                    'session_id': session_id,
                })
            except Exception:
                pass

        # Phase 2: switch to streaming tracker mode
        self._setup_streaming_tracker()

        if best_count < num_animals:
            print(f"    WARNING: Only {best_count}/{num_animals} flies "
                  f"detected after trying {max_retry_frames} frames!")
            return False
        return True

    def _setup_streaming_tracker(self):
        """Extract tracker+backbone from full model and init per-camera states."""
        model = self._full_predictor.model
        self.tracker = model.tracker
        # Attach the visual backbone so the tracker can compute features
        # The tracker's forward_image calls self.backbone.forward_image(),
        # which expects a SAM3VLBackbone (not just the ViT).
        self.tracker.backbone = model.detector.backbone

        # Create per-camera inference states
        H, W = self._img_size
        self._inference_states = []
        for cam_idx in range(self._num_cameras):
            inf_state = self.tracker.init_state(
                video_height=H,
                video_width=W,
                num_frames=self._num_frames + 10,
            )
            self._inference_states.append(inf_state)

        # Feed detected masks into the tracker
        for cam_idx in range(self._num_cameras):
            inf_state = self._inference_states[cam_idx]
            frame_data = self._frame0_masks[cam_idx]

            # Read the init frame from the video capture
            cap = self._video_caps[cam_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    self._frame_start + self._init_frame)
            ret, bgr_frame = cap.read()
            if not ret:
                print(f"    WARNING: Could not read init frame for "
                      f"camera {cam_idx}")
                continue
            frame_tensor = self._preprocess_frame(bgr_frame)

            # Add each detected object's mask
            for obj_id, data in frame_data.items():
                mask_tensor = torch.from_numpy(data['mask'].astype(np.float32))
                self.tracker.add_new_mask_with_frame(
                    inf_state,
                    frame_idx=self._init_frame,
                    obj_id=obj_id,
                    mask=mask_tensor,
                    frame_tensor=frame_tensor,
                )

            # Consolidate before tracking
            self.tracker.propagate_in_video_preflight(inf_state)

            # Reset capture to first tracking frame
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    self._frame_start + self._init_frame + 1)

        # Release the full model to free GPU memory
        self._release_detector()

    def _release_detector(self):
        """Free the full model, keeping only tracker + backbone."""
        if self._full_predictor is None:
            return
        # Clean up temp detection frames
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
        # The tracker is kept alive via self.tracker reference.
        # Deleting the predictor frees detector, text encoder, etc.
        self._full_predictor = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Detector released, streaming tracker active.")

    # ── Identity assignment ────────────────────────────────────────────

    def assign_identities(self, repro_tool, num_animals=2):
        """Assign cross-camera identities using multi-view triangulation.

        After assignment, cameras that are missing detections for some flies
        get synthetic box-prompt masks via 3D reprojection, so the tracker
        can follow all animals even if they weren't visible on the init frame.
        """
        from jarvis.prediction.multi_peak import assign_peaks_across_cameras

        k = num_animals
        peaks = torch.zeros(k, self._num_cameras, 2)
        maxvals = torch.zeros(k, self._num_cameras, 1)

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

        device = repro_tool.cameraMatrices.device
        downsampling_scale = torch.tensor([0.5, 0.5], device=device)
        assignments = assign_peaks_across_cameras(
            peaks.to(device), maxvals.to(device), repro_tool,
            downsampling_scale, confidence_threshold=0.1,
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
                        data['centroid'] - assigned_2d[cam].cpu().numpy()
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_obj = obj_id
                if best_obj is not None:
                    self._identity_map[cam][best_obj] = fly_idx

        # Fill missing fly detections on cameras via 3D reprojection
        self._fill_missing_detections(assignments, repro_tool, num_animals)

    def validate_identity_continuity(self, prev_fly_centers, repro_tool,
                                     num_animals=2):
        """Check that identity assignments are consistent with known positions.

        After a chunk transition, the fresh assign_identities() may swap
        fly indices relative to the previous chunk.  This method triangulates
        each fly's frame-0 position from the current identity map and compares
        to the tracker's known 3D positions from the end of the previous chunk.
        If swapping identities gives a better match, the identity map is
        corrected in-place.

        Args:
            prev_fly_centers: dict {fly_idx (int): (3,) tensor} of 3D centres
                from the MultiAnimalTracker at the end of the previous chunk.
            repro_tool: ReprojectionTool for triangulation.
            num_animals: number of animals (only 2-animal swap is supported).

        Returns:
            True if identities were swapped and corrected, False otherwise.
        """
        if num_animals != 2 or len(prev_fly_centers) < 2:
            return False

        # Triangulate each fly's frame-0 position under the current map
        current_centers = {}
        for fly_idx in range(num_animals):
            pts = torch.zeros(2, self._num_cameras)
            weights = torch.zeros(self._num_cameras, 1, 1)
            for cam in range(self._num_cameras):
                id_map = self._identity_map[cam]
                frame_data = self._frame0_masks[cam]
                for obj_id, mapped_fly in id_map.items():
                    if mapped_fly == fly_idx and obj_id in frame_data:
                        cx, cy = frame_data[obj_id]['centroid']
                        pts[0, cam] = cx
                        pts[1, cam] = cy
                        weights[cam, 0, 0] = frame_data[obj_id]['score'] * 255
                        break

            device = repro_tool.cameraMatrices.device
            valid = (weights[:, 0, 0] > 0).sum().item()
            if valid >= 2:
                center3D = repro_tool.reconstructPoint(
                    pts.to(device), weights.to(device))
                current_centers[fly_idx] = center3D

        if len(current_centers) < 2:
            return False

        prev_0 = prev_fly_centers.get(0)
        prev_1 = prev_fly_centers.get(1)
        curr_0 = current_centers.get(0)
        curr_1 = current_centers.get(1)
        if any(x is None for x in [prev_0, prev_1, curr_0, curr_1]):
            return False

        cost_keep = (torch.norm(curr_0.cpu() - prev_0.cpu()).item()
                     + torch.norm(curr_1.cpu() - prev_1.cpu()).item())
        cost_swap = (torch.norm(curr_0.cpu() - prev_1.cpu()).item()
                     + torch.norm(curr_1.cpu() - prev_0.cpu()).item())

        if cost_swap < cost_keep:
            for cam in range(self._num_cameras):
                id_map = self._identity_map[cam]
                for obj_id in list(id_map.keys()):
                    if id_map[obj_id] == 0:
                        id_map[obj_id] = 1
                    elif id_map[obj_id] == 1:
                        id_map[obj_id] = 0
            return True
        return False

    def _fill_missing_detections(self, assignments, repro_tool, num_animals):
        """For cameras missing a fly, create a synthetic mask from 3D reprojection.

        Uses the triangulated 3D position to project a box into the missing
        camera, then registers that box as an initial mask in the tracker so
        the fly can be tracked even if it wasn't detected on the init frame.
        """
        if self.tracker is None:
            return

        H, W = self._img_size
        # Determine which fly_idx each camera already has
        cam_has_fly = [[False] * num_animals for _ in range(self._num_cameras)]
        for cam in range(self._num_cameras):
            for mapped_fly in self._identity_map[cam].values():
                if mapped_fly < num_animals:
                    cam_has_fly[cam][mapped_fly] = True

        for fly_idx, animal in enumerate(assignments):
            if fly_idx >= num_animals:
                break
            center3D = animal.get('center3D')
            if center3D is None:
                continue

            # Reproject 3D position into all cameras
            reprojected = repro_tool.reprojectPoint(
                center3D.unsqueeze(0)
            )  # (num_cameras, 2)

            for cam in range(self._num_cameras):
                if cam_has_fly[cam][fly_idx]:
                    continue  # already detected

                cx, cy = reprojected[cam].cpu().numpy()
                # Skip if projected point is outside the image
                if cx < 0 or cy < 0 or cx >= W or cy >= H:
                    continue

                # Create a box mask around the reprojected center.
                # Use a generous radius (flies are small ~30-60px).
                radius = max(H, W) // 20
                x0 = max(0, int(cx - radius))
                y0 = max(0, int(cy - radius))
                x1 = min(W, int(cx + radius))
                y1 = min(H, int(cy + radius))

                box_mask = np.zeros((H, W), dtype=np.float32)
                box_mask[y0:y1, x0:x1] = 1.0

                # Assign a new obj_id that doesn't collide with existing ones
                existing_ids = set(self._frame0_masks[cam].keys())
                new_obj_id = max(existing_ids, default=0) + fly_idx + 100

                # Register this mask with the tracker
                inf_state = self._inference_states[cam]
                cap = self._video_caps[cam]
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        self._frame_start + self._init_frame)
                ret, bgr_frame = cap.read()
                if not ret:
                    continue

                frame_tensor = self._preprocess_frame(bgr_frame)
                mask_tensor = torch.from_numpy(box_mask)

                # Re-run preflight after adding new mask
                inf_state["tracking_has_started"] = False
                self.tracker.add_new_mask_with_frame(
                    inf_state,
                    frame_idx=self._init_frame,
                    obj_id=new_obj_id,
                    mask=mask_tensor,
                    frame_tensor=frame_tensor,
                )
                self.tracker.propagate_in_video_preflight(inf_state)

                # Reset capture position
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        self._frame_start + self._init_frame + 1)

                # Update bookkeeping
                self._frame0_masks[cam][new_obj_id] = {
                    'mask': box_mask > 0.5,
                    'centroid': np.array([cx, cy]),
                    'score': 0.3,
                }
                self._identity_map[cam][new_obj_id] = fly_idx
                print(f"    Camera {cam}: filled missing fly {fly_idx} "
                      f"via 3D reprojection at ({cx:.0f}, {cy:.0f})")

    # ── Streaming frame access ���──────────────────────────────────────

    def start_propagation(self):
        """No-op for API compatibility with SAM3StreamingTracker.

        Propagation is handled per-frame in get_next_frame.
        """
        self._current_frame_idx = self._init_frame

    @property
    def init_frame(self):
        """Frame index where detection succeeded."""
        return self._init_frame

    def get_frame0_masks(self, num_animals=2):
        """Get masks for the init frame (from detection, not propagation)."""
        self._prev_centroids = self._frame0_masks  # seed for swap detection
        return self._build_frame_output_ll(self._frame0_masks, num_animals)

    def get_next_frame(self, num_animals=2):
        """Track all objects on the next frame across all cameras.

        Args:
            num_animals: number of animals to return masks for.

        Returns:
            List of dicts per fly (same format as SAM3StreamingTracker),
            or None if all cameras have stopped.
        """
        self._current_frame_idx += 1
        frame_idx = self._current_frame_idx
        current_frame_masks = [{} for _ in range(self._num_cameras)]
        all_stopped = True

        for cam_idx in range(self._num_cameras):
            cap = self._video_caps[cam_idx]
            inf_state = self._inference_states[cam_idx]

            ret, bgr_frame = cap.read()
            if not ret:
                continue
            all_stopped = False

            frame_tensor = self._preprocess_frame(bgr_frame)

            try:
                fi, obj_ids, low_res, video_res, obj_scores = \
                    self.tracker.track_single_frame(
                        inf_state, frame_idx, frame_tensor,
                        trim_memory=True,
                    )
            except RuntimeError as e:
                print(f"    SAM3 tracking error on camera {cam_idx}: {e}")
                continue

            # Convert to per-object mask dict
            for i, obj_id in enumerate(obj_ids):
                mask = (video_res[i, 0] > 0.0).cpu().numpy()
                ys, xs = np.where(mask)
                centroid = (np.array([xs.mean(), ys.mean()])
                            if len(xs) > 0 else np.array([0.0, 0.0]))
                score = torch.sigmoid(obj_scores[i, 0]).item()
                current_frame_masks[cam_idx][int(obj_id)] = {
                    'mask': mask,
                    'centroid': centroid,
                    'score': score,
                }

        if all_stopped:
            return None

        # Detect and correct per-camera identity swaps
        self._correct_identity_swaps(current_frame_masks, num_animals)

        return self._build_frame_output_ll(current_frame_masks, num_animals)

    def _correct_identity_swaps(self, current_frame_masks, num_animals):
        """Detect and fix identity swaps using centroid continuity.

        For each camera, check if the current centroids are closer to the
        *other* fly's previous centroid.  If swapping assignments reduces
        total distance, update the identity map for that camera.

        This handles SAM3's tracker losing identity when flies cross paths.
        """
        if self._prev_centroids is None:
            self._prev_centroids = current_frame_masks
            return
        if num_animals != 2:
            self._prev_centroids = current_frame_masks
            return

        for cam in range(self._num_cameras):
            id_map = self._identity_map[cam]
            cam_masks = current_frame_masks[cam]
            prev_cam = self._prev_centroids[cam]

            # Get the two obj_ids mapped to fly 0 and fly 1
            fly_to_obj = {}
            for obj_id, fly_idx in id_map.items():
                if fly_idx < num_animals:
                    fly_to_obj[fly_idx] = obj_id

            if len(fly_to_obj) < 2:
                continue

            obj_0 = fly_to_obj.get(0)
            obj_1 = fly_to_obj.get(1)
            if obj_0 is None or obj_1 is None:
                continue

            # Current and previous centroids for each tracked obj_id
            curr_0 = cam_masks.get(obj_0, {}).get('centroid')
            curr_1 = cam_masks.get(obj_1, {}).get('centroid')
            prev_0 = prev_cam.get(obj_0, {}).get('centroid')
            prev_1 = prev_cam.get(obj_1, {}).get('centroid')

            if any(c is None for c in [curr_0, curr_1, prev_0, prev_1]):
                continue

            # Cost of keeping current assignment vs swapping
            cost_keep = (np.linalg.norm(curr_0 - prev_0) +
                         np.linalg.norm(curr_1 - prev_1))
            cost_swap = (np.linalg.norm(curr_0 - prev_1) +
                         np.linalg.norm(curr_1 - prev_0))

            if cost_swap < cost_keep * 0.7:  # swap must be clearly better
                id_map[obj_0], id_map[obj_1] = id_map[obj_1], id_map[obj_0]

        self._prev_centroids = current_frame_masks

    def _build_frame_output_ll(self, frame_masks, num_animals):
        """Convert per-camera mask dicts into per-fly format."""
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
        """Release all resources."""
        for cap in self._video_caps:
            try:
                cap.release()
            except Exception:
                pass
        self._video_caps = []
        self._inference_states = []
        self.tracker = None
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
        gc.collect()
        torch.cuda.empty_cache()
