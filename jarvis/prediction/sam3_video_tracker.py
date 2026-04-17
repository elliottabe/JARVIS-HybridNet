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
import time as _time

import cv2
import numpy as np
import torch
from tqdm import tqdm

# SAM3 is an external dependency
SAM3_PATH = "/home/eabe/Research/Github/sam3"
if SAM3_PATH not in sys.path:
    sys.path.insert(0, SAM3_PATH)

import gc


class _LogTqdm:
    """Drop-in for tqdm.tqdm that emits periodic log lines instead of a
    live progress bar. SAM3 propagation (sam3_multiplex_tracking,
    sam3_video_inference, io_utils frame loaders) wraps per-frame loops in
    `from tqdm.auto import tqdm`, which in non-tty SLURM logs produces
    thousands of useless carriage-return-flood lines. Logging every
    ``LOG_EVERY`` items OR ``LOG_INTERVAL_S`` seconds (whichever comes
    first), plus one line at start and finish, gives tractable output.
    """
    LOG_EVERY = 500
    LOG_INTERVAL_S = 30.0

    def __init__(self, iterable=None, desc=None, total=None,
                 disable=False, **_kw):
        self.iterable = iterable
        self.desc = desc or "iter"
        self.total = total if total is not None else (
            len(iterable) if hasattr(iterable, "__len__") else None)
        self.disable = disable
        self.n = 0
        self._t0 = None
        self._t_last = None

    def _emit(self, msg):
        if not self.disable:
            print(f"[tqdm] {msg}", flush=True)

    def __iter__(self):
        self._t0 = _time.time()
        self._t_last = self._t0
        tot = f"/{self.total}" if self.total else ""
        self._emit(f"{self.desc}: start (total={self.total})")
        last_logged = 0
        for item in self.iterable:
            yield item
            self.n += 1
            now = _time.time()
            if (self.n - last_logged >= self.LOG_EVERY
                    or now - self._t_last >= self.LOG_INTERVAL_S):
                rate = self.n / max(now - self._t0, 1e-6)
                self._emit(f"{self.desc}: {self.n}{tot}  "
                           f"({rate:.1f} it/s, {now - self._t0:.1f}s)")
                last_logged = self.n
                self._t_last = now
        elapsed = _time.time() - (self._t0 or _time.time())
        self._emit(f"{self.desc}: done {self.n}{tot} in {elapsed:.1f}s")

    def update(self, n=1):
        if self._t0 is None:
            self._t0 = _time.time()
            self._t_last = self._t0
            self._emit(f"{self.desc}: start (total={self.total})")
        self.n += n
        now = _time.time()
        if now - self._t_last >= self.LOG_INTERVAL_S:
            rate = self.n / max(now - self._t0, 1e-6)
            tot = f"/{self.total}" if self.total else ""
            self._emit(f"{self.desc}: {self.n}{tot}  "
                       f"({rate:.1f} it/s, {now - self._t0:.1f}s)")
            self._t_last = now

    def close(self):
        if self._t0 is not None:
            elapsed = _time.time() - self._t0
            tot = f"/{self.total}" if self.total else ""
            self._emit(f"{self.desc}: done {self.n}{tot} in {elapsed:.1f}s")
            self._t0 = None

    def set_description(self, d, refresh=None):
        self.desc = d

    def set_postfix(self, *a, **k):
        pass

    def refresh(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _patch_tqdm_for_sam3():
    """Replace tqdm.tqdm and tqdm.auto.tqdm so SAM3's `from tqdm.auto import
    tqdm` picks up _LogTqdm. Safe to call before sam3 is imported; modules
    that already did `from tqdm import tqdm` before this runs keep the
    original reference (e.g. jarvis training code)."""
    import tqdm as _tqdm_pkg
    import tqdm.auto as _tqdm_auto
    _tqdm_pkg.tqdm = _LogTqdm
    _tqdm_auto.tqdm = _LogTqdm


_patch_tqdm_for_sam3()

from sam3.model_builder import build_sam3_predictor, build_sam3_video_predictor


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
        Establish cross-camera identity using multi-view triangulation.

        SAM3 assigns independent obj_ids per camera. This method finds
        which obj_id in each camera corresponds to the same physical fly.

        Two-step anchor-based matching:

        (1) Scan up to 200 frames and rank candidates by the tuple
            (cams_ok, min_sep) — cams with ≥num_animals masks, and the
            minimum per-camera pairwise centroid separation of the
            top-num_animals masks (pixels). A well-separated anchor makes
            the enumeration below numerically stable.

        (2) At the anchor, enumerate 2^(C-1) per-camera peak orderings
            (fixing the first camera's ordering as reference). For each
            bitmask, triangulate each animal's 3D center across all
            cameras-with-data, reproject back, sum per-(animal, camera)
            2D residuals, and pick the bitmask minimizing total residual.
            If the runner-up bitmask's residual is within 2× of the
            winner's, fall back to the next-best anchor candidate from
            the scan (up to 5 attempts). Without this exhaustive search,
            the older naive Step-1 triangulation inside
            assign_peaks_across_cameras settles on 5-vs-2 mixed orderings
            on ambiguous anchors, locking in a per-camera identity swap
            for the whole bout.

        Per-camera sort tiebreak `(-score, cx, cy, obj_id)` only affects
        which two masks we keep when a camera has >num_animals objects —
        the enumeration makes the final ordering independent of this
        tiebreak.
        """
        dev = repro_tool.cameraMatrices.device

        # --- Step 1: rank anchor candidates -------------------------
        def _frame_separation(f):
            min_sep = float('inf')
            cams_ok = 0
            for cam in range(self.num_cameras):
                fd = self.masks[cam][f]
                if len(fd) < num_animals:
                    continue
                cams_ok += 1
                top = sorted(
                    fd.items(),
                    key=lambda x: (-x[1]['score'],
                                   float(x[1]['centroid'][0]),
                                   float(x[1]['centroid'][1]),
                                   int(x[0])),
                )[:num_animals]
                cents = np.stack([d['centroid'] for _, d in top])
                for i in range(num_animals):
                    for j in range(i + 1, num_animals):
                        d = float(np.linalg.norm(cents[i] - cents[j]))
                        if d < min_sep:
                            min_sep = d
            return cams_ok, (min_sep if min_sep < float('inf') else 0.0)

        scan_limit = min(self.num_frames, 200)
        ranked = []  # list of (cams_ok, min_sep, frame)
        for f in range(scan_limit):
            cams_ok, sep = _frame_separation(f)
            if cams_ok >= 2:  # need 2 cams for a triangulation
                ranked.append((cams_ok, sep, f))
        if not ranked:
            print("  [assign_identities] WARNING: no frame with ≥2 cams "
                  "having ≥num_animals masks; identity map will be empty")
            for cam in range(self.num_cameras):
                self.identity_map[cam] = {}
            return
        # Descending by (cams_ok, sep)
        ranked.sort(key=lambda r: (-r[0], -r[1]))

        # --- Step 2: per-anchor enumeration -------------------------
        if num_animals != 2:
            raise NotImplementedError(
                f"assign_identities enumeration only handles num_animals=2, "
                f"got {num_animals}"
            )

        def _evaluate_anchor(anchor_frame):
            # Collect top-num_animals masks per camera at anchor_frame.
            # Returns None if fewer than 2 cameras have enough masks.
            cams_with_data = []
            peaks_list = []  # peaks_list[i] ~ tensor (num_animals, 2) on dev
            obj_ids_list = []  # obj_ids_list[i] ~ list of int
            for cam in range(self.num_cameras):
                fd = self.masks[cam][anchor_frame]
                if len(fd) < num_animals:
                    continue
                sorted_objs = sorted(
                    fd.items(),
                    key=lambda x: (-x[1]['score'],
                                   float(x[1]['centroid'][0]),
                                   float(x[1]['centroid'][1]),
                                   int(x[0])),
                )[:num_animals]
                cents = np.stack([d['centroid'] for _, d in sorted_objs])
                cams_with_data.append(cam)
                peaks_list.append(torch.as_tensor(cents, dtype=torch.float32,
                                                  device=dev))
                obj_ids_list.append([int(oid) for oid, _ in sorted_objs])
            n_cd = len(cams_with_data)
            if n_cd < 2:
                return None

            # For each bitmask, compute total 2D reprojection residual.
            # Bit i (0-indexed, 0..n_cd-2) swaps cams_with_data[i+1]'s ordering.
            num_bm = 1 << (n_cd - 1)
            residuals_by_bm = np.zeros(num_bm, dtype=np.float64)
            # Cache per-cam 2D points per-animal as we go
            mvs_template = torch.zeros(self.num_cameras, 1, 1, device=dev)
            for i, cam in enumerate(cams_with_data):
                mvs_template[cam, 0, 0] = 1.0

            for bm in range(num_bm):
                # Resolve per-cam local ordering for this bitmask
                local_idx_a0 = np.empty(n_cd, dtype=np.int64)
                for i in range(n_cd):
                    swap = 0 if i == 0 else ((bm >> (i - 1)) & 1)
                    local_idx_a0[i] = swap  # a0's local peak index
                local_idx_a1 = 1 - local_idx_a0  # a1 takes the other peak

                total = 0.0
                for animal_idx, local_idx_arr in (
                        (0, local_idx_a0), (1, local_idx_a1)):
                    pts = torch.zeros(self.num_cameras, 2, device=dev)
                    for i, cam in enumerate(cams_with_data):
                        pts[cam] = peaks_list[i][int(local_idx_arr[i])]
                    # reconstructPoint wants (2, num_cameras) and
                    # (num_cameras, 1, 1) maxvals.
                    center3D = repro_tool.reconstructPoint(
                        pts.t().contiguous(), mvs_template)
                    reproj = repro_tool.reprojectPoint(
                        center3D.unsqueeze(0)).squeeze(0)  # (num_cams, 2)
                    for i, cam in enumerate(cams_with_data):
                        delta = reproj[cam] - peaks_list[i][
                            int(local_idx_arr[i])]
                        total += float(torch.linalg.norm(delta).item())
                residuals_by_bm[bm] = total

            order = np.argsort(residuals_by_bm)
            winner_bm = int(order[0])
            winner_res = float(residuals_by_bm[winner_bm])
            runner_up_res = (float(residuals_by_bm[int(order[1])])
                             if len(order) > 1 else winner_res * 10.0)
            margin = (runner_up_res / winner_res) if winner_res > 1e-6 else float('inf')

            # Store the 3D centers from the winner so we can reuse them
            # below when building identity_map for cams not in cams_with_data.
            winner_local_a0 = np.empty(n_cd, dtype=np.int64)
            for i in range(n_cd):
                swap = 0 if i == 0 else ((winner_bm >> (i - 1)) & 1)
                winner_local_a0[i] = swap
            winner_local_a1 = 1 - winner_local_a0

            centers3D = []
            for animal_idx, local_idx_arr in (
                    (0, winner_local_a0), (1, winner_local_a1)):
                pts = torch.zeros(self.num_cameras, 2, device=dev)
                for i, cam in enumerate(cams_with_data):
                    pts[cam] = peaks_list[i][int(local_idx_arr[i])]
                c3d = repro_tool.reconstructPoint(
                    pts.t().contiguous(), mvs_template)
                centers3D.append(c3d)

            return {
                'cams_with_data': cams_with_data,
                'peaks_list': peaks_list,
                'obj_ids_list': obj_ids_list,
                'winner_bm': winner_bm,
                'winner_res': winner_res,
                'runner_up_res': runner_up_res,
                'margin': margin,
                'winner_local_a0': winner_local_a0,
                'winner_local_a1': winner_local_a1,
                'centers3D': centers3D,
            }

        MAX_FALLBACKS = 5
        MARGIN_THRESH = 2.0
        attempts = []
        for cams_ok, sep, f in ranked[:MAX_FALLBACKS]:
            r = _evaluate_anchor(f)
            if r is None:
                continue
            attempts.append((f, cams_ok, sep, r))
            if r['margin'] >= MARGIN_THRESH:
                break
        if not attempts:
            print("  [assign_identities] WARNING: no evaluable anchor; "
                  "identity map will be empty")
            for cam in range(self.num_cameras):
                self.identity_map[cam] = {}
            return

        # Prefer the attempt with the highest runner-up margin.
        anchor_frame, cams_ok, sep, result = max(
            attempts, key=lambda a: a[3]['margin'])

        winner_bm = result['winner_bm']
        cams_with_data = result['cams_with_data']
        peaks_list = result['peaks_list']
        obj_ids_list = result['obj_ids_list']
        n_cd = len(cams_with_data)
        swapped_cams = [cams_with_data[i] for i in range(1, n_cd)
                        if (winner_bm >> (i - 1)) & 1]

        print(f"  [assign_identities] anchor frame={anchor_frame} "
              f"(cams_ok={cams_ok}, min_sep={sep:.1f}px) "
              f"swapped_cams={swapped_cams} "
              f"total_res={result['winner_res']:.1f}px "
              f"runner_up_ratio={result['margin']:.2f}x")
        if result['margin'] < MARGIN_THRESH:
            print(f"  [assign_identities] WARNING: runner-up margin "
                  f"{result['margin']:.2f}x < {MARGIN_THRESH}x after "
                  f"{len(attempts)} anchor attempts — identity may be "
                  f"ambiguous.")

        # --- Step 3: build identity_map ----------------------------
        for cam in range(self.num_cameras):
            self.identity_map[cam] = {}

        # Direct mapping for cams with data at anchor (we already know
        # which local peak index is fly 0 vs fly 1).
        for i, cam in enumerate(cams_with_data):
            a0_local = int(result['winner_local_a0'][i])
            a1_local = int(result['winner_local_a1'][i])
            self.identity_map[cam][obj_ids_list[i][a0_local]] = 0
            self.identity_map[cam][obj_ids_list[i][a1_local]] = 1

        # For cams NOT in cams_with_data (had <num_animals masks at
        # anchor), reproject both 3D centers into that cam and match
        # each surviving obj_id to the closer fly.
        centers3D = result['centers3D']
        for cam in range(self.num_cameras):
            if cam in cams_with_data:
                continue
            frame_data = self.masks[cam][anchor_frame]
            if not frame_data:
                continue
            reproj_per_fly = []
            for c3d in centers3D:
                r = repro_tool.reprojectPoint(
                    c3d.unsqueeze(0)).squeeze(0)  # (num_cameras, 2)
                reproj_per_fly.append(r[cam].detach().cpu().numpy())
            for obj_id, data in frame_data.items():
                dists = [float(np.linalg.norm(data['centroid'] - rf))
                         for rf in reproj_per_fly]
                self.identity_map[cam][int(obj_id)] = int(np.argmin(dists))

        # --- Step 4: male-as-fly0 via mask area ---------------------
        # Female D. melanogaster are ~15-20% larger than males. Sum
        # mask pixels per fly_idx across every-10th frame × all cams
        # and swap identity_map if fly0 is the larger (= female).
        area = np.zeros(num_animals, dtype=np.float64)
        frame_step = max(1, self.num_frames // 50)
        for cam in range(self.num_cameras):
            id_map = self.identity_map[cam]
            if not id_map:
                continue
            for f in range(0, self.num_frames, frame_step):
                for obj_id, data in self.masks[cam][f].items():
                    fly_idx = id_map.get(int(obj_id))
                    if fly_idx is None or fly_idx >= num_animals:
                        continue
                    area[fly_idx] += float(data['mask'].sum())
        a0, a1 = area[0], area[1]
        denom = min(a0, a1)
        ratio = (max(a0, a1) / denom) if denom > 0 else 0.0
        if a0 > 0 and a1 > 0 and ratio >= 1.05:
            if a0 > a1:
                for cam in range(self.num_cameras):
                    self.identity_map[cam] = {
                        oid: (1 - fi)
                        for oid, fi in self.identity_map[cam].items()
                    }
                print(f"  [assign_identities] sex-ID: "
                      f"area[0]={a0:.0f} area[1]={a1:.0f} "
                      f"ratio={ratio:.2f} -> swap (fly0=male)")
            else:
                print(f"  [assign_identities] sex-ID: "
                      f"area[0]={a0:.0f} area[1]={a1:.0f} "
                      f"ratio={ratio:.2f} (fly0 already smaller=male)")
        else:
            print(f"  [assign_identities] sex-ID: area[0]={a0:.0f} "
                  f"area[1]={a1:.0f} ratio={ratio:.2f} < 1.05 — "
                  f"ambiguous, keeping enumfix ordering")

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
        apply_temporal_disambiguation: base-SAM3 hotstart-delay toggle
            (ignored by SAM 3.1 multiplex, which always applies it)
        sam3_version: 'sam3.1' (multiplex, default) or 'sam3' (base)
        compile: torch.compile the multiplex backbones (SAM 3.1 only)
        max_num_objects: multiplex max tracked objects (SAM 3.1 only)
        checkpoint_path: explicit ckpt path; None = auto-download from HF
        use_fa3: enable FlashAttention-3 kernels (SAM 3.1 only). Default
            False because the prebuilt `flash_attn_3` wheel shipped with
            this env has an ABI mismatch; the FA2 fallback still works.
    """

    def __init__(self, gpu_id=1, text_prompt='insect',
                 apply_temporal_disambiguation=False,
                 sam3_version='sam3.1', compile=True,
                 max_num_objects=16, checkpoint_path=None,
                 use_fa3=False):
        self.gpu_id = gpu_id
        self.text_prompt = text_prompt
        self.sam3_version = sam3_version

        print(f"  Loading SAM3 video predictor ({sam3_version}) "
              f"on cuda:{gpu_id}...")
        # Multiplex places weights on the current CUDA device (no
        # gpus_to_use kwarg), so pin the device for construction.
        with torch.cuda.device(gpu_id):
            if sam3_version == 'sam3.1':
                self.predictor = build_sam3_predictor(
                    version='sam3.1',
                    compile=compile,
                    warm_up=False,
                    max_num_objects=max_num_objects,
                    multiplex_count=16,
                    checkpoint_path=checkpoint_path,
                    use_fa3=use_fa3,
                )
                # Async loader spawns a daemon thread whose current CUDA
                # device defaults to cuda:0, placing frames there while
                # the model lives on cuda:{gpu_id}. Force sync loading so
                # the torch.cuda.device(gpu_id) context in process_bout
                # governs frame placement too.
                self.predictor.async_loading_frames = False
            else:
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

        # Multiplex (SAM 3.1) creates helper tensors (token ids, pos
        # embeddings) on the current CUDA device. JARVIS runs on cuda:0
        # and flips the default, so pin the SAM3 GPU for the whole bout.
        try:
            with torch.cuda.device(self.gpu_id):
                self._process_bout_inner(
                    video_paths, frame_start, num_frames, num_animals,
                    bout_masks, tmp_base)
        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)

        return bout_masks

    def _process_bout_inner(self, video_paths, frame_start, num_frames,
                            num_animals, bout_masks, tmp_base):
        # Thread-local CUDA current device — belt-and-suspenders alongside
        # the torch.cuda.device(self.gpu_id) context wrapping this call,
        # so any sam3 internals that resolve torch.device("cuda") pick up
        # the right GPU.
        torch.cuda.set_device(self.gpu_id)
        for cam_idx, video_path in enumerate(video_paths):
            cam_name = os.path.splitext(
                os.path.basename(video_path)
            )[0]
            cam_dir = os.path.join(tmp_base, cam_name)

            print(f"    Extracting frames for {cam_name}...")
            self._extract_bout_frames(
                video_path, frame_start, num_frames, cam_dir
            )

            start_req = {
                'type': 'start_session',
                'resource_path': cam_dir,
            }
            # SAM 3.1 multiplex keeps backbone/memory features in GPU
            # memory for every propagated frame. On ~2000-frame bouts
            # that pushes past a 48 GiB A6000. Push per-frame buffers
            # to CPU RAM so VRAM stays bounded by the current frame.
            if self.sam3_version == 'sam3.1':
                start_req['offload_video_to_cpu'] = True
            response = self.predictor.handle_request(start_req)
            session_id = response['session_id']

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

            self.predictor.handle_request({
                'type': 'close_session',
                'session_id': session_id,
            })
            print(f"    {cam_name}: done")


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
