"""
Multi-animal 3D predictor for JARVIS-HybridNet.

Extends the single-animal JarvisPredictor3D to detect and reconstruct
N animals per frame by extracting multiple peaks from CenterDetect
heatmaps and running HybridNet independently for each detected animal.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.utils.reprojection import ReprojectionTool


class JarvisMultiAnimalPredictor3D(nn.Module):
    """
    Multi-animal version of JarvisPredictor3D.

    Runs CenterDetect once per frame, extracts top-K peaks via NMS, then
    runs HybridNet independently for each detected animal center.

    Args:
        cfg: project configuration
        num_animals: number of animals to detect per frame
        suppression_radius: NMS radius in heatmap pixels
        weights_center_detect: path to CenterDetect weights or 'latest'
        weights_hybridnet: path to HybridNet weights or 'latest'
        trt_mode: TensorRT mode ('off', 'new', 'previous')
        confidence_threshold: minimum heatmap confidence to consider a camera
            detection valid (used to count how many cameras see the animal).
            Lower values allow weaker detections through; reconstructPoint
            naturally downweights low-confidence cameras via its weighting.
        mask_scale: fraction of bounding box size to use when masking a
            detected animal before searching for the next one. 1.5 masks
            a region 1.5x the bounding box in each direction from the
            detected center, which fully covers the fly's heatmap response.
            Too small (e.g. 0.5) causes re-detection of the same animal.
            Ignored when use_sam3_mask=True.
        use_sam3_mask: use SAM3 segmentation for precise fly masking
            instead of rectangular mask. Falls back to rectangular if
            SAM3 is not available.
        sam3_device: device for SAM3 model ('cuda', 'cuda:1', etc.)
        sam3_text_prompt: text prompt for SAM3 segmentation
        sam3_constrain_keypoints: if True and SAM3 is active, also use
            SAM3 masks to constrain keypoint detection within the fly body
    """

    def __init__(self, cfg, num_animals=2, suppression_radius=4,
                 weights_center_detect='latest', weights_hybridnet='latest',
                 trt_mode='off', confidence_threshold=0.5, mask_scale=1.5,
                 use_sam3_mask=False, sam3_device='cuda',
                 sam3_text_prompt='fly', sam3_constrain_keypoints=False,
                 sam3_detect_confidence=0.15,
                 multi_peak_trained=False,
                 min_animal_separation_mm=0.0):
        super(JarvisMultiAnimalPredictor3D, self).__init__()
        self.cfg = cfg
        self.num_animals = num_animals
        self.suppression_radius = suppression_radius
        self.confidence_threshold = confidence_threshold
        self.mask_scale = mask_scale
        self.sam3_constrain_keypoints = sam3_constrain_keypoints
        self.multi_peak_trained = multi_peak_trained
        # Identity-collapse guard passed to multi_peak.assign_peaks_across_cameras.
        # When > 0, any pair of animals whose triangulated 3D centers are
        # within this distance is dropped down to a single detection so the
        # tracker cannot lock two tracks onto the same physical animal.
        self.min_animal_separation_mm = float(min_animal_separation_mm)

        # Optional SAM3 masker
        self.sam3_masker = None
        if use_sam3_mask:
            try:
                from jarvis.prediction.sam3_masker import SAM3Masker
                self.sam3_masker = SAM3Masker(
                    device=sam3_device,
                    text_prompt=sam3_text_prompt,
                    detect_confidence_threshold=sam3_detect_confidence,
                )
                print("  SAM3 masking enabled")
            except (ImportError, Exception) as e:
                print(f"  WARNING: SAM3 not available ({e}), "
                      "falling back to rectangular mask")

        self.centerDetect = EfficientTrack(
            'CenterDetectInference', self.cfg, weights_center_detect
        ).model
        self.hybridNet = HybridNet(
            'inference', self.cfg, weights_hybridnet
        ).model

        self.register_buffer('transform_mean', torch.tensor(
            self.cfg.DATASET.MEAN, device=torch.device('cuda')
        ).view(3, 1, 1))
        self.register_buffer('transform_std', torch.tensor(
            self.cfg.DATASET.STD, device=torch.device('cuda')
        ).view(3, 1, 1))
        self.bbox_hw = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE / 2)
        self.num_cameras = self.cfg.HYBRIDNET.NUM_CAMERAS
        self.bounding_box_size = self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE

        self.reproTool = ReprojectionTool()
        self.center_detect_img_size = int(self.cfg.CENTERDETECT.IMAGE_SIZE)

        if trt_mode == 'new':
            self._compile_trt_models()
        elif trt_mode == 'previous':
            self._load_trt_models()

    def _load_trt_models(self):
        import torch_tensorrt
        transpose2D_lib_dir = os.path.join(
            self.cfg.PARENT_DIR, 'libs',
            'conv_transpose2d_converter.cpython-39-x86_64-linux-gnu.so'
        )
        transpose3D_lib_dir = os.path.join(
            self.cfg.PARENT_DIR, 'libs',
            'conv_transpose3d_converter.cpython-39-x86_64-linux-gnu.so'
        )
        torch.ops.load_library(transpose3D_lib_dir)
        torch.ops.load_library(transpose2D_lib_dir)

        trt_path = os.path.join(
            self.cfg.PARENT_DIR, 'projects',
            self.cfg.PROJECT_NAME, 'trt-models', 'predict3D'
        )
        self.centerDetect = torch.jit.load(
            os.path.join(trt_path, 'centerDetect.pt')
        )
        self.hybridNet.effTrack = torch.jit.load(
            os.path.join(trt_path, 'keypointDetect.pt')
        )
        self.hybridNet.v2vNet = torch.jit.load(
            os.path.join(trt_path, 'hybridNet.pt')
        )

    def _compile_trt_models(self):
        import torch_tensorrt
        transpose2D_lib_dir = os.path.join(
            self.cfg.PARENT_DIR, 'libs',
            'conv_transpose2d_converter.cpython-39-x86_64-linux-gnu.so'
        )
        transpose3D_lib_dir = os.path.join(
            self.cfg.PARENT_DIR, 'libs',
            'conv_transpose3d_converter.cpython-39-x86_64-linux-gnu.so'
        )
        torch.ops.load_library(transpose3D_lib_dir)
        torch.ops.load_library(transpose2D_lib_dir)

        trt_path = os.path.join(
            self.cfg.PARENT_DIR, 'projects',
            self.cfg.PROJECT_NAME, 'trt-models', 'predict3D'
        )
        os.makedirs(trt_path, exist_ok=True)

        self.centerDetect = self.centerDetect.eval().cuda()
        traced_model = torch.jit.trace(
            self.centerDetect,
            [torch.randn((1, 3, 256, 256)).to("cuda")]
        )
        self.centerDetect = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input(
                (self.cfg.HYBRIDNET.NUM_CAMERAS, 3,
                 self.cfg.CENTERDETECT.IMAGE_SIZE,
                 self.cfg.CENTERDETECT.IMAGE_SIZE),
                dtype=torch.float
            )],
            enabled_precisions={torch.half},
        )
        torch.jit.save(
            self.centerDetect,
            os.path.join(trt_path, 'centerDetect.pt')
        )

        self.hybridNet.effTrack.eval().cuda()
        traced_model = torch.jit.trace(
            self.hybridNet.effTrack,
            [torch.randn((1, 3, self.bounding_box_size,
                          self.bounding_box_size)).to("cuda")]
        )
        self.hybridNet.effTrack = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input(
                (self.cfg.HYBRIDNET.NUM_CAMERAS, 3,
                 self.bounding_box_size, self.bounding_box_size),
                dtype=torch.float
            )],
            enabled_precisions={torch.half}
        )
        torch.jit.save(
            self.hybridNet.effTrack,
            os.path.join(trt_path, 'keypointDetect.pt')
        )

        self.hybridNet.v2vNet.eval().cuda()
        grid_size = int(
            self.cfg.HYBRIDNET.ROI_CUBE_SIZE / self.cfg.HYBRIDNET.GRID_SPACING
        )
        traced_model = torch.jit.trace(
            self.hybridNet.v2vNet,
            [torch.randn((1, self.cfg.KEYPOINTDETECT.NUM_JOINTS,
                          grid_size, grid_size, grid_size)).to("cuda")]
        )
        self.hybridNet.v2vNet = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input(
                (1, self.cfg.KEYPOINTDETECT.NUM_JOINTS,
                 grid_size, grid_size, grid_size),
                dtype=torch.float
            )],
            enabled_precisions={torch.half}
        )
        torch.jit.save(
            self.hybridNet.v2vNet,
            os.path.join(trt_path, 'hybridNet.pt')
        )

    def _detect_center(self, imgs, imgs_resized):
        """
        Run CenterDetect and extract the strongest peak.

        Returns:
            center3D: (3,) tensor, or None if detection failed
            centerHMs: (num_cameras, 2) int tensor of reprojected centers
            maxvals: (num_cameras, 1, 1) confidence values
        """
        outputs = self.centerDetect(imgs_resized)
        heatmaps_gpu = outputs[1].view(
            outputs[1].shape[0], outputs[1].shape[1], -1
        )
        m = heatmaps_gpu.argmax(2).view(
            heatmaps_gpu.shape[0], heatmaps_gpu.shape[1], 1
        )
        preds = torch.cat(
            (m % outputs[1].shape[2], m // outputs[1].shape[3]), dim=2
        )
        maxvals = heatmaps_gpu.gather(2, m)

        img_size = torch.tensor(
            [imgs.shape[3], imgs.shape[2]], device=torch.device('cuda')
        )
        downsampling_scale = torch.tensor(
            [imgs.shape[3] / float(self.center_detect_img_size),
             imgs.shape[2] / float(self.center_detect_img_size)],
            device=torch.device('cuda')
        ).float()

        num_cams_detect = torch.numel(maxvals[maxvals > self.confidence_threshold])
        maxvals_normed = maxvals / 255.

        if num_cams_detect < 2:
            return None, None, None

        center3D = self.reproTool.reconstructPoint(
            (preds.reshape(self.num_cameras, 2)
             * (downsampling_scale * 2)).transpose(0, 1),
            maxvals_normed
        )
        centerHMs = self.reproTool.reprojectPoint(
            center3D.unsqueeze(0)
        ).int()
        centerHMs[:, 0] = torch.clamp(
            centerHMs[:, 0], self.bbox_hw, img_size[0] - self.bbox_hw
        )
        centerHMs[:, 1] = torch.clamp(
            centerHMs[:, 1], self.bbox_hw, img_size[1] - self.bbox_hw
        )

        return center3D, centerHMs, maxvals

    def _run_hybridnet(self, imgs, center3D, centerHMs, cameraMatrices,
                       sam3_masks=None):
        """
        Run HybridNet for a single detected animal center.

        Args:
            imgs: (num_cameras, 3, H, W) original images
            center3D: (3,) detected 3D center
            centerHMs: (num_cameras, 2) reprojected 2D centers
            cameraMatrices: camera projection matrices
            sam3_masks: optional (num_cameras, H, W) bool masks for keypoint
                constraining. If provided and sam3_constrain_keypoints is True,
                used to mask cropped images and/or 2D heatmaps.

        Returns:
            dict with 'center3D', 'points3D', 'confidences'
        """
        img_size = torch.tensor(
            [imgs.shape[3], imgs.shape[2]], device=torch.device('cuda')
        )

        imgs_cropped = torch.zeros(
            (self.num_cameras, 3,
             self.bounding_box_size, self.bounding_box_size),
            device=torch.device('cuda')
        )

        for i in range(self.num_cameras):
            imgs_cropped[i] = imgs[
                i, :,
                centerHMs[i, 1] - self.bbox_hw:centerHMs[i, 1] + self.bbox_hw,
                centerHMs[i, 0] - self.bbox_hw:centerHMs[i, 0] + self.bbox_hw
            ]

        use_sam3_kp = (sam3_masks is not None and self.sam3_constrain_keypoints)

        if use_sam3_kp:
            # Crop-level masking: zero out non-fly pixels before keypoint detection
            for i in range(self.num_cameras):
                y1 = centerHMs[i, 1].item() - self.bbox_hw
                y2 = centerHMs[i, 1].item() + self.bbox_hw
                x1 = centerHMs[i, 0].item() - self.bbox_hw
                x2 = centerHMs[i, 0].item() + self.bbox_hw
                H, W = imgs.shape[2], imgs.shape[3]
                # Clamp to image bounds
                sy1 = max(0, y1) - y1
                sy2 = self.bounding_box_size - max(0, y2 - H)
                sx1 = max(0, x1) - x1
                sx2 = self.bounding_box_size - max(0, x2 - W)
                my1, my2 = max(0, y1), min(H, y2)
                mx1, mx2 = max(0, x1), min(W, x2)
                crop_mask = sam3_masks[i, my1:my2, mx1:mx2]
                # Expand mask for keypoint slack (wings, legs extend slightly)
                from jarvis.prediction.sam3_masker import dilate_mask
                crop_mask = dilate_mask(crop_mask, kernel_size=21)
                # Apply: zero out pixels outside the fly
                full_crop_mask = torch.zeros(
                    self.bounding_box_size, self.bounding_box_size,
                    dtype=torch.bool, device=imgs.device
                )
                full_crop_mask[sy1:sy2, sx1:sx2] = crop_mask
                imgs_cropped[i][:, ~full_crop_mask] = 0

        imgs_cropped = (
            (imgs_cropped - self.transform_mean) / self.transform_std
        )

        if use_sam3_kp:
            # Split HybridNet forward pass to apply heatmap-level masking
            # Step 1: Run effTrack for 2D heatmaps
            heatmaps_batch = self.hybridNet.effTrack(imgs_cropped)[1]
            # Shape: (num_cameras, num_joints, H_hm, W_hm)

            # Step 2: Apply SAM3 mask to heatmaps
            hm_h, hm_w = heatmaps_batch.shape[2], heatmaps_batch.shape[3]
            for i in range(self.num_cameras):
                from jarvis.prediction.sam3_masker import SAM3Masker
                hm_mask = SAM3Masker.crop_mask_for_heatmap(
                    sam3_masks[i], centerHMs[i], self.bbox_hw,
                    (hm_h, hm_w), dilation_kernel=11,
                )
                heatmaps_batch[i] *= hm_mask.unsqueeze(0)

            # Step 3: Continue with reprojection + V2VNet
            heatmaps_batch = heatmaps_batch.unsqueeze(0)  # add batch dim
            heatmaps_padded = F.pad(
                heatmaps_batch, [1, 1, 1, 1], mode='constant', value=0.
            )
            heatmaps3D = self.hybridNet.reproLayer(
                heatmaps_padded,
                center3D.int().unsqueeze(0),
                centerHMs.unsqueeze(0),
                cameraMatrices.unsqueeze(0),
            )
            heatmap_final = self.hybridNet.v2vNet(heatmaps3D / 255.)
            heatmap_final = self.hybridNet.softplus(heatmap_final)

            # Extract 3D keypoints (same as HybridNetBackbone.forward)
            norm = torch.sum(heatmap_final, dim=[2, 3, 4])
            x = torch.sum(
                torch.mul(heatmap_final, self.hybridNet.xx), dim=[2, 3, 4]
            ) / norm
            y = torch.sum(
                torch.mul(heatmap_final, self.hybridNet.yy), dim=[2, 3, 4]
            ) / norm
            z = torch.sum(
                torch.mul(heatmap_final, self.hybridNet.zz), dim=[2, 3, 4]
            ) / norm
            points3D = torch.stack([x, y, z], dim=2)
            confidences = torch.clamp(
                torch.max(
                    heatmap_final.view(*heatmap_final.shape[:2], -1), dim=2
                )[0],
                max=255.,
            ) / 255.
            points3D = (
                points3D.transpose(0, 1) * self.hybridNet.grid_spacing * 2
                - self.hybridNet.grid_size / 2.
                + center3D.int().unsqueeze(0)
            ).transpose(0, 1)
        else:
            # Standard HybridNet forward pass
            _, _, points3D, confidences = self.hybridNet(
                imgs_cropped.unsqueeze(0),
                img_size,
                centerHMs.unsqueeze(0),
                center3D.int().unsqueeze(0),
                cameraMatrices.unsqueeze(0),
            )

        return {
            'center3D': center3D,
            'points3D': points3D.squeeze(),
            'confidences': confidences.squeeze(),
        }

    def forward(self, imgs, cameraMatrices, precomputed_masks=None):
        """
        Detect and reconstruct multiple animals.

        Args:
            imgs: tensor (num_cameras, 3, H, W) of camera images
            cameraMatrices: tensor of camera projection matrices
            precomputed_masks: optional list of per-fly dicts from
                BoutMasks.get_frame(), each with 'masks', 'centroids', 'valid'

        Returns:
            list of dicts, one per detected animal, each containing:
                'center3D': (3,) tensor
                'points3D': (num_joints, 3) tensor
                'confidences': (num_joints,) tensor
        """
        self.reproTool.cameraMatrices = cameraMatrices

        # Ensure normalization buffers are on the same device as input
        if self.transform_mean.device != imgs.device:
            self.transform_mean = self.transform_mean.to(imgs.device)
            self.transform_std = self.transform_std.to(imgs.device)

        if precomputed_masks is not None:
            return self._forward_with_precomputed_masks(
                imgs, cameraMatrices, precomputed_masks
            )
        elif self.multi_peak_trained:
            return self._forward_multi_peak_only(imgs, cameraMatrices)
        elif self.sam3_masker is not None:
            return self._forward_multiview(imgs, cameraMatrices)
        else:
            return self._forward_mask_and_redetect(imgs, cameraMatrices)

    def _forward_with_precomputed_masks(self, imgs, cameraMatrices,
                                         precomputed_masks):
        """
        Use pre-computed SAM3 video masks for identity + constraining,
        CenterDetect for optimized crop centering.

        Args:
            precomputed_masks: list of per-fly dicts from BoutMasks.get_frame()
        """
        from jarvis.prediction.multi_peak import extract_top_k_peaks

        num_cameras = self.num_cameras
        k = self.num_animals
        img_size = torch.tensor(
            [imgs.shape[3], imgs.shape[2]], device=torch.device('cuda')
        )

        # Step 1: Run CenterDetect for optimal crop centers
        imgs_resized = transforms.functional.resize(
            imgs, [self.center_detect_img_size, self.center_detect_img_size]
        )
        imgs_resized = (
            (imgs_resized - self.transform_mean) / self.transform_std
        )
        cd_outputs = self.centerDetect(imgs_resized)
        cd_heatmaps = cd_outputs[1]  # (num_cameras, 1, H_cd, W_cd)

        cd_downsampling = torch.tensor(
            [imgs.shape[3] / float(self.center_detect_img_size),
             imgs.shape[2] / float(self.center_detect_img_size)],
            device=torch.device('cuda'),
        ).float()

        cd_peaks, cd_maxvals = extract_top_k_peaks(
            cd_heatmaps, k=k,
            suppression_radius=self.suppression_radius,
        )
        cd_peaks_img = cd_peaks * (cd_downsampling * 2).unsqueeze(0).unsqueeze(0)

        # Step 2: For each fly, match to nearest CenterDetect peak
        results = []
        used_cd_peaks = set()

        for fly_idx, fly_data in enumerate(precomputed_masks):
            fly_centroids = fly_data['centroids'].cuda()  # (num_cameras, 2)
            fly_valid = fly_data['valid']  # (num_cameras,)
            fly_masks_np = fly_data['masks']  # (num_cameras, H, W) bool

            num_valid = fly_valid.sum().item()
            if num_valid < 2:
                continue

            # Find nearest CenterDetect peak using mean 2D distance
            best_cd_idx = None
            best_dist = float('inf')
            for ci in range(k):
                if ci in used_cd_peaks:
                    continue
                dist = torch.norm(
                    cd_peaks_img[ci].cuda() - fly_centroids, dim=1
                )
                # Only consider cameras where this fly is valid
                valid_mask = fly_valid.cuda()
                if valid_mask.sum() > 0:
                    mean_dist = dist[valid_mask].mean().item()
                else:
                    mean_dist = float('inf')
                if mean_dist < best_dist:
                    best_dist = mean_dist
                    best_cd_idx = ci
            if best_cd_idx is not None:
                used_cd_peaks.add(best_cd_idx)

            # Use CenterDetect peak for triangulation + centering
            if best_cd_idx is not None:
                cd_mvals = cd_maxvals[best_cd_idx].cuda()
                cd_mvals_normed = cd_mvals / 255.
                num_detect = torch.sum(
                    cd_mvals.squeeze() > self.confidence_threshold
                ).item()
                if num_detect >= 2:
                    center3D = self.reproTool.reconstructPoint(
                        cd_peaks_img[best_cd_idx].cuda().transpose(0, 1),
                        cd_mvals_normed.unsqueeze(1),
                    )
                else:
                    # Fall back to SAM3 centroids for triangulation
                    mvals = torch.where(
                        fly_valid.cuda().unsqueeze(1),
                        torch.ones(num_cameras, 1, device='cuda') * 255,
                        torch.zeros(num_cameras, 1, device='cuda'),
                    )
                    center3D = self.reproTool.reconstructPoint(
                        fly_centroids.transpose(0, 1),
                        (mvals / 255.).unsqueeze(1),
                    )
            else:
                # No CD peak — use SAM3 centroids
                mvals = torch.where(
                    fly_valid.cuda().unsqueeze(1),
                    torch.ones(num_cameras, 1, device='cuda') * 255,
                    torch.zeros(num_cameras, 1, device='cuda'),
                )
                center3D = self.reproTool.reconstructPoint(
                    fly_centroids.transpose(0, 1),
                    (mvals / 255.).unsqueeze(1),
                )

            centerHMs = self.reproTool.reprojectPoint(
                center3D.unsqueeze(0)
            ).int()
            centerHMs[:, 0] = torch.clamp(
                centerHMs[:, 0], self.bbox_hw, img_size[0] - self.bbox_hw
            )
            centerHMs[:, 1] = torch.clamp(
                centerHMs[:, 1], self.bbox_hw, img_size[1] - self.bbox_hw
            )

            # Convert SAM3 masks to cuda tensor for keypoint constraining
            sam3_masks = fly_masks_np.cuda() if self.sam3_constrain_keypoints else None

            if not torch.isfinite(center3D).all():
                continue

            result = self._run_hybridnet(
                imgs, center3D, centerHMs, cameraMatrices,
                sam3_masks=sam3_masks,
            )
            results.append(result)

        return results

    def _forward_multiview(self, imgs, cameraMatrices):
        """
        Hybrid multi-view detection: SAM3 finds all flies and assigns
        identities via multi-view triangulation, then CenterDetect provides
        optimized crop centers for HybridNet keypoint quality.

        SAM3 masks are used for keypoint constraining (crop-level and
        heatmap-level masking).
        """
        from jarvis.prediction.multi_peak import (
            extract_top_k_peaks, assign_peaks_across_cameras,
        )

        num_cameras = self.num_cameras
        k = self.num_animals
        img_size = torch.tensor(
            [imgs.shape[3], imgs.shape[2]], device=torch.device('cuda')
        )

        # Step 1: SAM3 text-only detection on all cameras
        cam_detections = self.sam3_masker.detect_all_flies(imgs)

        # Step 2: Build SAM3 peak tensors for cross-camera assignment
        sam3_peaks = torch.zeros(k, num_cameras, 2, device=imgs.device)
        sam3_maxvals = torch.zeros(k, num_cameras, 1, device=imgs.device)

        for cam in range(num_cameras):
            det = cam_detections[cam]
            n_det = len(det['scores'])
            if n_det == 0:
                continue
            sorted_idx = det['scores'].argsort(descending=True)[:k]
            for pi, si in enumerate(sorted_idx):
                sam3_peaks[pi, cam] = det['centroids'][si]
                sam3_maxvals[pi, cam, 0] = det['scores'][si] * 255

        # Step 3: SAM3 cross-camera assignment to get 3D centers
        # SAM3 centroids are in image coords; use downsampling_scale=0.5
        # so the *2 scaling in assign_peaks_across_cameras is identity.
        downsampling_scale = torch.tensor([0.5, 0.5], device=imgs.device)
        assignments = assign_peaks_across_cameras(
            sam3_peaks, sam3_maxvals, self.reproTool, downsampling_scale,
            confidence_threshold=self.confidence_threshold,
            min_animal_separation_mm=self.min_animal_separation_mm,
        )

        if not assignments:
            return []

        # Step 4: Run CenterDetect to get optimized crop centers
        # CenterDetect produces heatmaps trained specifically for HybridNet
        # cropping, giving better keypoint quality than SAM3 centroids.
        imgs_resized = transforms.functional.resize(
            imgs, [self.center_detect_img_size, self.center_detect_img_size]
        )
        imgs_resized = (
            (imgs_resized - self.transform_mean) / self.transform_std
        )
        cd_outputs = self.centerDetect(imgs_resized)
        cd_heatmaps = cd_outputs[1]  # (num_cameras, 1, H_cd, W_cd)

        cd_downsampling = torch.tensor(
            [imgs.shape[3] / float(self.center_detect_img_size),
             imgs.shape[2] / float(self.center_detect_img_size)],
            device=torch.device('cuda'),
        ).float()

        # Extract top-k CenterDetect peaks
        cd_peaks, cd_maxvals = extract_top_k_peaks(
            cd_heatmaps, k=k,
            suppression_radius=self.suppression_radius,
        )
        # Scale CenterDetect peaks to image coords
        cd_peaks_img = cd_peaks * (cd_downsampling * 2).unsqueeze(0).unsqueeze(0)

        # Step 5: For each SAM3-assigned animal, find the nearest
        # CenterDetect peak and use it for HybridNet cropping.
        results = []
        used_cd_peaks = set()

        for animal in assignments:
            sam3_center3D = animal['center3D']
            sam3_2d = animal['points2D']  # (num_cameras, 2) image coords

            # Find closest CenterDetect peak to this animal's SAM3 position
            # Compare in 3D: triangulate each CD peak and pick nearest to
            # SAM3 3D center. Simpler: compare mean 2D distance across cameras.
            best_cd_idx = None
            best_dist = float('inf')
            for ci in range(k):
                if ci in used_cd_peaks:
                    continue
                # Mean distance across cameras between CD peak and SAM3 center
                dist = torch.norm(
                    cd_peaks_img[ci] - sam3_2d, dim=1
                ).mean().item()
                if dist < best_dist:
                    best_dist = dist
                    best_cd_idx = ci
            if best_cd_idx is not None:
                used_cd_peaks.add(best_cd_idx)

            # Use CenterDetect peak for triangulation + centering
            if best_cd_idx is not None:
                cd_mvals = cd_maxvals[best_cd_idx]  # (num_cameras, 1)
                cd_mvals_normed = cd_mvals / 255.
                num_detect = torch.sum(
                    cd_mvals.squeeze() > self.confidence_threshold
                ).item()
                if num_detect >= 2:
                    center3D = self.reproTool.reconstructPoint(
                        cd_peaks_img[best_cd_idx].transpose(0, 1),
                        cd_mvals_normed.unsqueeze(1),
                    )
                    centerHMs = self.reproTool.reprojectPoint(
                        center3D.unsqueeze(0)
                    ).int()
                else:
                    # CenterDetect didn't see this fly well — use SAM3 center
                    center3D = sam3_center3D
                    centerHMs = self.reproTool.reprojectPoint(
                        center3D.unsqueeze(0)
                    ).int()
            else:
                # No CD peak available — use SAM3 center
                center3D = sam3_center3D
                centerHMs = self.reproTool.reprojectPoint(
                    center3D.unsqueeze(0)
                ).int()

            # Clamp centers to valid crop range
            centerHMs[:, 0] = torch.clamp(
                centerHMs[:, 0], self.bbox_hw, img_size[0] - self.bbox_hw
            )
            centerHMs[:, 1] = torch.clamp(
                centerHMs[:, 1], self.bbox_hw, img_size[1] - self.bbox_hw
            )

            # Build SAM3 masks for this animal from detection results
            sam3_masks = None
            if self.sam3_constrain_keypoints:
                sam3_masks = torch.zeros(
                    num_cameras, imgs.shape[2], imgs.shape[3],
                    dtype=torch.bool, device=imgs.device,
                )
                for cam in range(num_cameras):
                    det = cam_detections[cam]
                    if len(det['centroids']) == 0:
                        continue
                    # Pick mask closest to the assigned 2D center
                    dists = torch.norm(
                        det['centroids'].to(imgs.device)
                        - centerHMs[cam].float().unsqueeze(0),
                        dim=1,
                    )
                    best_idx = dists.argmin().item()
                    sam3_masks[cam] = det['masks'][best_idx]

            if not torch.isfinite(center3D).all():
                continue

            result = self._run_hybridnet(
                imgs, center3D, centerHMs, cameraMatrices,
                sam3_masks=sam3_masks,
            )
            results.append(result)

        return results

    def _forward_multi_peak_only(self, imgs, cameraMatrices):
        """
        Pure multi-peak CenterDetect inference -- no SAM3 needed.

        Runs CenterDetect once, extracts top-K peaks via NMS, assigns
        peaks across cameras using multi-view geometry, then runs HybridNet
        for each detected animal.

        Requires CenterDetect to be trained on multi-fly data so the
        heatmap naturally contains one peak per animal.
        """
        from jarvis.prediction.multi_peak import (
            extract_top_k_peaks, assign_peaks_across_cameras,
        )

        img_size = torch.tensor(
            [imgs.shape[3], imgs.shape[2]], device=imgs.device
        )

        # Step 1: Run CenterDetect (single forward pass)
        imgs_resized = transforms.functional.resize(
            imgs, [self.center_detect_img_size, self.center_detect_img_size]
        )
        imgs_resized = (
            (imgs_resized - self.transform_mean) / self.transform_std
        )
        cd_outputs = self.centerDetect(imgs_resized)
        cd_heatmaps = cd_outputs[1]  # (num_cameras, 1, H_cd, W_cd)

        downsampling_scale = torch.tensor(
            [imgs.shape[3] / float(self.center_detect_img_size),
             imgs.shape[2] / float(self.center_detect_img_size)],
            device=imgs.device,
        ).float()

        # Step 2: Extract top-K peaks via iterative NMS
        peaks, maxvals = extract_top_k_peaks(
            cd_heatmaps, k=self.num_animals,
            suppression_radius=self.suppression_radius,
        )

        # Step 3: Assign peaks consistently across cameras
        assignments = assign_peaks_across_cameras(
            peaks, maxvals, self.reproTool, downsampling_scale,
            confidence_threshold=self.confidence_threshold,
            min_animal_separation_mm=self.min_animal_separation_mm,
        )

        # Step 4: Run HybridNet for each assigned animal
        results = []
        for animal in assignments:
            center3D = animal['center3D']
            centerHMs = self.reproTool.reprojectPoint(
                center3D.unsqueeze(0)
            ).int()
            centerHMs[:, 0] = torch.clamp(
                centerHMs[:, 0], self.bbox_hw, img_size[0] - self.bbox_hw
            )
            centerHMs[:, 1] = torch.clamp(
                centerHMs[:, 1], self.bbox_hw, img_size[1] - self.bbox_hw
            )

            if not torch.isfinite(center3D).all():
                continue

            result = self._run_hybridnet(
                imgs, center3D, centerHMs, cameraMatrices,
            )
            results.append(result)

        return results

    def _forward_mask_and_redetect(self, imgs, cameraMatrices):
        """
        Fallback: mask-and-redetect approach using CenterDetect.

        For each animal beyond the first, the previously detected animal's
        bounding box region is masked in the images before re-running
        CenterDetect.
        """
        results = []
        working_imgs = imgs

        for animal_idx in range(self.num_animals):
            imgs_resized = transforms.functional.resize(
                working_imgs,
                [self.center_detect_img_size, self.center_detect_img_size]
            )
            imgs_resized = (
                (imgs_resized - self.transform_mean) / self.transform_std
            )

            center3D, centerHMs, maxvals = self._detect_center(
                working_imgs, imgs_resized
            )

            if center3D is None:
                break

            if not torch.isfinite(center3D).all():
                continue

            result = self._run_hybridnet(
                imgs, center3D, centerHMs, cameraMatrices,
            )
            results.append(result)

            # Mask detected animal for next iteration
            if animal_idx < self.num_animals - 1:
                working_imgs = working_imgs.clone()
                mask_hw = int(self.bbox_hw * self.mask_scale)
                for i in range(self.num_cameras):
                    cx = centerHMs[i, 0].item()
                    cy = centerHMs[i, 1].item()
                    x1 = max(0, cx - mask_hw)
                    x2 = min(imgs.shape[3], cx + mask_hw)
                    y1 = max(0, cy - mask_hw)
                    y2 = min(imgs.shape[2], cy + mask_hw)
                    working_imgs[i, :, y1:y2, x1:x2] = 0

        return results
