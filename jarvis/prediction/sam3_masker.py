"""
SAM3 (Segment Anything 3) wrapper for precise fly segmentation masking.

Provides pixel-accurate masks to replace crude rectangular masking in the
multi-animal tracking pipeline, and optionally constrains keypoint detection
to within the segmented fly body.
"""

import sys

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

# SAM3 is an external dependency — add its path if not installed system-wide
SAM3_PATH = "/home/eabe/Research/Github/sam3"
if SAM3_PATH not in sys.path:
    sys.path.insert(0, SAM3_PATH)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def dilate_mask(mask, kernel_size=15):
    """Dilate a binary mask using max pooling.

    Args:
        mask: (H, W) bool tensor
        kernel_size: dilation kernel size (odd number)
    Returns:
        (H, W) bool tensor with dilated mask
    """
    padding = kernel_size // 2
    return F.max_pool2d(
        mask.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size, stride=1, padding=padding
    ).bool().squeeze(0).squeeze(0)


class SAM3Masker:
    """
    Wraps SAM3 image model to produce per-camera segmentation masks
    for detected flies.

    Args:
        device: device for SAM3 model ('cuda', 'cuda:0', 'cuda:1', etc.)
        confidence_threshold: minimum SAM3 detection confidence
        text_prompt: text description for the object to segment
        dilation_kernel: kernel size for mask dilation (0 to disable)
    """

    def __init__(self, device='cuda', confidence_threshold=0.3,
                 text_prompt='fly', dilation_kernel=15,
                 detect_confidence_threshold=0.15):
        self.device = device
        self.dilation_kernel = dilation_kernel
        self.detect_confidence_threshold = detect_confidence_threshold

        print(f"  Loading SAM3 model on {device}...")
        # Load on default cuda first — SAM3's tokenizer has a device bug
        # where it creates tensors on CPU that fail when the model is on
        # a non-default GPU (e.g. cuda:1).
        self.model = build_sam3_image_model(
            device='cuda',
            eval_mode=True,
            load_from_HF=True,
        )

        # Pre-compute text encoding on default cuda (tokenizer works here)
        self.text_prompt = text_prompt
        with torch.inference_mode():
            self.cached_text_outputs = self.model.backbone.forward_text(
                [text_prompt], device='cuda'
            )

        # Move model to target device if different from default cuda
        if device != 'cuda':
            self.model = self.model.to(device)
            self.cached_text_outputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in self.cached_text_outputs.items()
            }
            # Clear lazy coordinate caches in decoder layers -- these are
            # plain attributes (not registered buffers) that get stuck on
            # the original device after .to().
            for module in self.model.modules():
                if hasattr(module, 'compilable_cord_cache'):
                    module.compilable_cord_cache = None
                if hasattr(module, 'coord_cache'):
                    module.coord_cache = {}

        self.processor = Sam3Processor(
            self.model, resolution=1008, device=device,
            confidence_threshold=confidence_threshold,
        )
        print(f"  SAM3 loaded. Text prompt: '{text_prompt}'")

    @torch.inference_mode()
    def _segment_single_camera(self, img_tensor, center_xy, bbox_hw):
        """
        Run SAM3 on a single camera image with a box prompt around the
        detected center.

        Args:
            img_tensor: (3, H, W) float tensor [0, 1] on any device
            center_xy: (2,) int tensor [cx, cy] in pixel coords
            bbox_hw: int, half bounding box size for prompt box

        Returns:
            (H, W) bool tensor on same device as input — True = fly pixels
            Returns None if no detection.
        """
        H, W = img_tensor.shape[1], img_tensor.shape[2]

        # Convert to PIL for Sam3Processor
        img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(img_np)

        # Run backbone
        state = self.processor.set_image(pil_img)

        # Inject cached text features
        state["backbone_out"].update(self.cached_text_outputs)

        # Create box prompt: [cx, cy, w, h] normalized to [0, 1]
        cx = center_xy[0].item() / W
        cy = center_xy[1].item() / H
        bw = (2 * bbox_hw) / W
        bh = (2 * bbox_hw) / H
        # Clamp to valid range
        cx = max(0.01, min(0.99, cx))
        cy = max(0.01, min(0.99, cy))

        state = self.processor.add_geometric_prompt(
            box=[cx, cy, bw, bh], label=True, state=state
        )

        # Extract masks
        if "masks" not in state or state["masks"] is None or len(state["masks"]) == 0:
            return None

        masks = state["masks"]       # (N, 1, H, W) bool
        scores = state["scores"]     # (N,) float

        if len(masks) == 1:
            mask = masks[0, 0]  # (H, W)
        else:
            # Multiple detections: pick the one closest to the center
            target = torch.tensor(
                [center_xy[0].item(), center_xy[1].item()],
                device=masks.device, dtype=torch.float32,
            )
            best_idx = 0
            best_dist = float('inf')
            for i in range(len(masks)):
                ys, xs = torch.where(masks[i, 0])
                if len(xs) == 0:
                    continue
                centroid = torch.tensor(
                    [xs.float().mean(), ys.float().mean()],
                    device=masks.device,
                )
                dist = torch.norm(centroid - target).item()
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            mask = masks[best_idx, 0]  # (H, W)

        # Dilate the mask to provide margin
        if self.dilation_kernel > 0:
            mask = dilate_mask(mask, self.dilation_kernel)

        return mask.to(img_tensor.device)

    @torch.inference_mode()
    def compute_masks(self, imgs, centerHMs, bbox_hw):
        """
        Compute SAM3 segmentation masks for a detected fly across all cameras.

        Args:
            imgs: (num_cameras, 3, H, W) float tensor [0, 1]
            centerHMs: (num_cameras, 2) int tensor of detected center per camera
            bbox_hw: int, half bounding box size

        Returns:
            masks: (num_cameras, H, W) bool tensor — True = fly pixels
        """
        num_cameras = imgs.shape[0]
        H, W = imgs.shape[2], imgs.shape[3]
        masks = torch.zeros(num_cameras, H, W, dtype=torch.bool, device=imgs.device)

        for cam in range(num_cameras):
            mask = self._segment_single_camera(
                imgs[cam], centerHMs[cam], bbox_hw
            )
            if mask is not None:
                masks[cam] = mask

        return masks

    @torch.inference_mode()
    def detect_all_flies(self, imgs):
        """
        Detect ALL flies in each camera using text-only prompt (no box needed).

        Uses SAM3's grounded detection with the cached text prompt to find
        every fly in each camera view simultaneously.

        Args:
            imgs: (num_cameras, 3, H, W) float tensor [0, 1]

        Returns:
            list of per-camera dicts, each with:
                'masks': (N_i, H, W) bool tensor — one mask per detected fly
                'centroids': (N_i, 2) float tensor — [cx, cy] pixel coords
                'scores': (N_i,) float tensor — detection confidence
            N_i may differ per camera.
        """
        num_cameras = imgs.shape[0]
        H, W = imgs.shape[2], imgs.shape[3]
        results = []

        # Temporarily lower confidence threshold for text-only detection
        orig_thresh = self.processor.confidence_threshold
        self.processor.confidence_threshold = self.detect_confidence_threshold

        for cam in range(num_cameras):
            img_np = (imgs[cam].cpu().permute(1, 2, 0).numpy() * 255
                      ).astype(np.uint8)
            pil_img = PIL.Image.fromarray(img_np)

            # Run backbone on image
            state = self.processor.set_image(pil_img)

            # Inject cached text features (avoids re-computing text encoding)
            state["backbone_out"].update(self.cached_text_outputs)

            # Set dummy geometric prompt (text-only detection)
            state["geometric_prompt"] = self.model._get_dummy_prompt()

            # Run grounded detection
            state = self.processor._forward_grounding(state)

            if ("masks" not in state or state["masks"] is None
                    or len(state["masks"]) == 0):
                results.append({
                    'masks': torch.empty(0, H, W, dtype=torch.bool,
                                         device=imgs.device),
                    'centroids': torch.empty(0, 2, device=imgs.device),
                    'scores': torch.empty(0, device=imgs.device),
                })
                continue

            masks = state["masks"][:, 0]   # (N, H, W) bool
            scores = state["scores"]       # (N,) float

            # Compute mask centroids
            centroids = []
            for i in range(len(masks)):
                ys, xs = torch.where(masks[i])
                if len(xs) > 0:
                    centroids.append(torch.tensor(
                        [xs.float().mean(), ys.float().mean()]))
                else:
                    centroids.append(torch.tensor([0.0, 0.0]))
            centroids = torch.stack(centroids)

            # Dilate masks for margin
            if self.dilation_kernel > 0:
                dilated = []
                for i in range(len(masks)):
                    dilated.append(dilate_mask(masks[i], self.dilation_kernel))
                masks = torch.stack(dilated)

            results.append({
                'masks': masks.to(imgs.device),
                'centroids': centroids.to(imgs.device),
                'scores': scores.to(imgs.device),
            })

        # Restore original threshold
        self.processor.confidence_threshold = orig_thresh
        return results

    @staticmethod
    def crop_mask_for_heatmap(full_mask, centerHM, bbox_hw, heatmap_size,
                              dilation_kernel=21):
        """
        Crop a full-resolution SAM3 mask to the bounding box region and resize
        to 2D heatmap resolution for constraining keypoint detection.

        Args:
            full_mask: (H, W) bool tensor
            centerHM: (2,) int tensor [cx, cy]
            bbox_hw: int, half bounding box size
            heatmap_size: tuple (H_hm, W_hm)
            dilation_kernel: additional dilation for keypoint slack (0=none)

        Returns:
            (H_hm, W_hm) float tensor in [0, 1] — mask at heatmap resolution
        """
        H, W = full_mask.shape
        cx, cy = centerHM[0].item(), centerHM[1].item()

        y1 = max(0, cy - bbox_hw)
        y2 = min(H, cy + bbox_hw)
        x1 = max(0, cx - bbox_hw)
        x2 = min(W, cx + bbox_hw)

        crop = full_mask[y1:y2, x1:x2].float()

        if crop.numel() == 0:
            return torch.ones(heatmap_size, device=full_mask.device)

        # Resize to heatmap resolution
        crop_resized = F.interpolate(
            crop.unsqueeze(0).unsqueeze(0),
            size=heatmap_size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)

        # Binarize and optionally dilate for keypoint slack
        mask = (crop_resized > 0.3)
        if dilation_kernel > 0:
            mask = dilate_mask(mask, dilation_kernel)

        return mask.float()
