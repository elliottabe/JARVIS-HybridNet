"""
Quick sanity test: extract frames from Session0 mp4s at the start of bout 1
(frame 14045) on all 7 cameras, run SAM3 with text prompt "insect", and save
overlay visualizations so we can eyeball whether SAM3 reliably segments flies
on this camera setup.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

SESSION = Path("/data2/users/eabe/datasets/Johnson_lab/courtship/Session0/2025_10_20_13_20_04")
OUT = Path("/tmp/sam3_sanity_session0")
CAMS = [f"Cam{c}" for c in ("2012630", "2012631", "2012853", "2012855",
                             "2012857", "2012861", "2012862")]
FRAME_IDX = 14300  # middle of bout 1 (14045–14557)
TEXT_PROMPT = "insect"
CONF = 0.5

COLORS = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
    [0, 255, 255], [255, 128, 0],
], dtype=np.uint8)


def extract_frame(video_path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def overlay(rgb: np.ndarray, masks: torch.Tensor, boxes: torch.Tensor,
            scores: torch.Tensor) -> np.ndarray:
    out = rgb.copy()
    masks_np = masks.squeeze(1).cpu().numpy()
    for i, m in enumerate(masks_np):
        color = COLORS[i % len(COLORS)]
        overlay_layer = out.copy()
        overlay_layer[m] = color
        out = cv2.addWeighted(out, 0.55, overlay_layer, 0.45, 0)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    boxes_np = boxes.float().cpu().numpy()
    scores_np = scores.float().cpu().numpy()
    for i, (b, s) in enumerate(zip(boxes_np, scores_np)):
        x0, y0, x1, y1 = [int(v) for v in b]
        color = COLORS[i % len(COLORS)].tolist()
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, f"{s:.2f}", (x0, max(y0 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Building SAM3 image model …")
    model = build_sam3_image_model(device="cuda", load_from_HF=True)
    proc = Sam3Processor(model, resolution=1008, device="cuda",
                         confidence_threshold=CONF)

    for cam in CAMS:
        vp = SESSION / f"{cam}.mp4"
        rgb = extract_frame(vp, FRAME_IDX)
        print(f"{cam}: frame {FRAME_IDX} shape={rgb.shape}")

        pil = Image.fromarray(rgb)
        state = proc.set_image(pil)
        state = proc.set_text_prompt(TEXT_PROMPT, state)

        n_det = int(state["masks"].shape[0])
        scores = state["scores"].float()
        print(f"  detections at conf≥{CONF}: {n_det}"
              + (f"  scores={[round(float(s), 3) for s in scores.cpu()]}"
                 if n_det > 0 else ""))

        vis = overlay(rgb, state["masks"], state["boxes"], state["scores"])
        cv2.imwrite(str(OUT / f"{cam}_frame{FRAME_IDX}.jpg"), vis)

    print(f"Wrote overlays to {OUT}")


if __name__ == "__main__":
    main()
