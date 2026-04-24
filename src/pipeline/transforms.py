"""Video + overlay primitives (Linux-safe ffmpeg, plus seg/bbox overlays)."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def loop_frames(image: np.ndarray, n: int = 12) -> List[np.ndarray]:
    return [image.copy() for _ in range(n)]


def create_overlay_mask(
    image: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha: float = 0.45
) -> np.ndarray:
    """Blend a binary mask (any non-zero = positive) onto BGR image."""
    out = image.copy()
    if mask is None:
        return out
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return out
    overlay = np.zeros_like(out)
    overlay[binary == 1] = color
    return cv2.addWeighted(out, 1.0, overlay, alpha, 0)


def create_overlay_bbox(
    image: np.ndarray, bboxes: List[Tuple[float, float, float, float, str]],
    color_map=None,
) -> np.ndarray:
    out = image.copy()
    color_map = color_map or {}
    for x, y, w, h, label in bboxes:
        x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        color = color_map.get(label, (200, 200, 200))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        ty = max(th + 4, y - 6)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label, (x + 3, ty - 2), font, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out


def make_video(frames: List[np.ndarray], out_path: Path, fps: int = 12) -> None:
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
