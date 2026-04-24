"""TaskPipeline for M-067_tn3k_thyroid_nodule_segmentation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from core.pipeline import BasePipeline, TaskSample
from src.download.downloader import create_downloader
from .config import TaskConfig
from .transforms import (
    create_overlay_bbox,
    create_overlay_mask,
    loop_frames,
    make_video,
)


TARGET_SIZE = 1024


def _resize_pad_square(img: np.ndarray, target: int = TARGET_SIZE):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_h = target - nh
    pad_w = target - nw
    out = cv2.copyMakeBorder(out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))
    return out, scale


def _resize_mask(mask: np.ndarray, target: int = TARGET_SIZE):
    h, w = mask.shape[:2]
    scale = target / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    out = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    pad_h = target - nh
    pad_w = target - nw
    return cv2.copyMakeBorder(out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                              value=0)


class TaskPipeline(BasePipeline):
    def __init__(self, config=None):
        super().__init__(config or TaskConfig())
        self.downloader = create_downloader(self.config)
    def download(self):
        yield from self.downloader.iter_samples(limit=self.config.num_samples)
    def process_sample(self, raw, idx):
        img = cv2.imread(str(raw["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(raw["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: return None
        img_r, _ = _resize_pad_square(img)
        mask_r = _resize_mask(mask)
        has = bool((mask_r > 0).any())
        first = loop_frames(img_r, n=self.config.fps)
        last = [create_overlay_mask(f, mask_r, color=(0, 0, 255)) for f in first]
        gt = last if has else last[:5]
        sid = f"{self.config.domain}_{idx:05d}"
        out = Path(self.config.output_dir) / f"{self.config.domain}_task" / sid
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out/"first_frame.png"), first[0])
        cv2.imwrite(str(out/"final_frame.png"), last[0])
        make_video(first, out/"first_video.mp4", self.config.fps)
        make_video(last, out/"last_video.mp4", self.config.fps)
        make_video(gt, out/"ground_truth.mp4", self.config.fps)
        prompt = f"{self.config.task_prompt}\n\nGround-truth: {'POSITIVE — nodule visible.' if has else 'NEGATIVE — no nodule.'}"
        (out/"prompt.txt").write_text(prompt + "\n")
        meta = {"image_id": raw["image_id"], "split": raw.get("split"),
                 "has_nodule": has, "fps": self.config.fps,
                 "frames_per_video": len(first), "case_type": "D_single_image_loop"}
        (out/"metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        return TaskSample(task_id=sid, domain=self.config.domain, prompt=prompt,
                          first_image=first[0], final_image=last[0],
                          first_video=str(out/"first_video.mp4"),
                          last_video=str(out/"last_video.mp4"),
                          ground_truth_video=str(out/"ground_truth.mp4"),
                          metadata=meta)
    def run(self):
        out = []
        for idx, r in enumerate(self.download()):
            s = self.process_sample(r, idx)
            if s is not None: out.append(s)
        return out
