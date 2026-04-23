"""M-127 — Thyroid US cine nodule TRACKING + size-change classification.

Stanford AIMI Thyroid Ultrasound Cine-clip dataset (public S3 mirror).
HDF5 layout (verified against the real file):

    annot_id   (N,)          — bytes, e.g. b'1_', b'2_' ... one per frame
    frame_num  (N,)          — int/bytes, frame index within the clip
    image      (N, H, W)     — grayscale frames (uint8/float)
    mask       (N, H, W)     — binary nodule mask

One sample = one annot_id = one real ultrasound cine clip.

Overlays (distinct from M-124's static lesion localization):
  * Red bounding box of the nodule (from the mask) on every frame.
  * Per-frame size text in the top-left ("area: <px>").
  * White motion arrow from the previous frame's bbox-center to the current.
  * Last frame shows summary text with % area change + stable/growing/shrinking.
"""
from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from core.pipeline import BasePipeline, TaskSample
from src.download.downloader import create_downloader
from src.pipeline.config import TaskConfig


# ────────────────────────────────── helpers ────────────────────────────────

def _normalize_frame(img: np.ndarray) -> np.ndarray:
    """Grayscale -> BGR uint8 via min/max normalization."""
    img = np.asarray(img).astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.astype(np.uint8)


def _resize_keep(img: np.ndarray, mask: np.ndarray, max_side: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, mask.astype(np.uint8)
    scale = max_side / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(
        mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST
    )
    return img_r, mask_r


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) of the largest connected component, or None."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return None
    # Skip background (label 0); pick the largest by area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    x, y, w, h, _ = stats[best]
    return int(x), int(y), int(x + w), int(y + h)


def _bbox_center(bb: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bb
    return (x1 + x2) // 2, (y1 + y2) // 2


def _put_label(frame: np.ndarray, text: str, org: Tuple[int, int],
               color=(255, 255, 255), scale: float = 0.5, thickness: int = 1,
               bg=(0, 0, 0)) -> None:
    """Put a readable label with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(
        frame,
        (x - 2, y - th - 4),
        (x + tw + 2, y + baseline),
        bg,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y - 2), font, scale, color,
                thickness, cv2.LINE_AA)


def _classify_delta(initial: int, final: int) -> str:
    if initial <= 0:
        if final <= 0:
            return "STABLE"
        return "GROWING"
    pct = (final - initial) / initial
    if pct > 0.10:
        return "GROWING"
    if pct < -0.10:
        return "SHRINKING"
    return "STABLE"


def _percent_change(initial: int, final: int) -> float:
    if initial <= 0:
        return 0.0
    return 100.0 * (final - initial) / initial


def _render_overlay(
    base: np.ndarray,
    mask: np.ndarray,
    prev_center: Optional[Tuple[int, int]],
    frame_idx: int,
    total_frames: int,
    area_px: int,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """Draw bbox (red), size text + motion arrow (white)."""
    out = base.copy()
    bb = _bbox_from_mask(mask)
    center = None
    if bb is not None:
        x1, y1, x2, y2 = bb
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        center = _bbox_center(bb)
        # Motion arrow from previous center -> current center.
        if prev_center is not None:
            cv2.arrowedLine(
                out, prev_center, center, (255, 255, 255), 2,
                line_type=cv2.LINE_AA, tipLength=0.3,
            )
    # Frame counter top-right, area + nodule-state top-left.
    _put_label(out, f"frame {frame_idx+1}/{total_frames}",
               (out.shape[1] - 150, 22))
    if area_px > 0:
        _put_label(out, f"area: {area_px}px", (8, 22), color=(0, 0, 255))
    else:
        _put_label(out, "nodule: not visible", (8, 22), color=(0, 200, 255))
    return out, center


def _make_video(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    # ffmpeg via stdin, H.264 for QuickTime/Chrome compatibility.
    pad_w = w + (w & 1)
    pad_h = h + (h & 1)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-vf", f"pad={pad_w}:{pad_h}:0:0:color=black",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        for f in frames:
            if f.ndim == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return
    except FileNotFoundError:
        pass
    # Fallback: cv2 mp4v writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for f in frames:
            if f.ndim == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            vw.write(f)
    finally:
        vw.release()


# ─────────────────────────────── pipeline ──────────────────────────────────

class TaskPipeline(BasePipeline):
    def __init__(self, config: Optional[TaskConfig] = None):
        super().__init__(config or TaskConfig())
        self.task_config: TaskConfig = self.config  # type: ignore[assignment]
        self.downloader = create_downloader(self.task_config)
        self._h5_path: Optional[Path] = None
        self._clip_groups: dict = {}
        self._meta_rows: dict = {}

    # ── raw + clip grouping ───────────────────────────────────────────
    @staticmethod
    def _decode(v) -> str:
        if isinstance(v, (bytes, bytearray, np.bytes_)):
            try:
                return v.decode()
            except Exception:
                return v.decode("latin-1", errors="ignore")
        return str(v)

    def _group_clips(self, h5_path: Path) -> dict:
        with h5py.File(h5_path, "r") as f:
            annot = f["annot_id"][:]
            frame_num = f["frame_num"][:]
        groups = defaultdict(list)
        for i, a in enumerate(annot):
            aid = self._decode(a)
            try:
                fn = int(self._decode(frame_num[i]))
            except Exception:
                fn = 0
            groups[aid].append((fn, i))
        for aid in groups:
            groups[aid].sort(key=lambda x: x[0])
        return {aid: [ri for _, ri in vs] for aid, vs in groups.items()}

    def _load_metadata_csv(self, csv_path: Path) -> dict:
        out = {}
        if not csv_path.exists():
            return out
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                aid = (row.get("annot_id") or "").strip()
                if not aid:
                    continue
                out[aid] = row
                out[aid.rstrip("_")] = row
        return out

    # ── BasePipeline hooks ───────────────────────────────────────────
    def download(self) -> Iterator[dict]:
        # The downloader yields one dict with h5+csv paths; we open once here.
        for payload in self.downloader.iter_samples(limit=self.task_config.num_samples):
            h5_path = Path(payload["h5_path"])
            csv_path = Path(payload["csv_path"])
            if not h5_path.exists():
                # Try to locate any dataset.hdf5 under raw_dir/
                alt = list(Path(self.task_config.raw_dir).rglob("dataset.hdf5"))
                if alt:
                    h5_path = alt[0]
                else:
                    print(f"[err] dataset.hdf5 not found (looked at {h5_path})")
                    return
            self._h5_path = h5_path
            self._meta_rows = self._load_metadata_csv(csv_path)
            print(f"[intro] opening {h5_path}")
            with h5py.File(h5_path, "r") as f:
                for k in f.keys():
                    ds = f[k]
                    try:
                        print(f"[intro]   {k}: shape={ds.shape} dtype={ds.dtype}")
                    except Exception:
                        print(f"[intro]   {k}: (non-dataset)")
            self._clip_groups = self._group_clips(h5_path)
            print(f"[intro] {len(self._clip_groups)} unique clips "
                  f"({sum(len(v) for v in self._clip_groups.values())} total frames)")

            limit = self.task_config.num_samples
            count = 0
            for clip_id in sorted(self._clip_groups.keys(),
                                  key=lambda s: int("".join(ch for ch in s
                                                            if ch.isdigit()) or "0")):
                yield {
                    "clip_id": clip_id,
                    "row_indices": self._clip_groups[clip_id],
                    "meta_row": self._meta_rows.get(
                        clip_id,
                        self._meta_rows.get(clip_id.rstrip("_"), {}),
                    ),
                }
                count += 1
                if limit is not None and count >= limit:
                    return

    def process_sample(self, raw: dict, idx: int) -> Optional[TaskSample]:
        clip_id = raw["clip_id"]
        row_idxs: List[int] = raw["row_indices"]
        meta_row: dict = raw.get("meta_row") or {}
        n_total = len(row_idxs)
        if n_total == 0:
            return None

        # Evenly sub-sample frames if the clip is long.
        max_frames = self.task_config.max_frames
        if n_total > max_frames:
            pick = np.linspace(0, n_total - 1, max_frames).astype(int)
            sel = [row_idxs[i] for i in pick]
        else:
            sel = list(row_idxs)

        first_clip: List[np.ndarray] = []  # raw frames (first_video)
        last_clip: List[np.ndarray] = []   # raw + overlay (last_video / GT)
        areas: List[int] = []

        prev_center: Optional[Tuple[int, int]] = None
        with h5py.File(self._h5_path, "r") as f:
            img_ds = f["image"]
            mask_ds = f["mask"]
            for f_i, ri in enumerate(sel):
                img2d = img_ds[ri]
                mask2d = mask_ds[ri]
                bgr = _normalize_frame(img2d)
                bgr, mask_r = _resize_keep(bgr, mask2d, self.task_config.max_side)

                area = int((mask_r > 0).sum())
                areas.append(area)

                # first_video = just raw + frame counter (no annotation).
                raw_disp = bgr.copy()
                _put_label(raw_disp, f"frame {f_i+1}/{len(sel)}",
                           (raw_disp.shape[1] - 150, 22))
                first_clip.append(raw_disp)

                # last_video / GT = overlay with bbox + size + motion arrow.
                anno, cur_center = _render_overlay(
                    bgr, mask_r, prev_center, f_i, len(sel), area
                )
                last_clip.append(anno)
                if cur_center is not None:
                    prev_center = cur_center

        # Find first and last non-zero area frames for the size-change summary.
        pos = [i for i, a in enumerate(areas) if a > 0]
        if pos:
            initial_area = areas[pos[0]]
            final_area = areas[pos[-1]]
        else:
            initial_area = 0
            final_area = 0
        pct = _percent_change(initial_area, final_area)
        label = _classify_delta(initial_area, final_area)

        # Append a summary text banner on the last frame of the annotated clip.
        if last_clip:
            summary = last_clip[-1].copy()
            line1 = f"initial area: {initial_area}px   final area: {final_area}px"
            line2 = f"change: {pct:+.1f}%   -> {label}"
            _put_label(summary, line1, (8, summary.shape[0] - 32),
                       color=(255, 255, 255))
            _put_label(summary, line2, (8, summary.shape[0] - 10),
                       color=(0, 255, 0) if label == "STABLE"
                       else (0, 255, 255) if label == "GROWING"
                       else (0, 140, 255))
            last_clip[-1] = summary

        # Build output dir.
        sid = f"{self.task_config.domain}_{idx:05d}"
        out = Path(self.task_config.output_dir) / f"{self.task_config.domain}_task" / sid
        out.mkdir(parents=True, exist_ok=True)

        # Key frames.
        cv2.imwrite(str(out / "first_frame.png"), first_clip[0])
        cv2.imwrite(str(out / "final_frame.png"), last_clip[-1])

        # Videos.
        _make_video(first_clip, out / "first_video.mp4", self.task_config.fps)
        _make_video(last_clip, out / "last_video.mp4", self.task_config.fps)
        # ground_truth.mp4 = same as last_video (fully-annotated temporal track).
        _make_video(last_clip, out / "ground_truth.mp4", self.task_config.fps)

        # Prompt.
        meta_clean = {k: v for k, v in meta_row.items()
                      if v not in (None, "", "0")}
        prompt = (
            f"{self.task_config.task_prompt}\n\n"
            f"Clip {clip_id}: {n_total} frames in source (showing {len(sel)}).\n"
            f"Clinical metadata (TI-RADS + histopath): "
            f"{json.dumps(meta_clean, ensure_ascii=False)}.\n"
        )
        (out / "prompt.txt").write_text(prompt)

        # metadata.json — carries ground-truth size-change answer for eval.
        meta_out = {
            "task": "Thyroid US cine nodule tracking + size-change classification",
            "dataset": "Stanford AIMI Thyroid Ultrasound Cine-clip",
            "case_id": clip_id,
            "annot_id": clip_id,
            "modality": "thyroid ultrasound cine",
            "classes": ["thyroid_nodule"],
            "colors": {"thyroid_nodule_bbox": "red",
                       "motion_arrow": "white"},
            "fps": self.task_config.fps,
            "frames_per_video": len(sel),
            "num_frames_total": int(n_total),
            "case_type": "A_real_ultrasound_cine_sequence",
            "areas_px": areas,
            "initial_area_px": int(initial_area),
            "final_area_px": int(final_area),
            "percent_change": round(float(pct), 2),
            "ground_truth_label": label,
            "clinical": meta_row,
        }
        (out / "metadata.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=2)
        )

        return TaskSample(
            task_id=sid,
            domain=self.task_config.domain,
            prompt=prompt.strip(),
            first_image=cv2.cvtColor(first_clip[0], cv2.COLOR_BGR2RGB),
            final_image=cv2.cvtColor(last_clip[-1], cv2.COLOR_BGR2RGB),
            first_video=str(out / "first_video.mp4"),
            last_video=str(out / "last_video.mp4"),
            ground_truth_video=str(out / "ground_truth.mp4"),
            metadata=meta_out,
        )

    # Override to be robust to per-clip failures.
    def run(self):
        samples: List[TaskSample] = []
        for idx, raw in enumerate(self.download()):
            try:
                s = self.process_sample(raw, idx)
                if s is not None:
                    samples.append(s)
            except Exception as e:
                print(f"  [warn] clip {raw.get('clip_id')} failed: {e}")
            if idx and idx % 10 == 0:
                print(f"  processed {idx} / ? clips ...")
        print(f"Done. Wrote {len(samples)} samples "
              f"to {self.task_config.output_dir}/{self.task_config.domain}_task/")
        return samples
