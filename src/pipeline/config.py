"""Pipeline configuration for M-127 Thyroid US cine — nodule tracking + size change.

Distinct from:
- M-67 (tn3k):  static thyroid nodule segmentation on single frames.
- M-84 (ddti):  thyroid nodule classification / TI-RADS (single image).
- M-124:        thyroid cine LESION LOCALIZATION (per-frame bbox, static task).

M-127 emphasizes TEMPORAL TRACKING of the nodule across the full cine-clip
and classifies the nodule size evolution from first to last frame as one of
{stable, growing, shrinking}. Overlay adds per-frame bbox, size text and
frame-to-frame motion arrows that M-124 lacks.
"""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    domain: str = Field(default="thyroidus_cine_nodule_track")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw Stanford thyroid-cine data",
    )
    s3_prefix: str = Field(
        default="M-127/ThyroidUScine/thyroidultrasoundcineclip/",
        description="S3 key prefix for the dataset raw data (dataset.hdf5 + metadata.csv)",
    )

    fps: int = Field(default=6, description="Output video FPS")
    max_frames: int = Field(
        default=60,
        description="Cap frames per clip (evenly sub-sample if longer)",
    )
    max_side: int = Field(
        default=512,
        description="Max side for output frames; preserves aspect ratio",
    )

    raw_dir: Path = Field(default=Path("raw"))

    task_prompt: str = Field(
        default=(
            "Track the thyroid nodule across all frames of this ultrasound cine-clip. "
            "The nodule is annotated on each frame by a red bounding box; its cross-sectional "
            "area (pixel count) is printed in the top-left corner, and white arrows across "
            "frames show the nodule's motion from frame to frame. Calculate the nodule's "
            "size change from the FIRST to the LAST frame of the clip and CLASSIFY the "
            "nodule as one of: STABLE (|delta| <= 10% of initial area), GROWING "
            "(delta > +10%), or SHRINKING (delta < -10%). Report: (1) initial vs final "
            "area in pixels, (2) percent change, (3) stable/growing/shrinking label, "
            "(4) whether the nodule stays within the thyroid (no out-of-plane motion)."
        ),
        description="The task instruction shown to the reasoning model.",
    )
