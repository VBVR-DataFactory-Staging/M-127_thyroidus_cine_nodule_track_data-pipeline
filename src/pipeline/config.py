"""Pipeline configuration for M-127 — TN3K static thyroid ultrasound nodule segmentation.

Pivoted from the originally-scoped 'thyroidus_cine_nodule_track' (cine data
unavailable on S3) to TN3K static segmentation, which mirrors M-067's task family
on a different sampling of the same TN3K image pool.
"""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    domain: str = Field(default="tn3k_thyroid_nodule_segmentation_m127")
    s3_bucket: str = Field(default="med-vr-datasets")
    # NOTE: the raw TN3K layout on S3 sits inside a folder name with a SPACE.
    s3_prefix: str = Field(default="M-127/tn3k/Thyroid Dataset/tn3k/")
    fps: int = Field(default=12)
    raw_dir: Path = Field(default=Path("raw"))
    task_prompt: str = Field(
        default=(
            "This thyroid ultrasound image. Segment the thyroid nodule with a "
            "red binary mask on every frame. The nodule is a focal lesion within "
            "the thyroid parenchyma."
        ),
        description="Task instruction.",
    )
