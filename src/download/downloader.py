"""Downloader for M-127 TN3K (paired image + mask).

Reads images and masks from local ``raw_dir/{trainval,test}-image`` and
``raw_dir/{trainval,test}-mask``. If the local raw dir is empty, falls back
to syncing from the S3 mirror at
``s3://med-vr-datasets/M-127/tn3k/Thyroid Dataset/tn3k/`` (note the SPACE
in the path). Tries ``aws s3 sync`` (CLI) first since the EC2 already has
AWS credentials via the instance role; falls back to public-HTTP S3 GET.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import quote

from core.download import download_from_s3


class TaskDownloader:
    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _has_any_pair(self) -> bool:
        for split in ("trainval", "test"):
            img_dir = self.raw_dir / f"{split}-image"
            mask_dir = self.raw_dir / f"{split}-mask"
            if not img_dir.exists() or not mask_dir.exists():
                continue
            for img in img_dir.glob("*.jpg"):
                if (mask_dir / img.name).exists():
                    return True
        return False

    def _sync_via_aws_cli(self, bucket: str, prefix: str) -> bool:
        """Use the awscli (installed by bootstrap) — handles spaces & creds cleanly."""
        if shutil.which("aws") is None:
            return False
        s3_uri = f"s3://{bucket}/{prefix}"
        print(f"[download] aws s3 sync '{s3_uri}' '{self.raw_dir}/' --no-progress")
        try:
            res = subprocess.run(
                [
                    "aws", "s3", "sync", s3_uri, f"{self.raw_dir}/",
                    "--no-progress", "--region", "us-east-2",
                    "--exclude", "*", "--include", "*-image/*",
                    "--include", "*-mask/*",
                ],
                check=False,
            )
            if res.returncode == 0:
                return True
            print(f"[download] aws s3 sync exit={res.returncode}; falling back to HTTP")
            return False
        except Exception as e:
            print(f"[download] aws s3 sync raised {e!r}; falling back")
            return False

    def ensure_raw(self):
        if self._has_any_pair():
            print(f"[download] raw data already present at {self.raw_dir}")
            return
        bucket = self.config.s3_bucket
        prefix = self.config.s3_prefix

        if self._sync_via_aws_cli(bucket, prefix):
            return

        # HTTP fallback. URL-encode for the LIST query so spaces survive.
        encoded_prefix = quote(prefix, safe="/")
        print(
            f"[download] HTTP fallback: GET https://{bucket}.s3.us-east-2.amazonaws.com/"
            f"?prefix={encoded_prefix}"
        )
        download_from_s3(
            bucket_name=bucket,
            s3_prefix=encoded_prefix,
            local_dir=self.raw_dir,
            region="us-east-2",
        )

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        self.ensure_raw()
        count = 0
        # Order: trainval then test
        for split in ("trainval", "test"):
            img_dir = self.raw_dir / f"{split}-image"
            mask_dir = self.raw_dir / f"{split}-mask"
            if not img_dir.exists() or not mask_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    continue
                yield {
                    "image_id": f"{split}_{img_path.stem}",
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "split": split,
                }
                count += 1
                if limit is not None and count >= limit:
                    return

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        yield from self.iter_samples(limit=limit)


def create_downloader(config):
    return TaskDownloader(config)
