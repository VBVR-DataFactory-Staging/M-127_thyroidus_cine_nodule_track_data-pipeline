"""M-127 raw data downloader — Stanford Thyroid US cine-clip (HDF5).

Pulls dataset.hdf5 + metadata.csv from the public-read S3 mirror at
s3://med-vr-datasets/M-127/ThyroidUScine/thyroidultrasoundcineclip/
and exposes them under raw_dir/ for the pipeline to open.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from core.download import download_from_s3


class TaskDownloader:
    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(getattr(config, "raw_dir", "raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def ensure_raw(self):
        h5 = self.raw_dir / "dataset.hdf5"
        csv = self.raw_dir / "metadata.csv"
        if h5.exists() and csv.exists():
            print(f"Raw data already present at {self.raw_dir}, skipping sync.")
            return
        print(
            f"Raw data not found locally, syncing from "
            f"s3://{self.config.s3_bucket}/{self.config.s3_prefix} ..."
        )
        download_from_s3(
            bucket_name=self.config.s3_bucket,
            s3_prefix=self.config.s3_prefix,
            local_dir=self.raw_dir,
            region="us-east-2",
        )

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Yield a single dict pointing to the downloaded HDF5/CSV.

        The real per-clip iteration happens inside ``TaskPipeline.download``
        so we keep one HDF5 handle open across all samples.
        """
        self.ensure_raw()
        yield {
            "h5_path": str(self.raw_dir / "dataset.hdf5"),
            "csv_path": str(self.raw_dir / "metadata.csv"),
        }

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        """core.download.run_download() entry point."""
        yield from self.iter_samples(limit=limit)


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)
