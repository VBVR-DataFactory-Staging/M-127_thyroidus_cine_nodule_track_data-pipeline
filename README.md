# M-127_thyroidus_cine_nodule_track (TN3K static segmentation)

This pipeline was originally scoped as a thyroid US **cine** nodule-tracking
task. The cine source data was unavailable, so M-127 was **pivoted to TN3K
static thyroid ultrasound nodule segmentation** — same task family as M-067,
on a different sampling of the TN3K image pool.

The repo name `M-127_thyroidus_cine_nodule_track_data-pipeline` is preserved
for slot stability on `vm-dataset.com`.

Task prompt:
```
This thyroid ultrasound image. Segment the thyroid nodule with a red binary
mask on every frame. The nodule is a focal lesion within the thyroid parenchyma.
```

Raw data: `s3://med-vr-datasets/M-127/tn3k/Thyroid Dataset/tn3k/`
Output: `s3://vbvr-final-data/questions/M-127_thyroidus_cine_nodule_track_data-pipeline/`
