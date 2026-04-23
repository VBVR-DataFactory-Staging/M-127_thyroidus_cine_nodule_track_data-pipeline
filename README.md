# M-127 — Thyroid Ultrasound Cine-clip Nodule Tracking

Track a thyroid nodule across all frames of a real-time ultrasound cine-clip
(Stanford AIMI ThyroidUltrasoundCineClip dataset) and classify its size
evolution as **stable / growing / shrinking**.

Part of the Med-VR data-pipeline suite for the VBVR (Very Big Video Reasoning)
benchmark.

## Task

**Prompt shown to the model**:

> Track the thyroid nodule across all frames of this ultrasound cine-clip. The
> nodule is annotated on each frame by a red bounding box; its cross-sectional
> area (pixel count) is printed in the top-left corner, and white arrows across
> frames show the nodule's motion from frame to frame. Calculate the nodule's
> size change from the FIRST to the LAST frame of the clip and CLASSIFY the
> nodule as one of: STABLE (|delta| ≤ 10% of initial area), GROWING (delta >
> +10%), or SHRINKING (delta < -10%). Report: (1) initial vs final area in
> pixels, (2) percent change, (3) stable/growing/shrinking label, (4) whether
> the nodule stays within the thyroid (no out-of-plane motion).

### Overlay spec (distinct from M-124 static localization)

| Element | Color | Purpose |
|---|---|---|
| Nodule bounding box (per frame) | red | Track the nodule |
| Motion arrow (frame-to-frame) | white | Show temporal motion |
| Size text (`area: <px>`, top-left) | red/yellow | Per-frame measurement |
| Frame counter (top-right) | white | Progress indicator |
| Final-frame summary (bottom) | green/yellow/orange | Size-change verdict |

## S3 Raw Data

```
s3://med-vr-datasets/M-127/ThyroidUScine/thyroidultrasoundcineclip/
├── dataset.hdf5     (~34 GB, N frames flat arrays: image + mask + annot_id + frame_num)
└── metadata.csv     (192 clips, TI-RADS scores + histopath_diagnosis)
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate all 192 clips (downloads raw from S3 on first run)
python examples/generate.py

# Generate first 5 for a smoke test
python examples/generate.py --num-samples 5

# Custom output directory
python examples/generate.py --output data/my_output
```

## Output Layout

```
data/questions/thyroidus_cine_nodule_track_task/
├── thyroidus_cine_nodule_track_00000/
│   ├── first_frame.png
│   ├── final_frame.png
│   ├── first_video.mp4      ← raw cine clip (no overlay)
│   ├── last_video.mp4       ← overlay: bbox + size + motion arrow
│   ├── ground_truth.mp4     ← full annotated track + final summary
│   ├── prompt.txt
│   └── metadata.json        ← incl. ground-truth STABLE/GROWING/SHRINKING label
├── thyroidus_cine_nodule_track_00001/
└── ...
```

`metadata.json` carries the ground-truth answer:

```json
{
  "initial_area_px": 1234,
  "final_area_px":   1421,
  "percent_change":  15.15,
  "ground_truth_label": "GROWING",
  "areas_px":       [1200, 1230, ...],
  "clinical":       { "ti-rads_level": "3", ... }
}
```

## Distinctions from sibling thyroid pipelines

| Repo | Task | Modality | Temporal? |
|---|---|---|---|
| M-67  | tn3k thyroid nodule segmentation | still image | no |
| M-84  | ddti TI-RADS classification | still image | no |
| M-124 | Thyroid cine lesion localization | cine | per-frame bbox (static) |
| **M-127** | **Nodule tracking + size-change** | **cine** | **yes (temporal)** |

## Configuration

`src/pipeline/config.py` (`TaskConfig`):

| Field | Default | Description |
|---|---|---|
| `domain` | `thyroidus_cine_nodule_track` | Output directory name |
| `s3_bucket` | `med-vr-datasets` | Raw bucket |
| `s3_prefix` | `M-127/ThyroidUScine/thyroidultrasoundcineclip/` | Raw prefix |
| `fps` | `6` | Output video FPS |
| `max_frames` | `60` | Cap frames per clip (sub-sample if longer) |
| `max_side` | `512` | Max side of output frames |

## Repository Structure

```
M-127_thyroidus_cine_nodule_track_data-pipeline/
├── core/                ← shared pipeline framework (verbatim)
├── eval/                ← shared evaluation utilities
├── src/
│   ├── download/
│   │   └── downloader.py   ← S3 raw-data downloader (HDF5 + CSV)
│   └── pipeline/
│       ├── config.py        ← TaskConfig
│       └── pipeline.py      ← TaskPipeline (tracking + size-change logic)
├── examples/
│   └── generate.py
├── requirements.txt
├── README.md
└── LICENSE
```
