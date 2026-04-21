"""M-127: Thyroid US cine — track thyroid nodules across frames."""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np, h5py
from common import DATA_ROOT, write_task, COLORS, fit_square

PID="M-127"; TASK_NAME="thyroidus_cine_nodule_track"; FPS=5
PROMPT=("This is a thyroid ultrasound cine clip from the Stanford AIMI "
        "ThyroidUltrasoundCineClip dataset. Track and segment thyroid nodules "
        "across frames with red bounding boxes and outline.")

def process_clip(clip_key: str, images: np.ndarray, annot: np.ndarray, idx: int):
    # images: (T, H, W) uint8; annot: (T, H, W) binary mask OR (T, 4) bbox array
    n_frames=images.shape[0]
    step=max(1,n_frames//60); idxs=list(range(0,n_frames,step))[:60]
    ff,lf,gf,fl=[],[],[],[]
    for t in idxs:
        gray=images[t]
        if gray.ndim==2: rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        else: rgb=gray
        rgb=fit_square(rgb,512)
        ann=rgb.copy()
        has=False
        if annot is not None and annot.ndim==3:
            m=annot[t]
            if m.shape[:2]!=rgb.shape[:2]:
                m=cv2.resize(m.astype(np.uint8),(512,512),interpolation=cv2.INTER_NEAREST)
            if m.sum()>0:
                has=True
                cnts,_=cv2.findContours((m>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ann,cnts,-1,COLORS["red"],2)
                for c in cnts:
                    x,y,w,h=cv2.boundingRect(c)
                    cv2.rectangle(ann,(x,y),(x+w,y+h),COLORS["yellow"],2)
        ff.append(rgb); lf.append(ann); fl.append(has)
        if has: gf.append(ann)
    if not gf: gf=lf[:5]
    pick=next((i for i,f in enumerate(fl) if f),0)
    meta={"task":"Thyroid US cine nodule tracking","dataset":"Stanford Thyroid US Cine-clip",
          "case_id":clip_key,"modality":"thyroid ultrasound cine","classes":["thyroid_nodule"],
          "colors":{"thyroid_nodule":"red"},"fps":FPS,"frames_per_video":len(idxs),
          "case_type":"A_real_cine_sequence","num_frames_total":int(n_frames)}
    return write_task(PID,TASK_NAME,idx,ff[pick],lf[pick],ff,lf,gf,PROMPT,meta,FPS)

def main():
    h5_path=DATA_ROOT/"_extracted"/"M-127_ThyroidUSCine"/"thyroidultrasoundcineclip"/"dataset.hdf5"
    if not h5_path.exists():
        print(f"  h5 not found: {h5_path}"); return
    with h5py.File(str(h5_path),"r") as f:
        print(f"  top-level keys: {list(f.keys())[:10]}")
        # Find structure — try common layouts
        clips = []
        if "images" in f and "annotations" in f:
            # Single big datasets
            imgs=f["images"]; annots=f["annotations"]
            for i in range(min(len(imgs), 500)):
                clips.append((f"clip_{i:05d}", imgs[i], annots[i] if i<len(annots) else None))
        else:
            # Per-clip groups
            for key in f.keys():
                g=f[key]
                if isinstance(g, h5py.Group):
                    imgs=g.get("images") or g.get("frames") or g.get("image")
                    annots=g.get("annotations") or g.get("masks") or g.get("mask") or g.get("label")
                    if imgs is not None:
                        clips.append((key, imgs[()], annots[()] if annots is not None else None))
        print(f"  {len(clips)} thyroid clips")
        i=0
        for key, imgs, ann in clips:
            d=process_clip(key, imgs, ann, i)
            if d: i+=1

if __name__=="__main__": main()
