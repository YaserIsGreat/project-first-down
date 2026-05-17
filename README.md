# Project First Down

YOLOv8-based NFL player detection system built as a university dissertation. The goal is to automatically detect all 22 players in all-22 footage and use their positions to analyse pre-snap formations and defensive coverages.

## What It Does

Takes raw NFL all-22 camera footage and detects every player on the field by tracking their helmet rather than their full body. Helmets are used instead of full body bounding boxes because at the line of scrimmage players are tightly packed and overlap, helmets stay visible above the pile and produce cleaner, more separable detections. Once all 22 helmet coordinates are reliably extracted per frame, the downstream task is formation and coverage classification through spatial reasoning on those coordinates.

## Tech Stack

Python 3.13, YOLOv8 (Ultralytics), PyTorch, ffmpeg, Roboflow, OpenCV

## What Has Been Done

Raw all-22 NFL footage (Lions vs Rams) was sourced and individual plays were clipped using ffmpeg. A Python script was written to extract frames from the clips, producing 187 annotated frames across 8 game clips. The annotation schema was decided: two classes, `offence_helmet` and `defence_helmet`, rather than individual positions. Position inference will be handled geometrically from coordinates later rather than visually at the detection stage. A pretrained YOLOv8n baseline was run on the raw footage to establish a detection benchmark before fine-tuning.

## Currently In Progress

Manual annotation of the helmet dataset in Roboflow. The target is 500+ annotated helmets before the first fine-tuning run. Annotation is incremental, once an initial model is trained it will be used to pre-label new frames, and corrections will be added back into the dataset for the next training round.

## Next Steps

1. Complete annotation to training threshold
2. Fine-tune YOLOv8 on the custom helmet dataset
3. Validate reliable 22-helmet detection per frame
4. Extract per-frame helmet coordinates and separate by team
5. Build pre-snap formation classifier using predetermined rules (hopefully)
6. Detect snap event to trigger idle mode post-snap
7. Extend to defensive coverage and offensive route tracking (this is a whole other can of worms)

## Installation

```bash
git clone https://github.com/yourusername/project-first-down
cd project-first-down
pip install ultralytics opencv-python
```

## Running Inference

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # replace with fine-tuned weights once trained

model.predict(
    source="path/to/clip.mp4",
    save=True,
    classes=[0],
    conf=0.3
)
```
