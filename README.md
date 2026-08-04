# Project First Down

YOLOv8-based NFL player detection built as a portfolio project. The goal is to automatically detect all 22 players in all-22 footage and use their positions to analyse pre-snap formations and defensive coverages.

## What It Does

Takes raw NFL all-22 camera footage and detects every player on the field by tracking their helmet rather than their full body. Helmets are used instead of full body bounding boxes because at the line of scrimmage players are tightly packed and overlap — offensive linemen in particular bunch too tightly for body boxes to separate cleanly. Helmets stay visible above the pile and produce cleaner, more separable detections. Once all 22 helmet coordinates are reliably extracted per frame, the downstream task is formation and coverage classification through spatial reasoning on those coordinates.

## Tech Stack

Python 3.12, YOLOv8 (Ultralytics), PyTorch, ffmpeg, Roboflow, OpenCV

## Annotation Schema

Two classes, team-based:

| Class | Description |
|---|---|
| `lions_helmet` | Detroit Lions helmet |
| `rams_helmet` | Los Angeles Rams helmet |

Splitting by team rather than by offence/defence means the label stays correct no matter which side has the ball, so clips can be mixed freely in one dataset. Which team is on offence in a given play is metadata about the clip, not something the detector needs to learn. Position inference is handled geometrically from coordinates downstream rather than visually at the detection stage.

## What Has Been Done

Raw all-22 NFL footage (Lions vs Rams) was sourced and 8 individual plays were clipped using ffmpeg. A script extracts frames from those clips, producing 187 frames for annotation. A pretrained YOLOv8n COCO baseline was run across the clips to establish a detection benchmark before any fine-tuning.

## Currently In Progress

187 frames are annotated in Roboflow and ready to export. The first fine-tuning run has not happened yet — everything so far is the pretrained COCO baseline. Annotation is incremental: once an initial model is trained it will be used to pre-label new frames, and corrections will be fed back into the dataset for the next training round.

## Setup

```bash
git clone https://github.com/YaserIsGreat/project-first-down
cd project-first-down
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`ffmpeg` also needs to be on PATH for frame extraction.

To confirm the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

This should print a version ending in `+cu126` and `True`. If it prints `+cpu`, torch was resolved from PyPI instead of the PyTorch index — reinstall from `requirements.txt`.

## Usage

**Extract frames** from the clips at a fixed sample rate (2 fps by default):

```bash
python extract_Frames.py --fps 2
```

**Train** on a Roboflow YOLOv8 export unzipped into `./dataset`:

```bash
python train.py
```

Defaults to `yolov8n` at `imgsz=1280`, 100 epochs, patience 20. The image size matters: helmets occupy very few pixels in an all-22 frame, and downscaling to the usual 640 loses the detail that separates one helmet from the next in a tight cluster. Batch size defaults to 4, which is what fits alongside `imgsz=1280` on a 6 GB card — lower it to 2 on a CUDA out-of-memory error.

**Predict** over a clip using the trained weights:

```bash
python predict.py --source nflTrimmedAll22/ramsOffence03.mp4
```

Annotated video is written to `nflOutput/`.

## Repo Layout

Source footage, extracted frames, Roboflow exports, training runs and model weights are all gitignored — this repo holds code only. `yolov8n.pt` downloads automatically on first use.

## Next Steps

1. Export the annotated set from Roboflow and run the first fine-tune
2. Validate reliable 22-helmet detection per frame on held-out clips
3. Set up the model-assisted labelling loop — pre-label, correct, retrain
4. Extract per-frame helmet coordinates and separate by team
5. Add field-boundary filtering so sideline players are excluded at inference
6. Build the pre-snap formation classifier from coordinate geometry
7. Detect the snap event to trigger idle mode post-snap
8. Extend to defensive coverage and offensive route tracking (this is a whole other can of worms)

## Licence

MIT — see [LICENSE](LICENSE).
