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

Raw all-22 NFL footage (Lions vs Rams) was sourced and 8 individual plays were clipped using ffmpeg. A script extracts frames from those clips at 2 fps, producing 187 frames. A pretrained YOLOv8n COCO baseline was run across the clips to establish a detection benchmark before any fine-tuning.

## Dataset

112 of the 187 extracted frames are annotated, carrying **2,207 helmet boxes** — 1,107 `lions_helmet` and 1,100 `rams_helmet`, an even split. Annotation covers 6 of the 8 clips:

| Clip | Extracted | Annotated |
|---|---|---|
| lionsOffence01 | 16 | 16 |
| lionsOffence02 | 35 | 35 |
| ramsOffence01 | 18 | 18 |
| ramsOffence02 | 24 | 24 |
| ramsOffence03 | 26 | 18 |
| ramsOffence04 | 24 | 0 |
| ramsOffence05 | 19 | 0 |
| ramsOffence06 | 25 | 1 |

**Splits are assigned by clip, not by image.** Frames are sampled at 2 fps, so consecutive frames within a play are near duplicates. Splitting them randomly would put near-identical images on both sides of the train/validation boundary and inflate mAP into something meaningless. `download_dataset.py` holds out whole clips instead, so no play appears in more than one split. Both teams are visible in every frame, so class balance survives the coarser split.

Roboflow's own export ships a 108/2/2 split, which is why the download step rewrites it.

## Results

First fine-tune of `yolov8n` at `imgsz=1280`, batch 8, on an RTX 3060 Laptop GPU (6 GB). Early-stopped at epoch 52 of 100, best at epoch 32, about 7 minutes wall clock.

Evaluated on the held-out test clip (`ramsOffence01`, 18 images, 381 boxes) — a play the model never saw in training, and which took no part in early stopping or weight selection:

| Metric | Test | Validation |
|---|---|---|
| mAP@50 | **0.882** | 0.792 |
| mAP@50-95 | **0.395** | 0.305 |
| Precision | 0.885 | 0.817 |
| Recall | 0.829 | 0.732 |

Per class on test: `lions_helmet` mAP@50 0.889, `rams_helmet` mAP@50 0.875 — no meaningful bias toward either team.

Test scores above validation, which is the opposite of the usual pattern. With clip-level splits and only one clip per split, that difference is per-clip difficulty rather than anything meaningful about generalisation. Treat both numbers as directional until more clips are annotated.

On a completely unseen clip (`ramsOffence04`, never annotated), the model averages **16 helmets per frame** against a nominal 22 on the field. Sideline players are correctly ignored, an effect of deleting sideline false positives during annotation.

## Currently In Progress

Expanding the dataset. `ramsOffence04`, `ramsOffence05` and most of `ramsOffence06` are still unannotated — 67 frames. Those are the next targets for the model-assisted labelling loop, now that there is a model worth pre-labelling with.

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

**Download the dataset** from Roboflow into `./dataset`, re-split by clip and with `data.yaml` paths corrected:

```bash
python download_dataset.py --workspace WORKSPACE --project PROJECT --version 3
```

Needs `ROBOFLOW_API_KEY` in the environment. A pre-signed export link from the Roboflow UI works too, via `--link`, with no key required.

Two fixes are applied on every download, both of which otherwise have to be redone by hand each re-export. Roboflow writes `data.yaml` paths like `../train/images`, relative to the zip rather than to wherever it was unpacked — ultralytics resolves that one directory too high and reports it as a missing dataset. And the split is rebuilt by clip, as described above.

**Train** on the downloaded dataset:

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

1. Annotate the remaining 67 frames using the trained model to pre-label
2. Close the recall gap from 16 detected helmets per frame toward 22
3. Retrain on the expanded set and compare against the numbers above
4. Extract per-frame helmet coordinates and separate by team
5. Add field-boundary filtering so sideline players are excluded at inference
6. Build the pre-snap formation classifier from coordinate geometry
7. Detect the snap event to trigger idle mode post-snap
8. Extend to defensive coverage and offensive route tracking (this is a whole other can of worms)

## Licence

MIT — see [LICENSE](LICENSE).
