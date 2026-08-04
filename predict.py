"""Run trained helmet weights over a video (or a folder of videos) and save the
annotated result.

Examples:
    python predict.py --source nflTrimmedAll22/ramsOffence03.mp4
    python predict.py --source nflTrimmedAll22 --conf 0.4
    python predict.py --weights runs/helmet/yolov8n_1280/weights/best.pt --source clip.mp4
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "helmet" / "yolov8n_1280" / "weights" / "best.pt"
DEFAULT_SOURCE = REPO_ROOT / "nflTrimmedAll22"
DEFAULT_OUTPUT = REPO_ROOT / "nflOutput"

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect helmets in video using fine-tuned YOLOv8 weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-w", "--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="trained weights (.pt) to run")
    parser.add_argument("-s", "--source", type=Path, default=DEFAULT_SOURCE,
                        help="a video file, or a folder of video files")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                        help="parent folder for annotated output")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold; helmets cluster tightly, so keep this low")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="inference image size; should match what the model was trained at")
    parser.add_argument("--device", default=None,
                        help="'0' for the first GPU, 'cpu' to force CPU (default: auto)")
    parser.add_argument("--save-txt", action="store_true",
                        help="also write per-frame YOLO-format label files")
    return parser.parse_args()


def resolve_device(requested):
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def collect_sources(source):
    if source.is_file():
        return [source]
    if source.is_dir():
        clips = sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        if not clips:
            sys.exit(f"No video files found in {source}")
        return clips
    sys.exit(f"Source does not exist: {source}")


def run_clip(model, clip, args, device):
    """Predict over one clip, saving the annotated video. Returns detection counts."""
    counts = Counter()
    frames = 0

    # stream=True yields results frame by frame instead of holding the whole
    # video in memory; the annotated video is still written as it goes.
    results = model.predict(
        source=str(clip),
        stream=True,
        save=True,
        save_txt=args.save_txt,
        project=str(args.output),
        name=clip.stem,
        exist_ok=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
    )

    for result in results:
        frames += 1
        for class_id in result.boxes.cls.tolist():
            counts[result.names[int(class_id)]] += 1

    return frames, counts


def main():
    args = parse_args()

    if not args.weights.is_file():
        sys.exit(
            f"Weights not found: {args.weights}\n"
            "Train first with train.py, or pass --weights pointing at a .pt file."
        )

    clips = collect_sources(args.source)
    device = resolve_device(args.device)

    print(f"weights  {args.weights}")
    print(f"device   {device}")
    print(f"conf     {args.conf}   iou {args.iou}   imgsz {args.imgsz}")
    print(f"Running over {len(clips)} clip(s), writing to {args.output}\n")

    model = YOLO(str(args.weights))

    for clip in clips:
        frames, counts = run_clip(model, clip, args, device)
        detail = ", ".join(f"{name} {n}" for name, n in sorted(counts.items())) or "no detections"
        per_frame = sum(counts.values()) / frames if frames else 0
        print(f"  {clip.name}: {frames} frames, {detail} ({per_frame:.1f} per frame)")

    print(f"\nDone. Annotated video saved under {args.output}")


if __name__ == "__main__":
    main()
