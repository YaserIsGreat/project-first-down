"""Fine-tune YOLOv8n on the custom helmet dataset exported from Roboflow.

Expects a Roboflow "YOLOv8" export unzipped into ./dataset, which gives a
dataset/data.yaml alongside train/ valid/ test/ image folders.

imgsz defaults to 1280 rather than the usual 640: helmets occupy very few
pixels in an all-22 frame and downscaling to 640 loses the detail that
separates one helmet from the next in a tight line-of-scrimmage cluster.

Examples:
    python train.py
    python train.py --epochs 200 --batch 2
    python train.py --data path/to/data.yaml --device cpu
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "helmet"

EXPECTED_CLASSES = ["lions_helmet", "rams_helmet"]

# Measured on the RTX 3060 Laptop GPU (6 GB) at imgsz=1280 with yolov8n:
#   batch 4 -> 1.96 GB reserved
#   batch 8 -> 3.90 GB reserved
# 8 leaves ~2 GB of headroom for validation and mosaic spikes. 16 would not
# fit. Drop to 4 if you hit a CUDA out-of-memory error, or if you raise imgsz.
DEFAULT_BATCH_CUDA = 8
DEFAULT_BATCH_CPU = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n for two-class helmet detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="path to the Roboflow export's data.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="starting weights; downloads automatically if absent")
    parser.add_argument("--epochs", type=int, default=100,
                        help="maximum training epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="stop early after this many epochs without improvement")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="training image size")
    parser.add_argument("--batch", type=int, default=None,
                        help="batch size (default: 8 on CUDA, 2 on CPU)")
    parser.add_argument("--device", default=None,
                        help="'0' for the first GPU, 'cpu' to force CPU (default: auto)")
    parser.add_argument("--workers", type=int, default=4,
                        help="dataloader workers")
    parser.add_argument("--cache", choices=["ram", "disk", "none"], default="ram",
                        help="cache images between epochs; the dataset is small enough for RAM")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT,
                        help="parent folder for run outputs")
    parser.add_argument("--name", default="yolov8n_1280",
                        help="run name; results land in <project>/<name>")
    return parser.parse_args()


def resolve_device(requested):
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def check_dataset(data_path):
    """Fail early on a missing export, and warn if the classes look wrong."""
    if not data_path.is_file():
        sys.exit(
            f"No dataset found at {data_path}\n"
            "Export the annotated set from Roboflow in YOLOv8 format, unzip it "
            "into ./dataset, then rerun. Or point --data at the data.yaml."
        )

    with open(data_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    if list(names) != EXPECTED_CLASSES:
        print(
            f"Warning: expected classes {EXPECTED_CLASSES} but data.yaml has {list(names)}.\n"
            "         Training will still run, but check the Roboflow export is "
            "the team-based schema.",
            file=sys.stderr,
        )

    return names


def main():
    args = parse_args()

    names = check_dataset(args.data)
    device = resolve_device(args.device)

    if args.batch is None:
        args.batch = DEFAULT_BATCH_CPU if device == "cpu" else DEFAULT_BATCH_CUDA

    if device == "cpu":
        print("Warning: training on CPU. At imgsz=1280 this will take hours.", file=sys.stderr)

    print(f"data     {args.data}")
    print(f"classes  {list(names)}")
    print(f"model    {args.model}")
    print(f"device   {device}")
    print(f"imgsz    {args.imgsz}   batch {args.batch}")
    print(f"epochs   {args.epochs}  patience {args.patience}")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print(f"Run predict.py --weights \"{best}\" to use them.")


if __name__ == "__main__":
    main()
