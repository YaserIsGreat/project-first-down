"""Download a Roboflow dataset version into ./dataset, ready for train.py.

Two ways to fetch:

    # via the Roboflow API (repeatable; needs an API key)
    $env:ROBOFLOW_API_KEY = "your_key_here"
    python download_dataset.py --workspace WS --project PROJ --version 3

    # via a pre-signed export link from the Roboflow UI
    python download_dataset.py --link "https://app.roboflow.com/ds/XXXX?key=YYYY"

Workspace, project and version come from your Roboflow URL:
    app.roboflow.com/<workspace>/<project>/<version>

Two things are fixed after unpacking, both of which matter every re-export:

1. data.yaml paths. Roboflow writes "../train/images", relative to the zip
   rather than to wherever it was unpacked. Ultralytics resolves that one
   directory too high and reports it as a missing dataset.

2. The train/valid/test split, which is redistributed by clip. Frames are
   sampled at 2 fps, so consecutive frames within one play are near
   duplicates. Splitting them randomly puts near-identical images in both
   train and validation, which inflates mAP into meaninglessness. Holding out
   whole clips keeps the validation estimate honest.
"""

import argparse
import io
import os
import random
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = REPO_ROOT / "dataset"

API_BASE = "https://api.roboflow.com"
EXPORT_FORMAT = "yolov8"

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Roboflow YOLOv8 export and prepare it for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--link", default=None,
                        help="pre-signed Roboflow export link; skips the API entirely")
    parser.add_argument("--workspace", default=os.environ.get("ROBOFLOW_WORKSPACE"),
                        help="Roboflow workspace id (env: ROBOFLOW_WORKSPACE)")
    parser.add_argument("--project", default=os.environ.get("ROBOFLOW_PROJECT"),
                        help="Roboflow project id (env: ROBOFLOW_PROJECT)")
    parser.add_argument("--version", default=os.environ.get("ROBOFLOW_VERSION"),
                        help="dataset version number (env: ROBOFLOW_VERSION)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help="folder to unpack the dataset into")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing dataset folder")
    parser.add_argument("--split-mode", choices=["clip", "random", "keep"], default="clip",
                        help="clip: hold out whole clips. random: shuffle images "
                             "(leaks near-duplicate frames). keep: leave Roboflow's split")
    parser.add_argument("--ratios", default="70/20/10",
                        help="train/valid/test percentages for --split-mode clip or random")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for split assignment, so splits are reproducible")
    return parser.parse_args()


def parse_ratios(text):
    try:
        parts = [float(p) for p in text.split("/")]
    except ValueError:
        sys.exit(f"Could not parse --ratios {text!r}. Expected something like 70/20/10.")
    if len(parts) != 3 or sum(parts) <= 0:
        sys.exit(f"--ratios needs three positive numbers, got {text!r}")
    total = sum(parts)
    return [p / total for p in parts]


def request_export_link(workspace, project, version, api_key):
    """Ask Roboflow to prepare the export and hand back a download link."""
    url = f"{API_BASE}/{workspace}/{project}/{version}/{EXPORT_FORMAT}"
    response = requests.get(url, params={"api_key": api_key}, timeout=120)

    if response.status_code == 401:
        sys.exit("Roboflow rejected the API key (401). Check ROBOFLOW_API_KEY.")
    if response.status_code == 404:
        sys.exit(
            f"Roboflow returned 404 for {workspace}/{project}/{version}.\n"
            "Check the ids against your app.roboflow.com URL, and make sure the "
            "version has actually been generated."
        )
    response.raise_for_status()

    payload = response.json()
    link = payload.get("export", {}).get("link") or payload.get("link")
    if not link:
        sys.exit(f"No download link in Roboflow's response: {payload}")
    return link


def download_and_unpack(link, dest):
    print("Downloading export...")
    response = requests.get(link, timeout=600)
    response.raise_for_status()

    if "zip" not in response.headers.get("content-type", ""):
        sys.exit(
            "Roboflow did not return a zip. The link may have expired "
            f"(content-type: {response.headers.get('content-type')})."
        )

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(dest)

    print(f"Unpacked {len(response.content) / 1024**2:.1f} MB into {dest}")


def clip_of(filename):
    """Recover the source clip name from a Roboflow-mangled filename.

    lionsOffence01_0007_png.rf.<hash>.jpg -> lionsOffence01
    """
    stem = re.sub(r"\.rf\.[0-9a-fA-F]+\.\w+$", "", filename)
    stem = re.sub(r"_(png|jpg|jpeg|bmp|webp)$", "", stem, flags=re.IGNORECASE)
    match = re.match(r"^(.*)_\d+$", stem)
    return match.group(1) if match else stem


def gather_pairs(dest):
    """Collect every (image, label) pair across whatever splits Roboflow used."""
    pairs = []
    for split in SPLITS:
        images_dir = dest / split / "images"
        if not images_dir.is_dir():
            continue
        for image in sorted(images_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = dest / split / "labels" / f"{image.stem}.txt"
            pairs.append((image, label if label.is_file() else None))
    return pairs


def assign_by_clip(pairs, ratios, seed):
    """Assign whole clips to splits, greedily filling whichever is furthest behind."""
    by_clip = defaultdict(list)
    for image, label in pairs:
        by_clip[clip_of(image.name)].append((image, label))

    total = len(pairs)
    targets = dict(zip(SPLITS, (r * total for r in ratios)))
    counts = {s: 0 for s in SPLITS}
    assignment = {s: [] for s in SPLITS}

    # Largest clips first so the big ones land before quotas fill up.
    clips = sorted(by_clip, key=lambda c: (-len(by_clip[c]), c))
    rng = random.Random(seed)
    rng.shuffle(clips)
    clips.sort(key=lambda c: -len(by_clip[c]))

    for clip in clips:
        items = by_clip[clip]
        # deficit relative to target, normalised so small splits still get filled
        split = max(SPLITS, key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1))
        assignment[split].extend(items)
        counts[split] += len(items)

    return assignment, by_clip


def assign_randomly(pairs, ratios, seed):
    items = list(pairs)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = round(ratios[0] * n)
    n_valid = round(ratios[1] * n)
    return {
        "train": items[:n_train],
        "valid": items[n_train:n_train + n_valid],
        "test": items[n_train + n_valid:],
    }, None


def apply_assignment(dest, assignment):
    """Move image/label pairs into their assigned split folders."""
    staging = dest / "_staging"
    if staging.exists():
        shutil.rmtree(staging)

    # Stage first so a file never overwrites another mid-move.
    for split, items in assignment.items():
        for kind in ("images", "labels"):
            (staging / split / kind).mkdir(parents=True, exist_ok=True)
        for image, label in items:
            shutil.move(str(image), staging / split / "images" / image.name)
            if label is not None:
                shutil.move(str(label), staging / split / "labels" / label.name)

    for split in SPLITS:
        if (dest / split).exists():
            shutil.rmtree(dest / split)
        if (staging / split).exists():
            shutil.move(str(staging / split), dest / split)

    shutil.rmtree(staging, ignore_errors=True)


def write_data_yaml(dest):
    """Repoint data.yaml at absolute paths so ultralytics resolves it correctly."""
    data_yaml = dest / "data.yaml"
    if not data_yaml.is_file():
        sys.exit(f"Export unpacked but no data.yaml found in {dest}")

    with open(data_yaml, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["path"] = dest.as_posix()
    for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        images_dir = dest / folder / "images"
        if images_dir.is_dir() and any(images_dir.iterdir()):
            config[key] = f"{folder}/images"
        else:
            config.pop(key, None)

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return config


def summarise(dest, config, clips_by_split):
    print(f"\nclasses  {config.get('names')}")
    for split in SPLITS:
        images_dir = dest / split / "images"
        if not images_dir.is_dir():
            continue
        images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        boxes = sum(
            len([l for l in p.read_text().splitlines() if l.strip()])
            for p in (dest / split / "labels").glob("*.txt")
        )
        line = f"{split:6} {len(images):4} images  {boxes:5} boxes"
        if clips_by_split and clips_by_split.get(split):
            line += "  " + ", ".join(sorted(clips_by_split[split]))
        print(line)


def main():
    args = parse_args()
    ratios = parse_ratios(args.ratios)

    if args.dest.exists() and any(args.dest.iterdir()) and not args.overwrite:
        sys.exit(f"{args.dest} already exists and is not empty. Pass --overwrite to replace it.")
    if args.dest.exists() and args.overwrite:
        shutil.rmtree(args.dest)

    if args.link:
        link = args.link
    else:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            sys.exit(
                "ROBOFLOW_API_KEY is not set, and no --link was given. Either pass a\n"
                "pre-signed export link with --link, or find your key under\n"
                "Settings > API Keys in Roboflow and set it:\n"
                '    $env:ROBOFLOW_API_KEY = "your_key_here"'
            )
        for name, value, flag in (("workspace", args.workspace, "--workspace"),
                                  ("project", args.project, "--project"),
                                  ("version", args.version, "--version")):
            if not value:
                sys.exit(f"Missing {name}. Pass {flag} or set the matching environment variable.")
        link = request_export_link(args.workspace, args.project, args.version, api_key)

    download_and_unpack(link, args.dest)

    clips_by_split = None
    if args.split_mode != "keep":
        pairs = gather_pairs(args.dest)
        if not pairs:
            sys.exit(f"No images found under {args.dest}")

        if args.split_mode == "clip":
            assignment, by_clip = assign_by_clip(pairs, ratios, args.seed)
            clips_by_split = {
                split: {clip_of(img.name) for img, _ in items}
                for split, items in assignment.items()
            }
            print(f"\nRe-splitting {len(pairs)} images from {len(by_clip)} clips, by clip")
        else:
            assignment, _ = assign_randomly(pairs, ratios, args.seed)
            print(f"\nRe-splitting {len(pairs)} images at random")

        empty = [s for s, items in assignment.items() if not items]
        if empty:
            print(
                f"Warning: {', '.join(empty)} ended up empty. With few clips, clip-aware\n"
                "         splitting is coarse. Try --ratios or --split-mode random.",
                file=sys.stderr,
            )

        apply_assignment(args.dest, assignment)

    config = write_data_yaml(args.dest)
    summarise(args.dest, config, clips_by_split)

    print(f"\ndata.yaml ready at {args.dest / 'data.yaml'}")
    print("Run train.py to start fine-tuning.")


if __name__ == "__main__":
    main()
