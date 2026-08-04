"""Download a Roboflow dataset version straight into ./dataset, ready for train.py.

Needs a Roboflow API key, read from the ROBOFLOW_API_KEY environment variable
so the key never lands in the repo or in shell history:

    $env:ROBOFLOW_API_KEY = "your_key_here"      # PowerShell, this session only
    python download_dataset.py --workspace WS --project PROJ --version 1

Workspace, project and version come out of your Roboflow URL:

    app.roboflow.com/<workspace>/<project>/<version>

Also rewrites the paths in data.yaml. Roboflow emits paths like
"../train/images" that are relative to the zip rather than to wherever it was
unpacked, which ultralytics resolves one directory too high and reports as a
missing dataset. Fixing it here means it stays fixed across re-exports.
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = REPO_ROOT / "dataset"

API_BASE = "https://api.roboflow.com"
EXPORT_FORMAT = "yolov8"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Roboflow dataset version as a YOLOv8 export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    return parser.parse_args()


def require(value, name, flag):
    if not value:
        sys.exit(f"Missing {name}. Pass {flag} or set the matching environment variable.")
    return value


def request_export(workspace, project, version, api_key):
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

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(dest)

    size_mb = len(response.content) / 1024**2
    print(f"Unpacked {size_mb:.1f} MB into {dest}")


def fix_data_yaml(dest):
    """Repoint data.yaml at an absolute path so ultralytics resolves it correctly."""
    data_yaml = dest / "data.yaml"
    if not data_yaml.is_file():
        sys.exit(f"Export unpacked but no data.yaml found in {dest}")

    with open(data_yaml, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["path"] = dest.as_posix()
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (dest / folder / "images").is_dir():
            config[split] = f"{folder}/images"
        else:
            config.pop(split, None)

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return config


def count_images(dest, folder):
    images = dest / folder / "images"
    return len(list(images.iterdir())) if images.is_dir() else 0


def main():
    args = parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set. Find your key under Settings > API Keys\n"
            "in Roboflow, then in PowerShell:\n"
            '    $env:ROBOFLOW_API_KEY = "your_key_here"'
        )

    workspace = require(args.workspace, "workspace", "--workspace")
    project = require(args.project, "project", "--project")
    version = require(args.version, "version", "--version")

    if args.dest.exists() and any(args.dest.iterdir()) and not args.overwrite:
        sys.exit(f"{args.dest} already exists and is not empty. Pass --overwrite to replace it.")

    link = request_export(workspace, project, version, api_key)
    download_and_unpack(link, args.dest)
    config = fix_data_yaml(args.dest)

    print(f"\nclasses  {config.get('names')}")
    for folder in ("train", "valid", "test"):
        n = count_images(args.dest, folder)
        if n:
            print(f"{folder:8} {n} images")

    print(f"\ndata.yaml ready at {args.dest / 'data.yaml'}")
    print("Run train.py to start fine-tuning.")


if __name__ == "__main__":
    main()
