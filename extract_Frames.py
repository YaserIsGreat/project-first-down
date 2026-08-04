"""Extract frames from the trimmed all-22 clips for annotation in Roboflow.

Frames are named <clip>_<index>.<ext> so every image traces back to its source
clip, e.g. ramsOffence03_0007.png.

Examples:
    python extract_Frames.py
    python extract_Frames.py --fps 5
    python extract_Frames.py -i nflTrimmedAll22 -o frames --fps 4 --overwrite
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = REPO_ROOT / "nflTrimmedAll22"
DEFAULT_OUTPUT = REPO_ROOT / "frames"
DEFAULT_FPS = 2.0

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from video clips at a fixed sample rate using ffmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=Path, default=DEFAULT_INPUT,
        help="folder containing the source video clips",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help="folder to write extracted frames into",
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="frames to sample per second of video",
    )
    parser.add_argument(
        "--ext", choices=["png", "jpg"], default="png",
        help="output image format; png is lossless, jpg is smaller",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace frames from a previous extraction instead of skipping the clip",
    )
    return parser.parse_args()


def find_clips(input_dir):
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def existing_frames(output_dir, clip_name, ext):
    return sorted(output_dir.glob(f"{clip_name}_*.{ext}"))


def extract(clip, output_dir, fps, ext):
    """Run ffmpeg over one clip. Returns the number of frames written."""
    pattern = output_dir / f"{clip.stem}_%04d.{ext}"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(clip),
        "-vf", f"fps={fps}",
        str(pattern),
    ]

    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"  ffmpeg failed on {clip.name} (exit {result.returncode})", file=sys.stderr)
        return 0

    return len(existing_frames(output_dir, clip.stem, ext))


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found on PATH. Install it and try again.")

    if not args.input.is_dir():
        sys.exit(f"Input folder does not exist: {args.input}")

    clips = find_clips(args.input)
    if not clips:
        sys.exit(f"No video files found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Extracting at {args.fps} fps from {len(clips)} clip(s) into {args.output}")

    total = 0
    for clip in clips:
        stale = existing_frames(args.output, clip.stem, args.ext)
        if stale and not args.overwrite:
            print(f"  {clip.name}: {len(stale)} frames already present, skipping (use --overwrite)")
            continue
        for frame in stale:
            frame.unlink()

        written = extract(clip, args.output, args.fps, args.ext)
        print(f"  {clip.name}: {written} frames")
        total += written

    print(f"Done. {total} frames written to {args.output}")


if __name__ == "__main__":
    main()
