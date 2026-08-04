"""Rule-based pre-snap analysis drawn over a clip.

Turns tracked helmet coordinates into a read of the play:

  * which team has the ball, from which side holds still before the snap
  * where the line of scrimmage is
  * who is on the line, who is wide, who is in the backfield
  * pre-snap motion, with the path the man in motion took
  * post-snap routes for the wide players

Everything here is geometry over helmet coordinates. There is no second model.

    python analyze_play.py --source nflTrimmedAll22/ramsOffence05.mp4
    python analyze_play.py --source nflTrimmedAll22/ramsOffence05.mp4 --slow 0.6

A note on what is reliable. Team identification and motion detection lean on
large, robust differences and hold up well. Route lines depend on a single
track surviving the pile, so they fragment when a player is occluded. Treat
the routes as illustrative rather than complete.
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "helmet" / "yolov8n_1280_v3" / "weights" / "best.pt"
DEFAULT_TRACKER = REPO_ROOT / "helmet_tracker.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "analysis"

# BGR
C_OFFENSE = (80, 220, 90)
C_DEFENSE = (60, 140, 255)
C_LOS = (0, 235, 235)
C_MOTION = (235, 80, 235)
C_ROUTE = (255, 200, 60)
C_LINE = (255, 255, 255)
C_PANEL = (18, 18, 18)


def parse_args():
    p = argparse.ArgumentParser(
        description="Rule-based pre-snap formation and motion analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-s", "--source", type=Path, required=True, help="video clip to analyse")
    p.add_argument("-w", "--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None)
    p.add_argument("--slow", type=float, default=0.7,
                   help="playback speed multiplier; 0.7 is a touch slower than real time")
    p.add_argument("--route-len", type=int, default=45,
                   help="frames of route trail to keep drawn behind each wide player")
    return p.parse_args()


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------

def track_clip(args, device):
    """Return per-frame lists of detections with stable track ids."""
    model = YOLO(str(args.weights))
    frames = []
    for r in model.track(
        source=str(args.source), stream=True, persist=True,
        tracker=str(args.tracker), imgsz=args.imgsz, conf=args.conf,
        device=device, verbose=False, save=False,
    ):
        rec = []
        b = r.boxes
        if b is not None and b.id is not None:
            for (cx, cy, w, h), tid, cls, cf in zip(
                b.xywh.tolist(), b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist()
            ):
                rec.append({"id": tid, "cls": cls, "x": cx, "y": cy,
                            "w": w, "h": h, "conf": cf})
        frames.append(dedupe(rec))
    return frames, model.names


def dedupe(dets):
    """Drop near-coincident boxes on the same helmet.

    The tracker occasionally holds two ids on one player, which double-counts
    him in the formation read. Anything closer than roughly half a helmet
    width is the same head.
    """
    keep = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if any((d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2
               < (0.6 * max(d["w"], k["w"])) ** 2 for k in keep):
            continue
        keep.append(d)
    return keep


# --------------------------------------------------------------------------
# camera stabilisation
# --------------------------------------------------------------------------

def stabilised_steps(frames):
    """Per-frame {id: (dx, dy)} with global camera motion removed.

    An all-22 camera pans and zooms constantly. Without removing that, every
    player looks like they are moving at once and both snap detection and
    motion detection fall apart. The median displacement across all tracked
    helmets is a decent estimate of camera motion, since most players are
    stationary or moving incoherently; subtracting it leaves real motion.
    """
    steps = []
    prev = {}
    for f in frames:
        cur = {d["id"]: (d["x"], d["y"]) for d in f}
        shared = [i for i in cur if i in prev]
        raw = {i: (cur[i][0] - prev[i][0], cur[i][1] - prev[i][1]) for i in shared}
        if len(raw) >= 4:
            mdx = statistics.median(v[0] for v in raw.values())
            mdy = statistics.median(v[1] for v in raw.values())
        else:
            mdx = mdy = 0.0
        steps.append({i: (v[0] - mdx, v[1] - mdy) for i, v in raw.items()})
        prev = cur
    return steps


def smooth(series, k=6):
    n = len(series)
    return [statistics.fmean(series[max(0, i - k):i + k + 1]) for i in range(n)]


def energy_curve(steps):
    out = []
    for s in steps:
        vals = [(dx * dx + dy * dy) ** 0.5 for dx, dy in s.values()]
        out.append(statistics.fmean(vals) if vals else 0.0)
    return smooth(out)


def detect_snap(energy):
    """Find the moment a settled offence bursts into motion.

    Looking for the first jump above a baseline is not enough: a clip can open
    with the camera still finding the play, and a late pan can out-rank the
    snap itself. What identifies a snap is the *transition* — a genuinely
    quiet stretch immediately followed by sustained movement — so score every
    candidate on that contrast and require the run-up to be below the clip's
    median motion.
    """
    n = len(energy)
    lo, hi = max(15, n // 12), n - 15
    if hi <= lo:
        return n // 3
    med = statistics.median(energy)

    best_score, best_i = None, None
    for i in range(lo, hi):
        pre = statistics.fmean(energy[max(0, i - 25):i])
        post = statistics.fmean(energy[i:i + 20])
        if pre > med:
            continue  # the run-up was not quiet, so this is not a snap
        score = post - pre
        if best_score is None or score > best_score:
            best_score, best_i = score, i
    return best_i if best_i is not None else n // 3


# --------------------------------------------------------------------------
# play geometry
# --------------------------------------------------------------------------

def net_displacement(frames, steps, lo, hi):
    """Stabilised net movement per track over [lo, hi)."""
    acc = defaultdict(lambda: [0.0, 0.0])
    seen = defaultdict(int)
    for i in range(lo, min(hi, len(steps))):
        for tid, (dx, dy) in steps[i].items():
            acc[tid][0] += dx
            acc[tid][1] += dy
            seen[tid] += 1
    return {t: (acc[t][0] ** 2 + acc[t][1] ** 2) ** 0.5
            for t in acc if seen[t] >= max(4, (hi - lo) * 0.25)}


def team_of(frames, lo, hi):
    """Majority class per track over a window."""
    votes = defaultdict(lambda: defaultdict(int))
    for f in frames[lo:hi]:
        for d in f:
            votes[d["id"]][d["cls"]] += 1
    return {t: max(v, key=v.get) for t, v in votes.items()}


def classify_offense(disp, teams):
    """The offence is the side that holds still before the snap.

    Both teams put five men on the line, so counting the front does not
    separate them. But the offence has to be set and the defence does not,
    which shows up clearly as a lower median pre-snap displacement.
    Excludes the top mover on each side so a man in motion cannot flip it.
    """
    by_team = defaultdict(list)
    for tid, d in disp.items():
        if tid in teams:
            by_team[teams[tid]].append(d)

    stats = {}
    for cls, vals in by_team.items():
        if len(vals) < 3:
            continue
        trimmed = sorted(vals)[:-1]  # drop the biggest mover (possible motion)
        stats[cls] = statistics.median(trimmed)
    if len(stats) < 2:
        return None, None, stats
    order = sorted(stats, key=lambda c: stats[c])
    return order[0], order[1], stats


def frame_positions(frames, idx, teams):
    return [(d["id"], teams.get(d["id"], d["cls"]), np.array([d["x"], d["y"]]))
            for d in frames[idx]]


def play_axis(pos, off_cls, def_cls):
    """Unit vector pointing from the defence toward the offence."""
    o = np.array([p for _, c, p in pos if c == off_cls])
    d = np.array([p for _, c, p in pos if c == def_cls])
    if len(o) < 2 or len(d) < 2:
        return None, None
    axis = o.mean(0) - d.mean(0)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return None, None
    axis = axis / n
    return axis, np.array([-axis[1], axis[0]])


def analyse_formation(pos, axis, lateral, off_cls, def_cls, helmet_w):
    """Split the offence into linemen, wide players and backfield."""
    off = [(t, p) for t, c, p in pos if c == off_cls]
    dfn = [(t, p) for t, c, p in pos if c == def_cls]
    if len(off) < 5 or not dfn:
        return None

    s_off = {t: float(np.dot(p, axis)) for t, p in off}
    s_def = {t: float(np.dot(p, axis)) for t, p in dfn}
    los = (max(s_def.values()) + min(s_off.values())) / 2.0

    # five offensive players nearest the line of scrimmage
    line_ids = [t for t, _ in sorted(s_off.items(), key=lambda kv: kv[1])[:5]]
    lat = {t: float(np.dot(p, lateral)) for t, p in off}
    line_lat = [lat[t] for t in line_ids]
    lo, hi = min(line_lat), max(line_lat)
    span = max(hi - lo, 1.0)
    # Scale the "is he split out" threshold to helmet size rather than to the
    # line's own span, so a noisy line width cannot swallow a real receiver.
    margin = max(span * 0.22, 2.2 * helmet_w)

    wide_left, wide_right, backfield = [], [], []
    for t, p in off:
        if t in line_ids:
            continue
        if lat[t] < lo - margin:
            wide_left.append(t)
        elif lat[t] > hi + margin:
            wide_right.append(t)
        else:
            backfield.append(t)

    return {
        "los": los, "line_ids": line_ids, "s_off": s_off,
        "wide_left": wide_left, "wide_right": wide_right, "backfield": backfield,
        "wide": wide_left + wide_right,
    }


def detect_motion(frames, steps, teams, off_cls, snap):
    """A single offensive player travelling far while the rest are set."""
    lo = max(0, snap - 90)
    disp = net_displacement(frames, steps, lo, snap)
    off = {t: d for t, d in disp.items() if teams.get(t) == off_cls}
    if len(off) < 4:
        return None
    med = statistics.median(off.values())
    tid, best = max(off.items(), key=lambda kv: kv[1])
    if best < max(med * 4.0, 45.0):
        return None

    # when did it start? first frame where this track begins moving steadily
    start = snap
    run = 0
    for i in range(lo, snap):
        dx, dy = steps[i].get(tid, (0.0, 0.0))
        if (dx * dx + dy * dy) ** 0.5 > 0.8:
            run += 1
            if run >= 4:
                start = i - run
                break
        else:
            run = 0
    return {"id": tid, "start": max(lo, start), "dist": best, "median": med}


def path_of(frames, tid, lo, hi):
    pts = []
    for i in range(max(0, lo), min(hi, len(frames))):
        for d in frames[i]:
            if d["id"] == tid:
                pts.append((i, int(d["x"]), int(d["y"])))
                break
    return pts


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def panel(img, lines, org=(24, 24), width=430):
    h = 20 + 30 * len(lines)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + h), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
    cv2.rectangle(img, (x, y), (x + width, y + h), (90, 90, 90), 1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (x + 14, y + 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)


def draw_los(img, axis, los, shape):
    h, w = shape[:2]
    lateral = np.array([-axis[1], axis[0]])
    centre = axis * los
    p1 = centre + lateral * 4000
    p2 = centre - lateral * 4000
    cv2.line(img, tuple(np.int32(p1)), tuple(np.int32(p2)), C_LOS, 2, cv2.LINE_AA)


def label(img, text, pt, colour, scale=0.5):
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def render(args, frames, names, info, device):
    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.source.stem}_analysis.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps * args.slow, (w, h))

    snap = info["snap"]
    off_cls, def_cls = info["off_cls"], info["def_cls"]
    form = info["formation"]
    motion = info["motion"]
    axis = info["axis"]
    off_name = names[off_cls].replace("_helmet", "").upper()
    def_name = names[def_cls].replace("_helmet", "").upper()

    # The motion is a pre-snap event, so the trail stops at the snap. Letting
    # it run on would draw the player's actual carry and read as motion.
    motion_path = path_of(frames, motion["id"], motion["start"], snap) if motion else []
    route_hist = defaultdict(list)

    idx = 0
    while True:
        ok, img = cap.read()
        if not ok or idx >= len(frames):
            break
        dets = frames[idx]
        pre = idx < snap

        if axis is not None and form is not None:
            draw_los(img, axis, form["los"], img.shape)

        for d in dets:
            tid, cls = d["id"], d["cls"]
            team = info["teams"].get(tid, cls)
            colour = C_OFFENSE if team == off_cls else C_DEFENSE
            x, y, bw, bh = d["x"], d["y"], d["w"], d["h"]
            p1 = (int(x - bw / 2), int(y - bh / 2))
            p2 = (int(x + bw / 2), int(y + bh / 2))
            cv2.rectangle(img, p1, p2, colour, 2)

            if form is not None and team == off_cls:
                if tid in form["line_ids"]:
                    cv2.circle(img, (int(x), int(y + bh)), 4, C_LINE, -1, cv2.LINE_AA)
                else:
                    # every skill player gets a route, not just the split-out
                    # ones, otherwise a bunched formation draws almost nothing
                    if tid in form["wide"]:
                        label(img, "WR", (p1[0], p1[1] - 6), C_ROUTE)
                    if not pre:
                        route_hist[tid].append((int(x), int(y)))

            if motion and tid == motion["id"] and pre and idx >= motion["start"]:
                label(img, "MOTION", (p1[0] - 6, p1[1] - 6), C_MOTION, 0.55)

        # routes
        for tid, pts in route_hist.items():
            trail = pts[-args.route_len:]
            if len(trail) > 1:
                cv2.polylines(img, [np.int32(trail)], False, C_ROUTE, 2, cv2.LINE_AA)

        # motion path
        if motion and motion_path:
            upto = [(px, py) for fi, px, py in motion_path if fi <= idx]
            if len(upto) > 1:
                cv2.polylines(img, [np.int32(upto)], False, C_MOTION, 2, cv2.LINE_AA)
                cv2.arrowedLine(img, upto[0], upto[-1], C_MOTION, 2, cv2.LINE_AA, tipLength=0.04)
                cv2.circle(img, upto[0], 6, C_MOTION, 2, cv2.LINE_AA)

        lines = [
            (f"OFFENSE  {off_name}", C_OFFENSE),
            (f"DEFENSE  {def_name}", C_DEFENSE),
        ]
        if form is not None:
            n_off = (len(form["line_ids"]) + len(form["wide"]) + len(form["backfield"]))
            lines.append((f"PRE-SNAP READ   {n_off} offense tracked", C_LINE))
            lines.append((f"ON THE LINE  {len(form['line_ids'])}", C_LINE))
            lines.append((
                f"WIDE {len(form['wide_left'])}L/{len(form['wide_right'])}R"
                f"   OFF-LINE {len(form['backfield'])}", C_LINE))
        if pre:
            lines.append(("PRE-SNAP", C_LOS))
            if motion and idx >= motion["start"]:
                lines.append(("MOTION DETECTED", C_MOTION))
        else:
            lines.append((f"SNAP  +{(idx - snap) / fps:0.1f}s", C_LOS))
        panel(img, lines)

        writer.write(img)
        idx += 1

    cap.release()
    writer.release()
    return out_path, idx


def main():
    args = parse_args()
    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")
    if not args.source.is_file():
        sys.exit(f"Clip not found: {args.source}")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"tracking {args.source.name} on device {device}...")
    frames, names = track_clip(args, device)

    steps = stabilised_steps(frames)
    energy = energy_curve(steps)
    snap = detect_snap(energy)

    teams = team_of(frames, 0, snap)
    disp = net_displacement(frames, steps, 0, snap)
    off_cls, def_cls, stats = classify_offense(disp, teams)
    if off_cls is None:
        sys.exit("Could not tell the two sides apart; not enough stable tracks.")

    ref = max(0, snap - 6)
    pos = frame_positions(frames, ref, teams)
    widths = [d["w"] for d in frames[ref]] or [28.0]
    helmet_w = statistics.median(widths)
    axis, lateral = play_axis(pos, off_cls, def_cls)
    form = (analyse_formation(pos, axis, lateral, off_cls, def_cls, helmet_w)
            if axis is not None else None)
    motion = detect_motion(frames, steps, teams, off_cls, snap)

    print(f"\nsnap at frame {snap} of {len(frames)}")
    for cls, v in stats.items():
        print(f"  {names[cls]:14} median pre-snap movement {v:6.1f} px")
    print(f"  -> offence {names[off_cls]}, defence {names[def_cls]}")
    if form:
        print(f"  line {len(form['line_ids'])}, wide {len(form['wide_left'])}L/"
              f"{len(form['wide_right'])}R, backfield {len(form['backfield'])}")
    if motion:
        print(f"  motion: track {motion['id']} moved {motion['dist']:.0f} px "
              f"from frame {motion['start']} (team median {motion['median']:.0f} px)")
    else:
        print("  motion: none detected")

    info = {"snap": snap, "off_cls": off_cls, "def_cls": def_cls, "teams": teams,
            "formation": form, "motion": motion, "axis": axis}
    out_path, n = render(args, frames, names, info, device)
    print(f"\nwrote {n} frames to {out_path}")


if __name__ == "__main__":
    main()
