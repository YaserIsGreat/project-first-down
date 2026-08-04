"""Rule-based pre-snap analysis drawn onto the field.

Turns tracked helmet coordinates into a read of the play:

  * which team has the ball, from which side holds still before the snap
  * where the line of scrimmage is
  * who is on the line, who is split wide, who is off it
  * pre-snap motion, with the path the man in motion took
  * post-snap routes

Everything is geometry over helmet coordinates. There is no second model.

The important detail is that none of it happens in screen coordinates. The
all-22 camera pans and zooms through the play -- often more than doubling
magnification once the ball is away -- so field_motion.py estimates a
homography from every frame back to the snap. Positions are measured in that
stable field frame, and overlays are defined there and projected back, which
keeps the line of scrimmage on the grass through a zoom instead of sliding
across the screen.

    python analyze_play.py --source nflTrimmedAll22/ramsOffence05.mp4
    python analyze_play.py --source nflTrimmedAll22/ramsOffence05.mp4 --slow 0.6
    python analyze_play.py --source clip.mp4 --anchor screen   # old behaviour

What is reliable: team identification, the line of scrimmage, and motion
detection all lean on large, robust differences. Route lines depend on one
track surviving the pile, so they fragment under occlusion.
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

import field_motion as fm

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
    p.add_argument("-s", "--source", type=Path, required=True)
    p.add_argument("-w", "--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None)
    p.add_argument("--slow", type=float, default=0.7,
                   help="playback speed multiplier")
    p.add_argument("--route-len", type=int, default=32,
                   help="frames of route trail kept behind each skill player; "
                        "trails are drawn on the field, so a zoom magnifies them too")
    p.add_argument("--anchor", choices=["field", "screen"], default="field",
                   help="field: pin overlays to the grass through pans and zooms")
    return p.parse_args()


# --------------------------------------------------------------------------
# detection and tracking
# --------------------------------------------------------------------------

def dedupe(dets):
    """Drop near-coincident boxes: two ids on one helmet double-count a player."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if any((d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2
               < (0.6 * max(d["w"], k["w"])) ** 2 for k in keep):
            continue
        keep.append(d)
    return keep


def track_clip(args, device):
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


# --------------------------------------------------------------------------
# field-space positions
# --------------------------------------------------------------------------

def field_positions(frames, homs):
    """Per-frame {track id: field-space point}."""
    out = []
    for i, f in enumerate(frames):
        H = homs[i] if i < len(homs) else np.eye(3)
        if not f:
            out.append({})
            continue
        pts = fm.apply(H, [(d["x"], d["y"]) for d in f])
        out.append({d["id"]: p for d, p in zip(f, pts)})
    return out


def median_fallback_positions(frames):
    """Screen-space positions with the median shift removed (no homography)."""
    out, offset = [], np.zeros(2)
    prev = {}
    for f in frames:
        cur = {d["id"]: np.array([d["x"], d["y"]]) for d in f}
        shared = [i for i in cur if i in prev]
        if len(shared) >= 4:
            deltas = np.array([cur[i] - prev[i] for i in shared])
            offset = offset + np.median(deltas, axis=0)
        out.append({i: p - offset for i, p in cur.items()})
        prev = cur
    return out


def steps_from(positions):
    steps = []
    for i in range(len(positions)):
        if i == 0:
            steps.append({})
            continue
        a, b = positions[i - 1], positions[i]
        steps.append({t: b[t] - a[t] for t in b if t in a})
    return steps


def smooth(series, k=6):
    return [statistics.fmean(series[max(0, i - k):i + k + 1]) for i in range(len(series))]


def energy_curve(steps):
    out = []
    for s in steps:
        vals = [float(np.linalg.norm(v)) for v in s.values()]
        out.append(statistics.fmean(vals) if vals else 0.0)
    return smooth(out)


def detect_snap(energy):
    """Find the moment a settled offence bursts into motion.

    A first-jump-above-baseline rule is not enough: a clip can open with the
    camera still finding the play, and a late pan can outscore the snap. What
    identifies a snap is the transition -- a genuinely quiet stretch followed
    by sustained movement -- so score candidates on that contrast and require
    the run-up to sit below the clip's median motion.
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
            continue
        score = post - pre
        if best_score is None or score > best_score:
            best_score, best_i = score, i
    return best_i if best_i is not None else n // 3


# --------------------------------------------------------------------------
# play geometry, all in field space
# --------------------------------------------------------------------------

def net_displacement(steps, lo, hi):
    acc = defaultdict(lambda: np.zeros(2))
    seen = defaultdict(int)
    for i in range(max(1, lo), min(hi, len(steps))):
        for tid, v in steps[i].items():
            acc[tid] += v
            seen[tid] += 1
    need = max(4, (hi - lo) * 0.25)
    return {t: float(np.linalg.norm(acc[t])) for t in acc if seen[t] >= need}


def team_of(frames, lo, hi):
    votes = defaultdict(lambda: defaultdict(int))
    for f in frames[lo:hi]:
        for d in f:
            votes[d["id"]][d["cls"]] += 1
    return {t: max(v, key=v.get) for t, v in votes.items()}


def classify_offense(disp, teams):
    """The offence is the side that holds still before the snap.

    Both teams put five men on the line, so counting the front cannot separate
    them. But the offence must be set and the defence need not be, which shows
    up directly as a lower median pre-snap displacement. The largest mover on
    each side is dropped so a man in motion cannot flip the verdict.
    """
    by_team = defaultdict(list)
    for tid, d in disp.items():
        if tid in teams:
            by_team[teams[tid]].append(d)

    stats = {}
    for cls, vals in by_team.items():
        if len(vals) < 3:
            continue
        stats[cls] = statistics.median(sorted(vals)[:-1])
    if len(stats) < 2:
        return None, None, stats
    order = sorted(stats, key=lambda c: stats[c])
    return order[0], order[1], stats


def play_axis(pts_by_team):
    o, d = pts_by_team["off"], pts_by_team["def"]
    if len(o) < 2 or len(d) < 2:
        return None, None
    axis = np.mean(o, axis=0) - np.mean(d, axis=0)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return None, None
    axis = axis / n
    return axis, np.array([-axis[1], axis[0]])


def analyse_formation(off_pts, def_pts, axis, lateral, helmet_w):
    if len(off_pts) < 5 or not def_pts:
        return None

    s_off = {t: float(np.dot(p, axis)) for t, p in off_pts.items()}
    s_def = {t: float(np.dot(p, axis)) for t, p in def_pts.items()}
    los = (max(s_def.values()) + min(s_off.values())) / 2.0

    line_ids = [t for t, _ in sorted(s_off.items(), key=lambda kv: kv[1])[:5]]
    lat = {t: float(np.dot(p, lateral)) for t, p in off_pts.items()}
    line_lat = [lat[t] for t in line_ids]
    lo, hi = min(line_lat), max(line_lat)
    span = max(hi - lo, 1.0)
    margin = max(span * 0.22, 2.2 * helmet_w)

    wide_left, wide_right, off_line = [], [], []
    for t in off_pts:
        if t in line_ids:
            continue
        if lat[t] < lo - margin:
            wide_left.append(t)
        elif lat[t] > hi + margin:
            wide_right.append(t)
        else:
            off_line.append(t)

    all_lat = list(lat.values()) + [float(np.dot(p, lateral)) for p in def_pts.values()]
    return {
        "los": los, "line_ids": line_ids,
        "wide_left": wide_left, "wide_right": wide_right, "off_line": off_line,
        "wide": wide_left + wide_right,
        "lat_min": min(all_lat), "lat_max": max(all_lat),
    }


def detect_motion(steps, positions, teams, off_cls, snap):
    lo = max(1, snap - 90)
    disp = net_displacement(steps, lo, snap)
    off = {t: d for t, d in disp.items() if teams.get(t) == off_cls}
    if len(off) < 4:
        return None
    med = statistics.median(off.values())
    tid, best = max(off.items(), key=lambda kv: kv[1])
    if best < max(med * 4.0, 45.0):
        return None

    start, run = snap, 0
    for i in range(lo, snap):
        v = steps[i].get(tid)
        if v is not None and float(np.linalg.norm(v)) > 0.8:
            run += 1
            if run >= 4:
                start = i - run
                break
        else:
            run = 0
    return {"id": tid, "start": max(lo, start), "dist": best, "median": med}


def field_path(positions, tid, lo, hi):
    pts = []
    for i in range(max(0, lo), min(hi, len(positions))):
        p = positions[i].get(tid)
        if p is not None:
            pts.append((i, p))
    return pts


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def panel(img, lines, org=(24, 24), width=470):
    h = 20 + 30 * len(lines)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + h), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
    cv2.rectangle(img, (x, y), (x + width, y + h), (90, 90, 90), 1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (x + 14, y + 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)


def label(img, text, pt, colour, scale=0.5):
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def to_screen(Hinv, pts):
    """Project field-space points into the current frame."""
    if Hinv is None or len(pts) == 0:
        return None
    try:
        return np.int32(fm.apply(Hinv, pts))
    except cv2.error:
        return None


def visible(pts, shape, pad=6000):
    """Reject projections that have blown up behind the camera."""
    if pts is None or len(pts) == 0:
        return False
    h, w = shape[:2]
    return bool(np.all(np.abs(pts[:, 0]) < w + pad) and np.all(np.abs(pts[:, 1]) < h + pad))


def render(args, frames, names, info, homs):
    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{args.source.stem}_analysis.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(fps * args.slow, 1.0), (w, h))

    snap, form, motion = info["snap"], info["formation"], info["motion"]
    off_cls, def_cls, axis = info["off_cls"], info["def_cls"], info["axis"]
    positions = info["positions"]
    off_name = names[off_cls].replace("_helmet", "").upper()
    def_name = names[def_cls].replace("_helmet", "").upper()

    # Line of scrimmage as a field-space segment, spanning the players plus a
    # margin. Defined once on the field; projected into every frame.
    los_seg = None
    if form is not None and axis is not None:
        pad = (form["lat_max"] - form["lat_min"]) * 0.35 + 60
        lateral = np.array([-axis[1], axis[0]])
        base = axis * form["los"]
        los_seg = np.array([base + lateral * (form["lat_min"] - pad),
                            base + lateral * (form["lat_max"] + pad)])

    motion_path = field_path(positions, motion["id"], motion["start"], snap) if motion else []
    route_hist = defaultdict(list)

    idx = 0
    while True:
        ok, img = cap.read()
        if not ok or idx >= len(frames):
            break
        pre = idx < snap

        Hinv = None
        if idx < len(homs):
            try:
                Hinv = np.linalg.inv(homs[idx])
            except np.linalg.LinAlgError:
                Hinv = None

        if los_seg is not None:
            seg = to_screen(Hinv, los_seg)
            if visible(seg, img.shape):
                cv2.line(img, tuple(seg[0]), tuple(seg[1]), C_LOS, 2, cv2.LINE_AA)

        for d in frames[idx]:
            tid = d["id"]
            team = info["teams"].get(tid, d["cls"])
            colour = C_OFFENSE if team == off_cls else C_DEFENSE
            x, y, bw, bh = d["x"], d["y"], d["w"], d["h"]
            p1 = (int(x - bw / 2), int(y - bh / 2))
            p2 = (int(x + bw / 2), int(y + bh / 2))
            cv2.rectangle(img, p1, p2, colour, 2)

            if form is not None and team == off_cls:
                if tid in form["line_ids"]:
                    cv2.circle(img, (int(x), int(y + bh)), 4, C_LINE, -1, cv2.LINE_AA)
                else:
                    if tid in form["wide"]:
                        label(img, "WR", (p1[0], p1[1] - 6), C_ROUTE)
                    if not pre and tid in positions[idx]:
                        route_hist[tid].append(positions[idx][tid])

            if motion and tid == motion["id"] and pre and idx >= motion["start"]:
                label(img, "MOTION", (p1[0] - 6, p1[1] - 6), C_MOTION, 0.55)

        for tid, pts in route_hist.items():
            # Only trail players still being detected, otherwise a lost track
            # leaves its line hanging across the frame for the rest of the play.
            if tid not in positions[idx]:
                continue
            trail = to_screen(Hinv, np.array(pts[-args.route_len:]))
            if trail is not None and len(trail) > 1 and visible(trail, img.shape):
                cv2.polylines(img, [trail], False, C_ROUTE, 2, cv2.LINE_AA)

        if motion and motion_path:
            upto = np.array([p for fi, p in motion_path if fi <= idx])
            trail = to_screen(Hinv, upto) if len(upto) else None
            if trail is not None and len(trail) > 1 and visible(trail, img.shape):
                cv2.polylines(img, [trail], False, C_MOTION, 2, cv2.LINE_AA)
                cv2.arrowedLine(img, tuple(trail[0]), tuple(trail[-1]),
                                C_MOTION, 2, cv2.LINE_AA, tipLength=0.04)
                cv2.circle(img, tuple(trail[0]), 6, C_MOTION, 2, cv2.LINE_AA)

        lines = [(f"OFFENSE  {off_name}", C_OFFENSE),
                 (f"DEFENSE  {def_name}", C_DEFENSE)]
        if form is not None:
            n_off = len(form["line_ids"]) + len(form["wide"]) + len(form["off_line"])
            lines.append((f"PRE-SNAP READ   {n_off} offense tracked", C_LINE))
            lines.append((f"ON THE LINE  {len(form['line_ids'])}", C_LINE))
            lines.append((f"WIDE {len(form['wide_left'])}L/{len(form['wide_right'])}R"
                          f"   OFF-LINE {len(form['off_line'])}", C_LINE))
        if pre:
            lines.append(("PRE-SNAP", C_LOS))
            if motion and idx >= motion["start"]:
                lines.append(("MOTION DETECTED", C_MOTION))
        else:
            lines.append((f"SNAP  +{(idx - snap) / fps:0.1f}s", C_LOS))
        if info["zoom"] is not None:
            lines.append((f"camera zoom  {info['zoom'][idx]:.2f}x", (170, 170, 170)))
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

    homs, zoom = [], None
    if args.anchor == "field":
        print("estimating camera motion...")
        homs, rep = fm.camera_homographies(args.source, frames, 0)
        print(f"  {rep['frames']} frames, {rep['failures']} failed estimates")
        positions = field_positions(frames, homs)
    else:
        positions = median_fallback_positions(frames)

    steps = steps_from(positions)
    snap = detect_snap(energy_curve(steps))

    # Re-anchor the field frame on the snap so measurements and overlays are
    # expressed where the play actually starts.
    if homs:
        try:
            rebase = np.linalg.inv(homs[snap])
            homs = [rebase @ H for H in homs]
            positions = field_positions(frames, homs)
            steps = steps_from(positions)
        except np.linalg.LinAlgError:
            pass
        zoom = [1.0 / max(fm.scale_of(H), 1e-6) for H in homs]

    teams = team_of(frames, 0, snap)
    disp = net_displacement(steps, 1, snap)
    off_cls, def_cls, stats = classify_offense(disp, teams)
    if off_cls is None:
        sys.exit("Could not tell the two sides apart; not enough stable tracks.")

    ref = max(0, snap - 6)
    off_pts = {t: p for t, p in positions[ref].items() if teams.get(t) == off_cls}
    def_pts = {t: p for t, p in positions[ref].items() if teams.get(t) == def_cls}
    helmet_w = statistics.median([d["w"] for d in frames[ref]] or [28.0])
    axis, lateral = play_axis({"off": list(off_pts.values()), "def": list(def_pts.values())})
    form = (analyse_formation(off_pts, def_pts, axis, lateral, helmet_w)
            if axis is not None else None)
    motion = detect_motion(steps, positions, teams, off_cls, snap)

    print(f"\nsnap at frame {snap} of {len(frames)}")
    if zoom:
        print(f"camera zoom over the clip: {min(zoom):.2f}x to {max(zoom):.2f}x")
    for cls, v in stats.items():
        print(f"  {names[cls]:14} median pre-snap movement {v:6.1f} px")
    print(f"  -> offence {names[off_cls]}, defence {names[def_cls]}")
    if form:
        print(f"  line {len(form['line_ids'])}, wide {len(form['wide_left'])}L/"
              f"{len(form['wide_right'])}R, off-line {len(form['off_line'])}")
    if motion:
        print(f"  motion: track {motion['id']} moved {motion['dist']:.0f} px "
              f"from frame {motion['start']} (team median {motion['median']:.0f} px)")
    else:
        print("  motion: none detected")

    info = {"snap": snap, "off_cls": off_cls, "def_cls": def_cls, "teams": teams,
            "formation": form, "motion": motion, "axis": axis,
            "positions": positions, "zoom": zoom}
    out_path, n = render(args, frames, names, info, homs)
    print(f"\nwrote {n} frames to {out_path}")


if __name__ == "__main__":
    main()
