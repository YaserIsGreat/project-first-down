"""Estimate where the camera is looking, frame by frame.

The all-22 camera pans and zooms throughout a play. Anything drawn in screen
coordinates therefore slides off the grass as soon as the operator moves, and
any measurement taken in screen pixels is contaminated by camera motion rather
than player motion.

This module estimates a homography from every frame back to a chosen reference
frame by tracking features on the field itself. Players are masked out first,
since they move independently of the camera and would otherwise be treated as
evidence about where the camera went.

With those homographies in hand:

  * a point can be converted to a stable field frame, so distances and
    displacements are comparable across a zoom
  * a shape defined on the field can be projected back into any frame, so a
    line of scrimmage stays on the grass instead of on the screen
"""

import cv2
import numpy as np

# Feature tracking settings. The field is mostly flat green, so usable corners
# come from yard lines, numbers, hashes and logos: allow plenty of them and a
# low quality floor.
MAX_CORNERS = 1500
QUALITY = 0.006
MIN_DISTANCE = 8
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# How far to grow each helmet box before excluding it. A helmet box is small
# but the player under it is not, and his jersey and limbs move with him.
PLAYER_MASK_SCALE = 3.0


def _player_mask(shape, dets):
    """White where the camera can be measured, black over players."""
    mask = np.full(shape[:2], 255, dtype=np.uint8)
    for d in dets:
        half_w = d["w"] * PLAYER_MASK_SCALE / 2
        # Extend well below the helmet to cover the body.
        x0 = int(d["x"] - half_w)
        x1 = int(d["x"] + half_w)
        y0 = int(d["y"] - d["h"] * 1.5)
        y1 = int(d["y"] + d["h"] * 6.0)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 0, -1)
    return mask


def _step_homography(prev_gray, gray, prev_mask):
    """Homography mapping the previous frame onto the current one."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, MAX_CORNERS, QUALITY, MIN_DISTANCE,
                                 mask=prev_mask, blockSize=7)
    if p0 is None or len(p0) < 25:
        return None, 0

    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
    if p1 is None:
        return None, 0

    # Verify by tracking back; keep only points that land where they started.
    p0r, status_r, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK_PARAMS)
    if p0r is None:
        return None, 0
    ok = (status.ravel() == 1) & (status_r.ravel() == 1)
    ok &= np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1) < 1.5

    a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    if len(a) < 20:
        return None, len(a)

    H, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.5, maxIters=3000, confidence=0.995)
    if H is None:
        return None, len(a)
    return H, int(inliers.sum()) if inliers is not None else len(a)


def camera_homographies(source, per_frame_dets, ref_idx):
    """Homography mapping each frame's pixels into the reference frame.

    Returns (homographies, report). Frames whose estimate fails inherit the
    previous transform, which keeps the sequence usable rather than dropping a
    hole in the middle of it.
    """
    cap = cv2.VideoCapture(str(source))
    steps = []          # frame i-1 -> frame i
    failures = 0
    prev_gray = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            dets = per_frame_dets[idx - 1] if idx - 1 < len(per_frame_dets) else []
            H, _ = _step_homography(prev_gray, gray, _player_mask(gray.shape, dets))
            if H is None:
                H = np.eye(3)
                failures += 1
            steps.append(H)
        prev_gray = gray
        idx += 1
    cap.release()

    n = idx
    if n == 0:
        return [], {"frames": 0, "failures": 0}

    # Chain steps into "frame i -> frame 0", then rebase onto the reference.
    to_first = [np.eye(3)]
    for H in steps:
        to_first.append(to_first[-1] @ np.linalg.inv(H))

    ref_idx = max(0, min(ref_idx, len(to_first) - 1))
    rebase = np.linalg.inv(to_first[ref_idx])
    homs = [rebase @ H for H in to_first]
    homs = [H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H for H in homs]

    return homs, {"frames": n, "failures": failures}


def apply(H, pts):
    """Transform an (N, 2) array of points by a homography."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def apply_one(H, pt):
    return apply(H, [pt])[0]


def scale_of(H):
    """Roughly how much this transform magnifies, for sanity checks."""
    return float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
