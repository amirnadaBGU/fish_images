"""Turn a midline polyline into keypoints and a curved length.

Steps:
  1. Extend both ends of the skeleton path to the true mask boundary, since the
     medial axis stops short of the snout and the tail-fin tip.
  2. Resample the polyline into K evenly spaced keypoints along arc length.
  3. The curved length is the sum of distances between consecutive keypoints.
"""
from __future__ import annotations

import numpy as np


def _extend_to_boundary(start_xy, dir_xy, mask, step=2.0, max_steps=800):
    H, W = mask.shape
    d = dir_xy / (np.linalg.norm(dir_xy) + 1e-9)
    p = start_xy.astype(np.float32).copy()
    last = p.copy()
    for _ in range(max_steps):
        p = p + d * step
        xi, yi = int(round(p[0])), int(round(p[1]))
        if xi < 0 or yi < 0 or xi >= W or yi >= H or mask[yi, xi] == 0:
            break
        last = p.copy()
    return last


def extend_to_tips(poly: np.ndarray, mask: np.ndarray, look: int = 15) -> np.ndarray:
    """Extend a midline polyline outward at both ends to the mask boundary."""
    m = (mask > 0).astype(np.uint8)
    n = len(poly)
    look = min(look, n - 1)
    dir_start = poly[0] - poly[look]
    dir_end = poly[-1] - poly[-1 - look]
    tip0 = _extend_to_boundary(poly[0], dir_start, m)
    tip1 = _extend_to_boundary(poly[-1], dir_end, m)
    return np.vstack([tip0, poly, tip1])


def resample_keypoints(poly: np.ndarray, k: int = 9) -> np.ndarray:
    """Resample a polyline into k points evenly spaced by arc length."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    targets = np.linspace(0.0, total, k)
    out = []
    for t in targets:
        i = int(np.searchsorted(arc, t))
        i = min(max(i, 1), len(arc) - 1)
        r = (t - arc[i - 1]) / (arc[i] - arc[i - 1] + 1e-9)
        out.append(poly[i - 1] + r * (poly[i] - poly[i - 1]))
    return np.array(out, dtype=np.float32)


def polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def center_of_mass(mask: np.ndarray):
    """Centroid (cx, cy) of the fish mask in pixel coordinates."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    return (float(xs.mean()), float(ys.mean()))


def label_tips(keys: np.ndarray):
    """Label the two fish tips by image position.

    Returns (tl, br) where tl is the tip with the smaller x+y (more
    top-left) and br the larger (more bottom-right). Stable regardless of
    swimming direction.
    """
    a, b = keys[0], keys[-1]
    if (a[0] + a[1]) <= (b[0] + b[1]):
        return a, b
    return b, a


def measure(poly: np.ndarray, mask: np.ndarray, k: int = 9):
    """Return (keypoints, full_polyline, metrics dict)."""
    full = extend_to_tips(poly, mask)
    keys = resample_keypoints(full, k=k)
    metrics = curved_metrics(keys, mask, k)
    metrics["skeleton_arc_px"] = polyline_length(full)
    return keys, full, metrics


def curved_metrics(keys: np.ndarray, mask: np.ndarray, k: int | None = None):
    """Metrics derived purely from keypoints (+ mask for CoM).

    Reused by the editor: when keypoints are moved, call this to recompute.
    """
    if k is None:
        k = len(keys)
    cx, cy = center_of_mass(mask)
    tl, br = label_tips(keys)
    return {
        "curved_length_px": polyline_length(keys),
        "straight_tip_to_tip_px": float(np.linalg.norm(keys[0] - keys[-1])),
        "num_keypoints": int(k),
        "com_x": cx,
        "com_y": cy,
        "tip_tl_x": float(tl[0]),
        "tip_tl_y": float(tl[1]),
        "tip_br_x": float(br[0]),
        "tip_br_y": float(br[1]),
    }
