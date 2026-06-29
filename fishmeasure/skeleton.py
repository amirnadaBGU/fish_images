"""Skeleton / medial-axis extraction and the geodesic longest path.

The mask is smoothed, thinned to a 1-pixel-wide skeleton (morphological
thinning via skimage.skeletonize), then treated as an 8-connected graph. The
fish's main body axis is the longest geodesic path on that graph, found with a
double-BFS (find the farthest node A from an arbitrary start, then the farthest
node B from A; the A->B path is the diameter of the skeleton tree).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _neighbors(p, pts):
    y, x = p
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            q = (y + dy, x + dx)
            if q in pts:
                yield q


def _bfs(src, pts):
    dist = {src: 0}
    prev = {src: None}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in _neighbors(u, pts):
            if v not in dist:
                dist[v] = dist[u] + 1
                prev[v] = u
                dq.append(v)
    return dist, prev


def midline_path(mask: np.ndarray, smooth_ksize: int = 31) -> np.ndarray:
    """Return the body-axis polyline as an (N, 2) array of (x, y) points.

    Points are ordered from one tip of the skeleton to the other.
    """
    m = (mask > 0).astype(np.uint8) * 255
    if smooth_ksize and smooth_ksize >= 3:
        k = smooth_ksize | 1  # force odd
        m = cv2.GaussianBlur(m, (k, k), 0)
    binary = m > 127

    skel = skeletonize(binary)
    ys, xs = np.where(skel)
    if len(xs) < 2:
        raise ValueError("Skeleton too small to measure.")
    pts = set(zip(ys.tolist(), xs.tolist()))

    # endpoints have a single skeleton neighbor; prefer them as BFS seeds
    endpoints = [p for p in pts if sum(1 for _ in _neighbors(p, pts)) == 1]
    start = endpoints[0] if endpoints else next(iter(pts))

    d1, _ = _bfs(start, pts)
    a = max(d1, key=d1.get)
    d2, prev2 = _bfs(a, pts)
    b = max(d2, key=d2.get)

    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = prev2[cur]
    path.reverse()

    return np.array([[x, y] for (y, x) in path], dtype=np.float32)
