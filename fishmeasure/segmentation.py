"""Fish segmentation.

Two stages:
  1. Coarse color separation in LAB space. The fish is warm (brown/red) and the
     water is cyan/blue, so (a* + b*) gives strong contrast; Otsu thresholds it.
  2. Refinement with OpenCV GrabCut (iterative graph-cut). GrabCut is run on a
     downscaled copy for memory safety, then the mask is upscaled back.

This is the classical fallback used because SAM/deep models could not be
installed in the original environment (see CLAUDE.md). The interface is kept
simple so a SAM-based segmenter can be dropped in later with the same signature:
    segment(image_bgr) -> mask (uint8, 0/255, full resolution)
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _largest_component(mask: np.ndarray, min_size: int = 2000) -> np.ndarray:
    lbl, n = ndimage.label(mask > 0)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] < min_size:
        return np.zeros_like(mask)
    return ((lbl == best).astype(np.uint8)) * 255


def color_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Coarse warm-vs-cool LAB segmentation, returns the largest warm blob."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128.0  # green(-) <-> red(+)
    b = lab[:, :, 2] - 128.0  # blue(-) <-> yellow(+)
    warm = a + b
    warm = (warm - warm.min()) / (warm.max() - warm.min() + 1e-9)
    wu = (warm * 255).astype(np.uint8)
    _, th = cv2.threshold(wu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    return _largest_component(th)


def grabcut_refine(image_bgr: np.ndarray, seed: np.ndarray,
                   scale: float = 1 / 3, iters: int = 8) -> np.ndarray:
    """Refine a coarse mask with GrabCut at reduced resolution (memory safe)."""
    H, W = image_bgr.shape[:2]
    small = cv2.resize(image_bgr, (max(1, int(W * scale)), max(1, int(H * scale))),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    m = cv2.resize(seed, (w, h), interpolation=cv2.INTER_NEAREST)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    probable = cv2.dilate(m, k, iterations=2)
    sure = cv2.erode(m, k, iterations=1)

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[probable > 0] = cv2.GC_PR_FGD
    gc[sure > 0] = cv2.GC_FGD

    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)

    res = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kk)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kk)
    res = _largest_component(res)
    res = ndimage.binary_fill_holes(res > 0).astype(np.uint8) * 255
    return cv2.resize(res, (W, H), interpolation=cv2.INTER_NEAREST)


def segment_in_bbox(image_bgr: np.ndarray, bbox, refine: bool = True,
                    grabcut_scale: float = 1 / 3) -> np.ndarray:
    """Segment only inside ``bbox`` (already padded), return a full-frame mask.

    ``bbox`` is (x0, y0, x1, y1) in full-image pixels. The segmentation runs on
    the crop and the result is placed back at the right location; everything
    outside the box is background. Used after YOLO detection so segmentation
    (and the centroid) ignore rings/glare elsewhere in the frame.
    """
    H, W = image_bgr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    full = np.zeros((H, W), np.uint8)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return full
    crop = image_bgr[y0:y1, x0:x1]
    m = segment(crop, refine=refine, grabcut_scale=grabcut_scale)  # may raise
    full[y0:y1, x0:x1] = m
    return full


def segment(image_bgr: np.ndarray, refine: bool = True,
            grabcut_scale: float = 1 / 3) -> np.ndarray:
    """Full segmentation: color mask, optionally refined with GrabCut.

    Returns a full-resolution uint8 mask (0 background, 255 fish).
    Raises ValueError if no fish-like region is found.
    """
    coarse = color_mask(image_bgr)
    if coarse.max() == 0:
        raise ValueError("No fish region found (color segmentation empty).")
    if not refine:
        return coarse
    refined = grabcut_refine(image_bgr, coarse, scale=grabcut_scale)
    # Fall back to the coarse mask if GrabCut collapses.
    if refined.sum() < 0.3 * coarse.sum():
        return coarse
    return refined
