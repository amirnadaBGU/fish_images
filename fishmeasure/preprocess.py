"""Image preprocessing applied BEFORE detection and segmentation.

- ``apply_filters``: CLAHE on the L channel (local contrast) + an unsharp-mask
  sharpen. Pixel geometry is preserved, so masks/keypoints/lengths computed on
  the filtered image map 1:1 back to the original frame.
- ``letterbox``: replicates Ultralytics ``LetterBox(new, auto=False,
  scaleup=False)`` — resize keeping aspect ratio (no upscaling) onto a square
  ``imgsz`` canvas padded with grey (114). Used to feed YOLO exactly like
  ``model.val()`` does. ``scale_box_back`` inverts it so detections return to
  full-resolution coordinates.

Pure OpenCV/NumPy — no torch — so this module imports anywhere.
"""
from __future__ import annotations

import cv2
import numpy as np


def apply_filters(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE (on L) + unsharp-mask sharpening. Returns a same-size BGR image."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    clahed = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(clahed, (55, 55), 0)
    sharpened = cv2.addWeighted(clahed, 1.8, blurred, -0.8, 0)
    return sharpened


def letterbox(image, new_shape=640, color=(114, 114, 114), scaleup=False):
    """Resize+pad to a square ``new_shape`` (auto=False). Returns (img, r, (dw, dh))."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (new_shape[1] - nw) / 2.0, (new_shape[0] - nh) / 2.0
    resized = image if (nw, nh) == (w, h) else cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (dw, dh)


def scale_box_back(box, r, dw, dh):
    """Map a [x0,y0,x1,y1] box from letterbox space back to original pixels."""
    x0, y0, x1, y1 = box
    return [(x0 - dw) / r, (y0 - dh) / r, (x1 - dw) / r, (y1 - dh) / r]
