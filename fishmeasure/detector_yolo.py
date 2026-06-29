"""YOLO fish detection (bounding boxes).

The pipeline starts here: detect the fish, then segment only inside its
bounding box (+ padding). Uses an Ultralytics YOLO ``.pt`` model. ultralytics
(and torch) are imported lazily so the rest of the package works without them.

Default weights: ``model.pt`` in the working directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# class-name heuristics for picking the whole-fish box among fish + parts
FISH_KEYWORDS = ("fish", "body")
PART_KEYWORDS = ("head", "tail", "fin", "eye", "mouth", "caudal", "dorsal")

_MODEL_CACHE: dict = {}
DEFAULT_WEIGHTS = "model.pt"


def load_model(weights: str = DEFAULT_WEIGHTS):
    """Load (and cache) an Ultralytics YOLO model."""
    key = str(weights)
    if key not in _MODEL_CACHE:
        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        from ultralytics import YOLO  # lazy import (needs torch)
        _MODEL_CACHE[key] = YOLO(weights)
    return _MODEL_CACHE[key]


def weights_available(weights: str = DEFAULT_WEIGHTS) -> bool:
    return Path(weights).exists()


def detect_boxes(image_bgr, weights: str = DEFAULT_WEIGHTS, conf: float = 0.25,
                 val_preprocess: bool = True, imgsz: int = 640):
    """Run YOLO and return a list of (name, confidence, [x0,y0,x1,y1]).

    With ``val_preprocess`` (default) the image is square-letterboxed to
    ``imgsz`` (matching ``model.val()``: auto=False, scaleup=False), predicted
    in that space, and boxes are mapped back to the input image's pixels — so
    callers always get full-resolution coordinates.
    """
    from .preprocess import letterbox, scale_box_back
    model = load_model(weights)

    if val_preprocess:
        lb_img, r, (dw, dh) = letterbox(image_bgr, imgsz)
        res = model.predict(lb_img, imgsz=imgsz, conf=conf, verbose=False)[0]
    else:
        res = model.predict(image_bgr, conf=conf, verbose=False)[0]

    names = res.names
    boxes = getattr(res, "boxes", None)
    out = []
    if boxes is None or len(boxes) == 0:
        return out
    for i in range(len(boxes)):
        cls = int(boxes.cls[i])
        name = str(names.get(cls, cls)).lower() if isinstance(names, dict) else str(names[cls]).lower()
        xyxy = [float(v) for v in boxes.xyxy[i].tolist()]
        if val_preprocess:
            xyxy = scale_box_back(xyxy, r, dw, dh)
        out.append((name, float(boxes.conf[i]), xyxy))
    return out


def detect_fish(image_bgr, weights: str = DEFAULT_WEIGHTS, conf: float = 0.25,
                fish_class: str | None = None):
    """Return the best whole-fish bbox [x0,y0,x1,y1], or None if no fish found.

    Picks the highest-confidence detection whose class looks like a whole fish
    (name contains 'fish' and not a part keyword). Falls back to the
    largest-area detection if no class matches.
    """
    cands = detect_boxes(image_bgr, weights=weights, conf=conf)
    if not cands:
        return None
    if fish_class is not None:
        fish = [c for c in cands if c[0] == fish_class.lower()]
    else:
        fish = [c for c in cands
                if any(k in c[0] for k in FISH_KEYWORDS)
                and not any(p in c[0] for p in PART_KEYWORDS)]
    if not fish:
        # no class matched -> use the largest box overall
        def area(c):
            x0, y0, x1, y1 = c[2]
            return (x1 - x0) * (y1 - y0)
        return max(cands, key=area)[2]
    return max(fish, key=lambda c: c[1])[2]


def detect_parts(image_bgr, weights: str = DEFAULT_WEIGHTS, conf: float = 0.25):
    """Return {part_name: (cx, cy)} for detected fish parts (head/tail/...)."""
    parts = {}
    for name, cf, (x0, y0, x1, y1) in detect_boxes(image_bgr, weights=weights, conf=conf):
        if any(p in name for p in PART_KEYWORDS):
            parts.setdefault(name, ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    return parts


def pad_bbox(bbox, image_shape, pad: float = 0.2):
    """Expand a bbox by ``pad`` fraction on each side, clamped to the image."""
    H, W = image_shape[:2]
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    x0 -= w * pad; x1 += w * pad
    y0 -= h * pad; y1 += h * pad
    return (max(0, int(round(x0))), max(0, int(round(y0))),
            min(W, int(round(x1))), min(H, int(round(y1))))
