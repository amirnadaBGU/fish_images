"""SAM / SAM 2 segmentation prompted by the YOLO detection box.

Supports:
  - SAM 2   (sam2_t.pt / sam2_s.pt / sam2_b.pt / sam2_l.pt)   ultralytics >= 8.3
  - SAM 2.1 (sam2.1_b.pt etc.)                                  ultralytics >= 8.3
  - MobileSAM (mobile_sam.pt)  legacy fallback

Ultralytics auto-downloads the requested weights on first use.

API is identical to ``segmentation.segment_in_bbox``:
    segment_with_box(image_bgr, bbox, weights=...) -> uint8 0/255 mask (full frame)
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_SAM_WEIGHTS = "sam2_b.pt"   # SAM 2 base; ~80 MB; good quality/speed balance
_SAM_CACHE: dict = {}

_SAM2_PREFIXES = ("sam2",)  # matches sam2_*.pt and sam2.1_*.pt


def _is_sam2(weights: str) -> bool:
    return any(str(weights).lower().startswith(p) for p in _SAM2_PREFIXES)


def load_sam(weights: str = DEFAULT_SAM_WEIGHTS):
    """Load and cache a SAM or SAM 2 model via ultralytics."""
    key = str(weights)
    if key not in _SAM_CACHE:
        from ultralytics import SAM  # lazy (needs torch); works for SAM1, SAM2, MobileSAM
        _SAM_CACHE[key] = SAM(weights)
    return _SAM_CACHE[key]


def segment_with_box(image_bgr: np.ndarray, bbox,
                     weights: str = DEFAULT_SAM_WEIGHTS) -> np.ndarray:
    """Return a full-frame 0/255 mask for the fish inside ``bbox`` via SAM/SAM2.

    ``bbox`` is [x0, y0, x1, y1] in full-image pixels (the YOLO detection box).
    SAM 2 gives substantially better boundary quality than MobileSAM, especially
    along fins and translucent edges.
    """
    H, W = image_bgr.shape[:2]
    model = load_sam(weights)

    # SAM 2 accepts the same bboxes= kwarg as SAM 1
    res = model(image_bgr, bboxes=[list(map(float, bbox))], verbose=False)[0]

    masks = getattr(res, "masks", None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        return np.zeros((H, W), np.uint8)

    data = masks.data
    arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)  # (n, h, w)

    # pick the largest-area mask (model may return several)
    areas = arr.reshape(arr.shape[0], -1).sum(axis=1)
    best = arr[int(np.argmax(areas))]
    mask = (best > 0.5).astype(np.uint8) * 255

    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def sam_available(weights: str = DEFAULT_SAM_WEIGHTS) -> bool:
    """True if SAM/SAM2 can be used (ultralytics importable)."""
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def sam_info(weights: str = DEFAULT_SAM_WEIGHTS) -> str:
    """Human-readable label for the active segmenter."""
    name = str(weights)
    if name.startswith("sam2.1"):
        return f"SAM 2.1 ({name})"
    if name.startswith("sam2"):
        return f"SAM 2 ({name})"
    if "mobile" in name.lower():
        return f"MobileSAM ({name})"
    return f"SAM ({name})"
