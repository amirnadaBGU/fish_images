"""End-to-end pipeline: image (BGR ndarray) -> MeasureResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .detector_yolo import DEFAULT_WEIGHTS, detect_fish, pad_bbox, weights_available
from .measurement import measure
from .preprocess import apply_filters
from .quality import assess
from .segmentation import segment, segment_in_bbox
from .skeleton import midline_path

_WARNED_NO_WEIGHTS = False


@dataclass
class MeasureResult:
    curved_length_px: float
    straight_tip_to_tip_px: float
    skeleton_arc_px: float
    num_keypoints: int
    keypoints: np.ndarray            # (k, 2) xy  (empty if not measurable)
    midline: np.ndarray             # (N, 2) xy full polyline
    mask: np.ndarray                # uint8 0/255
    com_x: float = float("nan")     # center of mass (mask centroid)
    com_y: float = float("nan")
    tip_tl_x: float = float("nan")  # top-left tip (smaller x+y)
    tip_tl_y: float = float("nan")
    tip_br_x: float = float("nan")  # bottom-right tip (larger x+y)
    tip_br_y: float = float("nan")
    problematic: bool = False        # auto-flagged for review
    flags: List[str] = field(default_factory=list)
    bbox: Optional[list] = None       # YOLO detection box [x0,y0,x1,y1] (full-image px)
    bbox_pad: Optional[list] = None   # padded box actually segmented
    px_per_unit: Optional[float] = None
    unit: str = "px"
    extra: dict = field(default_factory=dict)

    @property
    def curved_length_unit(self) -> Optional[float]:
        if self.px_per_unit and self.curved_length_px == self.curved_length_px:
            return self.curved_length_px / self.px_per_unit
        return None

    @property
    def measurable(self) -> bool:
        return len(self.keypoints) > 0


def _empty(mask, flags, px_per_unit, unit):
    return MeasureResult(
        curved_length_px=float("nan"),
        straight_tip_to_tip_px=float("nan"),
        skeleton_arc_px=float("nan"),
        num_keypoints=0,
        keypoints=np.empty((0, 2), np.float32),
        midline=np.empty((0, 2), np.float32),
        mask=mask,
        problematic=True,
        flags=flags,
        px_per_unit=px_per_unit,
        unit=unit if px_per_unit else "px",
    )


def measure_image(image_bgr, k: int = 9, refine: bool = True,
                  px_per_unit: Optional[float] = None, unit: str = "px",
                  weights: str = DEFAULT_WEIGHTS, use_detector: bool = True,
                  pad: float = 0.2, conf: float = 0.25,
                  fish_class: Optional[str] = None, prefilter: bool = True,
                  segmenter: str = "sam", sam_weights: str = "sam2_b.pt"):
    """Filter, detect the fish, segment inside its bbox (+pad), and measure it.

    Pipeline: apply_filters (CLAHE + sharpen) -> YOLO detect (square-letterbox
    640) -> crop to bbox + ``pad`` -> segment only there -> skeleton ->
    keypoints -> curved length. Detection + segmentation both run on the
    filtered image; geometry is unchanged so coordinates stay in full-res px.
    Set ``prefilter=False`` if the input image is already filtered (callers that
    save the filtered frame do this to avoid filtering twice). If detection
    finds no fish, the frame is flagged ``no_fish``. Falls back to whole-frame
    segmentation if a detector model isn't available. Never raises.
    """
    global _WARNED_NO_WEIGHTS
    proc = apply_filters(image_bgr) if prefilter else image_bgr
    empty_mask = np.zeros(proc.shape[:2], np.uint8)
    det_box = None
    pad_box = None

    if use_detector and weights_available(weights):
        try:
            bbox = detect_fish(proc, weights=weights, conf=conf, fish_class=fish_class)
        except Exception:  # noqa: BLE001 - detector failure -> treat as no fish
            bbox = None
        if bbox is None:
            return _empty(empty_mask, ["no_fish"], px_per_unit, unit)
        det_box = [round(float(v), 1) for v in bbox]
        if segmenter == "sam":
            try:
                from .segmenter_sam import segment_with_box
                mask = segment_with_box(proc, bbox, weights=sam_weights)
            except Exception as _sam_exc:  # noqa: BLE001 - SAM failure -> fall back to grabcut
                print(f"[fishmeasure] SAM failed ({_sam_exc!r}), falling back to GrabCut.")
                pbox = pad_bbox(bbox, proc.shape, pad)
                pad_box = [float(v) for v in pbox]
                try:
                    mask = segment_in_bbox(proc, pbox, refine=refine)
                except Exception:  # noqa: BLE001
                    mask = empty_mask
        else:
            pbox = pad_bbox(bbox, proc.shape, pad)
            pad_box = [float(v) for v in pbox]
            try:
                mask = segment_in_bbox(proc, pbox, refine=refine)
            except Exception:  # noqa: BLE001
                mask = empty_mask
    else:
        if use_detector and not _WARNED_NO_WEIGHTS:
            print(f"[fishmeasure] YOLO weights '{weights}' not found — "
                  "falling back to whole-frame segmentation.")
            _WARNED_NO_WEIGHTS = True
        try:
            mask = segment(proc, refine=refine)
        except Exception:  # noqa: BLE001 - treat as no fish
            mask = empty_mask

    problematic, flags, measurable = assess(mask, proc.shape)
    if not measurable:
        return _empty(mask, flags, px_per_unit, unit)

    try:
        poly = midline_path(mask)
        keys, full, m = measure(poly, mask, k=k)
    except Exception:  # noqa: BLE001 - mask present but unmeasurable
        return _empty(mask, flags + ["skeleton_failed"], px_per_unit, unit)

    return MeasureResult(
        curved_length_px=m["curved_length_px"],
        straight_tip_to_tip_px=m["straight_tip_to_tip_px"],
        skeleton_arc_px=m["skeleton_arc_px"],
        num_keypoints=m["num_keypoints"],
        keypoints=keys, midline=full, mask=mask,
        com_x=m["com_x"], com_y=m["com_y"],
        tip_tl_x=m["tip_tl_x"], tip_tl_y=m["tip_tl_y"],
        tip_br_x=m["tip_br_x"], tip_br_y=m["tip_br_y"],
        problematic=problematic, flags=flags,
        bbox=det_box, bbox_pad=pad_box,
        px_per_unit=px_per_unit, unit=unit if px_per_unit else "px",
    )
