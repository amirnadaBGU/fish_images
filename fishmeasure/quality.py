"""Automatic quality flags for a frame's fish mask.

Flags a frame as *problematic* (so it can be reviewed/skipped) when:
  - ``no_fish``      : nothing segmented.
  - ``tiny``         : segmented blob too small to trust (likely noise).
  - ``oversegmented``: blob covers most of the frame (segmentation grabbed
                       background / water glare).
  - ``cut_off:<edges>`` : the fish touches one or more image borders, i.e. it
                       is partially out of frame so its true length is unknown.

Thresholds are deliberately conservative and tunable via the function args.
"""
from __future__ import annotations

import numpy as np

# A fish that touches the frame edge is "cut off"; small margin absorbs the
# resolution downscaling used during segmentation.
DEFAULT_BORDER_MARGIN = 4
DEFAULT_MIN_AREA_FRAC = 0.0006   # below this -> tiny/no_fish
DEFAULT_MAX_AREA_FRAC = 0.55     # above this -> oversegmented


def assess(mask: np.ndarray, image_shape,
           min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
           max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
           border_margin: int = DEFAULT_BORDER_MARGIN):
    """Return (problematic: bool, flags: list[str], measurable: bool).

    ``measurable`` is False when the mask can't be measured at all (no fish /
    tiny); in that case the caller should not attempt skeletonization.
    """
    H, W = image_shape[:2]
    area = int((mask > 0).sum())
    flags: list[str] = []

    if area == 0:
        return True, ["no_fish"], False
    if area < min_area_frac * H * W:
        return True, ["tiny"], False

    if area > max_area_frac * H * W:
        flags.append("oversegmented")

    ys, xs = np.where(mask > 0)
    edges = []
    if xs.min() <= border_margin:
        edges.append("left")
    if xs.max() >= W - 1 - border_margin:
        edges.append("right")
    if ys.min() <= border_margin:
        edges.append("top")
    if ys.max() >= H - 1 - border_margin:
        edges.append("bottom")
    if edges:
        flags.append("cut_off:" + "+".join(edges))

    problematic = bool(flags)
    return problematic, flags, True
