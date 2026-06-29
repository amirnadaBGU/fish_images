"""Recompute a frame's metrics after its keypoints are edited in the review UI.

Only keypoint-derived metrics change (length, straight length, tip positions).
Center of mass comes from the segmentation mask and is intentionally preserved
(per project decision: CoM is not user-editable).
"""
from __future__ import annotations

import numpy as np

from .measurement import label_tips, polyline_length


def recompute_record(rec: dict, keypoints, meta: dict) -> dict:
    """Update ``rec`` in place from a new list of (x, y) keypoints.

    Returns the same dict. CoM (``com_x``/``com_y``) is left untouched.
    """
    keys = np.asarray(keypoints, dtype=np.float32)
    if keys.ndim != 2 or keys.shape[0] < 2 or keys.shape[1] != 2:
        raise ValueError("keypoints must be a list of >=2 [x, y] pairs")

    length = polyline_length(keys)
    straight = float(np.linalg.norm(keys[0] - keys[-1]))
    tl, br = label_tips(keys)

    rec["keypoints"] = [[round(float(x), 2), round(float(y), 2)] for x, y in keys]
    rec["num_keypoints"] = int(len(keys))
    rec["curved_length_px"] = round(length, 2)
    rec["straight_tip_to_tip_px"] = round(straight, 2)
    rec["bending_ratio"] = round(length / straight, 4) if straight > 0 else None
    rec["tip_tl_x"] = round(float(tl[0]), 2)
    rec["tip_tl_y"] = round(float(tl[1]), 2)
    rec["tip_br_x"] = round(float(br[0]), 2)
    rec["tip_br_y"] = round(float(br[1]), 2)
    rec["edited"] = True

    ppu = meta.get("px_per_unit")
    if ppu:
        unit = meta.get("unit", "unit")
        rec[f"curved_length_{unit}"] = round(length / ppu, 3)
    return rec
