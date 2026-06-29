"""Overlay rendering for QA: mask tint, midline, keypoints and length text."""
from __future__ import annotations

import cv2
import numpy as np


def draw_overlay(image_bgr, mask, keys, poly, metrics, max_side=1400):
    ov = image_bgr.copy()
    green = np.zeros_like(ov)
    green[:, :, 1] = 255
    sel = mask > 0
    ov[sel] = cv2.addWeighted(image_bgr, 0.5, green, 0.5, 0)[sel]

    for i in range(len(poly) - 1):
        cv2.line(ov, tuple(poly[i].astype(int)), tuple(poly[i + 1].astype(int)),
                 (0, 200, 255), 4)
    for i in range(len(keys) - 1):
        cv2.line(ov, tuple(keys[i].astype(int)), tuple(keys[i + 1].astype(int)),
                 (0, 0, 255), 8)
    for i, p in enumerate(keys):
        color = (255, 0, 255) if i in (0, len(keys) - 1) else (0, 255, 255)
        cv2.circle(ov, tuple(p.astype(int)), 18, color, -1)

    # center of mass marker
    if metrics.get("com_x") == metrics.get("com_x"):  # not NaN
        cx, cy = int(metrics["com_x"]), int(metrics["com_y"])
        cv2.drawMarker(ov, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 60, 6)
        cv2.circle(ov, (cx, cy), 12, (255, 255, 0), 3)

    txt = f"curved: {metrics['curved_length_px']:.0f} px  ({metrics['num_keypoints']} keypoints)"
    cv2.putText(ov, txt, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
    cv2.putText(ov, txt, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

    h, w = ov.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        ov = cv2.resize(ov, (int(w * scale), int(h * scale)))
    return ov
