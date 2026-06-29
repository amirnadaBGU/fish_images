"""Video -> per-frame measurement, producing a self-contained output folder.

Layout produced for a video named ``clip.mp4`` under ``out_root``::

    out_root/clip/
        frames/    frame_000000.png ...   (raw extracted frames)
        overlays/  frame_000000.png ...   (measurement overlays)
        masks/     frame_000000.png ...   (segmentation masks, for the review UI)
        measure.json                      (source of truth, editable)
    out_root/clip.xlsx                     (report; regenerated from json)

``measure.json`` holds, per frame, the keypoints and metrics. The review UI
edits the keypoints there and the Excel report is regenerated from it, so the
JSON — not the xlsx — is the canonical store (avoids Excel file-lock issues).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from .pipeline import measure_image
from .preprocess import apply_filters
from .report import write_report
from .visualize import draw_overlay


def _result_to_record(name: str, frame_index: int, res) -> dict:
    def r(v, nd=2):
        # NaN -> None so measure.json stays valid JSON for the web editor
        return None if (isinstance(v, float) and v != v) else round(float(v), nd)

    rec = {
        "name": name,
        "frame_index": frame_index,
        "curved_length_px": r(res.curved_length_px),
        "straight_tip_to_tip_px": r(res.straight_tip_to_tip_px),
        "skeleton_arc_px": r(res.skeleton_arc_px),
        "num_keypoints": res.num_keypoints,
        "com_x": r(res.com_x),
        "com_y": r(res.com_y),
        "tip_tl_x": r(res.tip_tl_x),
        "tip_tl_y": r(res.tip_tl_y),
        "tip_br_x": r(res.tip_br_x),
        "tip_br_y": r(res.tip_br_y),
        "bending_ratio": round(res.curved_length_px / res.straight_tip_to_tip_px, 4)
                         if res.straight_tip_to_tip_px and res.straight_tip_to_tip_px == res.straight_tip_to_tip_px
                         else None,
        "keypoints": [[round(float(x), 2), round(float(y), 2)] for x, y in res.keypoints],
        "problematic": bool(res.problematic),
        "flags": ";".join(res.flags),
        "bbox": res.bbox,            # YOLO detection box [x0,y0,x1,y1] or None
        "edited": False,
    }
    if res.px_per_unit:
        rec["px_per_unit"] = res.px_per_unit
        rec["unit"] = res.unit
        rec[f"curved_length_{res.unit}"] = r(res.curved_length_unit, 3)
    return rec


def _metrics_for_overlay(res) -> dict:
    return {
        "curved_length_px": res.curved_length_px,
        "num_keypoints": res.num_keypoints,
        "com_x": res.com_x,
        "com_y": res.com_y,
    }


def make_overlay(frame, res, max_side: int = 1400):
    """Measured overlay for good frames, red 'PROBLEM' banner for flagged ones."""
    if res.measurable:
        ov = draw_overlay(frame, res.mask, res.keypoints, res.midline,
                          _metrics_for_overlay(res))
        if res.problematic:
            _banner(ov, "REVIEW: " + ";".join(res.flags), (0, 165, 255))
        return ov
    ov = frame.copy()
    _banner(ov, "PROBLEM: " + (";".join(res.flags) or "no_fish"), (0, 0, 255))
    h, w = ov.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        ov = cv2.resize(ov, (int(w * scale), int(h * scale)))
    return ov


def _banner(img, text, color):
    h, w = img.shape[:2]
    th = max(40, h // 18)
    cv2.rectangle(img, (0, 0), (w, th), color, -1)
    cv2.putText(img, text, (15, int(th * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                th / 45.0, (255, 255, 255), 2, cv2.LINE_AA)


def process_video(video_path, out_root="output", every: int = 15,
                  k: int = 9, refine: bool = True,
                  px_per_unit: Optional[float] = None, unit: str = "cm",
                  weights: str = "model.pt", use_detector: bool = True,
                  pad: float = 0.2, segmenter: str = "sam",
                  sam_weights: str = "sam2_b.pt", progress=print) -> Path:
    """Process a video and return the path to its output folder."""
    video_path = Path(video_path)
    out_dir = Path(out_root) / video_path.stem
    frames_dir = out_dir / "frames"
    overlays_dir = out_dir / "overlays"
    masks_dir = out_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    records = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            name = f"frame_{idx:06d}"
            proc = apply_filters(frame)   # filter once; save + measure on filtered
            cv2.imwrite(str(frames_dir / f"{name}.png"), proc)
            res = measure_image(proc, k=k, refine=refine,
                                px_per_unit=px_per_unit, unit=unit,
                                weights=weights, use_detector=use_detector, pad=pad,
                                prefilter=False, segmenter=segmenter, sam_weights=sam_weights)
            cv2.imwrite(str(overlays_dir / f"{name}.png"), make_overlay(proc, res))
            cv2.imwrite(str(masks_dir / f"{name}.png"), res.mask)
            rec = _result_to_record(name, idx, res)
            rec["timestamp_s"] = round(idx / fps, 3) if fps else None
            records.append(rec)
            if res.problematic:
                progress(f"  {name}: PROBLEM [{rec['flags']}]")
            else:
                progress(f"  {name}: {rec['curved_length_px']} px")
        idx += 1
    cap.release()

    meta = {
        "video": video_path.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "fps": fps,
        "every": every,
        "keypoints": k,
        "px_per_unit": px_per_unit,
        "unit": unit if px_per_unit else "px",
        "frames": records,
    }
    (out_dir / "measure.json").write_text(json.dumps(meta, indent=2))
    write_report(meta, Path(out_root) / f"{video_path.stem}.xlsx")
    progress(f"Wrote {len(records)} frame(s) -> {out_dir}")
    return out_dir
