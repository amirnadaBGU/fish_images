#!/usr/bin/env python3
"""Local web app to review and correct fish measurements.

Run:
    python review.py --out output            # then open http://127.0.0.1:5000

Browse each video's frames, drag mis-placed keypoints, and the curved length
updates live. Saving rewrites measure.json, regenerates the frame overlay, and
rebuilds the video's .xlsx report. Center of mass is shown but not editable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from fishmeasure.detector_yolo import detect_fish, pad_bbox, weights_available
from fishmeasure.editing import recompute_record
from fishmeasure.report import write_report
from fishmeasure.segmentation import segment, segment_in_bbox
from fishmeasure.visualize import draw_overlay

app = Flask(__name__)
OUT_ROOT = Path("output")
WEIGHTS = "model.pt"
SEGMENTER = "sam"
SAM_WEIGHTS = "sam2_b.pt"
WEB_DIR = Path(__file__).parent / "web"


def _recompute_mask(img):
    """Detection-first mask (matches measure.py); falls back to whole frame.

    The saved frame is already filtered, so no extra filtering here.
    """
    if weights_available(WEIGHTS):
        bbox = detect_fish(img, weights=WEIGHTS)
        if bbox is None:
            return np.zeros(img.shape[:2], np.uint8)
        if SEGMENTER == "sam":
            from fishmeasure.segmenter_sam import segment_with_box
            return segment_with_box(img, bbox, weights=SAM_WEIGHTS)
        return segment_in_bbox(img, pad_bbox(bbox, img.shape, 0.2))
    return segment(img)


def _videos():
    if not OUT_ROOT.exists():
        return []
    return sorted(p.name for p in OUT_ROOT.iterdir()
                  if p.is_dir() and (p / "measure.json").exists())


def _meta_path(video: str) -> Path:
    p = OUT_ROOT / video / "measure.json"
    if not p.exists():
        abort(404)
    return p


def _load_meta(video: str) -> dict:
    return json.loads(_meta_path(video).read_text())


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/videos")
def api_videos():
    return jsonify(_videos())


@app.route("/api/videos/<video>")
def api_video(video):
    return jsonify(_load_meta(video))


@app.route("/api/videos/<video>/frames/<frame>")
def api_frame_image(video, frame):
    # frame is the record 'name' (e.g. frame_000123); serve the raw frame png
    img = OUT_ROOT / video / "frames" / f"{frame}.png"
    if not img.exists():
        abort(404)
    return send_file(img, mimetype="image/png")


@app.route("/api/videos/<video>/frames/<frame>/mask")
def api_frame_mask(video, frame):
    """Serve the segmentation mask used for measurement.

    Saved by measure.py under masks/. For videos processed before masks were
    saved, recompute the mask on demand from the frame and cache it.
    """
    masks_dir = OUT_ROOT / video / "masks"
    mask_png = masks_dir / f"{frame}.png"
    if not mask_png.exists():
        frame_png = OUT_ROOT / video / "frames" / f"{frame}.png"
        if not frame_png.exists():
            abort(404)
        img = cv2.imread(str(frame_png))
        try:
            mask = _recompute_mask(img)
        except Exception:  # no fish / failure -> empty mask
            mask = np.zeros(img.shape[:2], np.uint8)
        masks_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_png), mask)
    return send_file(mask_png, mimetype="image/png")


@app.route("/api/videos/<video>/frames/<frame>", methods=["POST"])
def api_save(video, frame):
    meta = _load_meta(video)
    payload = request.get_json(force=True)
    keypoints = payload.get("keypoints")
    if not keypoints:
        abort(400, "no keypoints")

    rec = next((r for r in meta["frames"] if r["name"] == frame), None)
    if rec is None:
        abort(404, "frame not in measure.json")

    recompute_record(rec, keypoints, meta)
    if "com_x" in payload and "com_y" in payload:
        rec["com_x"] = round(float(payload["com_x"]), 2)
        rec["com_y"] = round(float(payload["com_y"]), 2)
    # optional: clear the problematic flag if the reviewer fixed it
    if payload.get("clear_problem"):
        rec["problematic"] = False
        rec["flags"] = ""

    # persist json
    _meta_path(video).write_text(json.dumps(meta, indent=2))

    # regenerate this frame's overlay (no mask tint; keypoints define the line)
    frame_img = OUT_ROOT / video / "frames" / f"{frame}.png"
    if frame_img.exists():
        img = cv2.imread(str(frame_img))
        keys = np.asarray(rec["keypoints"], dtype=np.float32)
        empty_mask = np.zeros(img.shape[:2], np.uint8)
        metrics = {"curved_length_px": rec["curved_length_px"],
                   "num_keypoints": rec["num_keypoints"],
                   "com_x": rec.get("com_x") if rec.get("com_x") is not None else float("nan"),
                   "com_y": rec.get("com_y") if rec.get("com_y") is not None else float("nan")}
        overlay = draw_overlay(img, empty_mask, keys, keys, metrics)
        cv2.imwrite(str(OUT_ROOT / video / "overlays" / f"{frame}.png"), overlay)

    # rebuild the workbook from the (now edited) json
    write_report(meta, OUT_ROOT / f"{video}.xlsx")

    return jsonify(rec)


@app.route("/api/videos/<video>/frames/<frame>", methods=["DELETE"])
def api_delete(video, frame):
    meta = _load_meta(video)
    before = len(meta["frames"])
    meta["frames"] = [r for r in meta["frames"] if r["name"] != frame]
    if len(meta["frames"]) == before:
        abort(404, "frame not in measure.json")

    # remove the frame + overlay images from disk
    for sub in ("frames", "overlays"):
        p = OUT_ROOT / video / sub / f"{frame}.png"
        if p.exists():
            p.unlink()

    _meta_path(video).write_text(json.dumps(meta, indent=2))
    write_report(meta, OUT_ROOT / f"{video}.xlsx")
    return jsonify({"deleted": frame, "remaining": len(meta["frames"])})


def main(argv=None):
    global OUT_ROOT, WEIGHTS, SEGMENTER, SAM_WEIGHTS
    ap = argparse.ArgumentParser(description="Fish measurement review/edit web app.")
    ap.add_argument("--out", default="output", help="Output root produced by measure.py.")
    ap.add_argument("--weights", default="model.pt", help="YOLO weights for on-demand mask recompute.")
    ap.add_argument("--segmenter", choices=["grabcut", "sam"], default="sam",
                    help="Segmentation backend for on-demand mask recompute.")
    ap.add_argument("--sam-weights", default="sam2_b.pt",
                    help="SAM/SAM2 weights when --segmenter sam (default: sam2_b.pt).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args(argv)
    OUT_ROOT = Path(args.out)
    WEIGHTS = args.weights
    SEGMENTER = args.segmenter
    SAM_WEIGHTS = args.sam_weights
    print(f"Serving review app for '{OUT_ROOT}' at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
