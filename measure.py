#!/usr/bin/env python3
"""Fish length measurement CLI.

Examples
--------
A video (every Nth frame) -> per-video folder + Excel report:
    python measure.py clip.mp4 --every 15 --out output

A folder of images or a single image:
    python measure.py path/to/folder --out output
    python measure.py fish.png --out output

Calibrate pixels -> cm (e.g. a 2 cm reference is 180 px wide -> 90 px/cm):
    python measure.py clip.mp4 --px-per-unit 90 --unit cm

For each video/folder you get output/<name>/ with frames/, overlays/ and
measure.json (the editable source of truth), plus output/<name>.xlsx with
length, center of mass and both tip positions per fish.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

from fishmeasure.pipeline import measure_image
from fishmeasure.preprocess import apply_filters
from fishmeasure.report import write_report
from fishmeasure.video import process_video, make_overlay, _result_to_record

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


def process_images(paths, name, out_root, args):
    """Measure a list of image paths as one 'set' -> folder + xlsx, like a video."""
    out_dir = Path(out_root) / name
    frames_dir = out_dir / "frames"
    overlays_dir = out_dir / "overlays"
    masks_dir = out_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [skip] {p.name}: unreadable")
            continue
        proc = apply_filters(img)   # filter once; save + measure on filtered
        res = measure_image(proc, k=args.keypoints, refine=not args.no_refine,
                            px_per_unit=args.px_per_unit, unit=args.unit,
                            weights=args.weights, use_detector=not args.no_detect,
                            pad=args.pad, prefilter=False,
                            segmenter=args.segmenter, sam_weights=args.sam_weights)
        cv2.imwrite(str(frames_dir / f"{p.stem}.png"), proc)
        cv2.imwrite(str(overlays_dir / f"{p.stem}.png"), make_overlay(proc, res))
        cv2.imwrite(str(masks_dir / f"{p.stem}.png"), res.mask)
        rec = _result_to_record(p.stem, i, res)
        records.append(rec)
        if res.problematic:
            print(f"  {p.stem}: PROBLEM [{rec['flags']}]")
        else:
            print(f"  {p.stem}: {rec['curved_length_px']} px")

    meta = {
        "video": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "fps": 0.0,
        "every": 1,
        "keypoints": args.keypoints,
        "px_per_unit": args.px_per_unit,
        "unit": args.unit if args.px_per_unit else "px",
        "frames": records,
    }
    (out_dir / "measure.json").write_text(json.dumps(meta, indent=2))
    write_report(meta, Path(out_root) / f"{name}.xlsx")
    print(f"Wrote {len(records)} item(s) -> {out_dir}")
    return out_dir


def process_directory(src, out_root, args):
    """Process a directory that may contain videos, still images, or both.

    Each video is processed into its own subfolder (frames sampled every
    ``--every`` frames). All still images are collected into one group
    ("<dir>_stills"). Use --recursive to descend into subfolders.
    """
    items = sorted(src.rglob("*") if args.recursive else src.iterdir())
    videos = [p for p in items if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    images = [p for p in items if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"Directory: {len(videos)} video(s) + {len(images)} still image(s)")
    if not videos and not images:
        print("No videos or images found.", file=sys.stderr)
        return 1

    for v in videos:
        print(f"Video: {v.name} (every {args.every} frame(s))")
        process_video(v, out_root=out_root, every=args.every, k=args.keypoints,
                      refine=not args.no_refine, px_per_unit=args.px_per_unit,
                      unit=args.unit, weights=args.weights,
                      use_detector=not args.no_detect, pad=args.pad,
                      segmenter=args.segmenter, sam_weights=args.sam_weights)
    if images:
        name = f"{src.name}_stills" if videos else src.name
        print(f"Stills: {len(images)} image(s) -> {name}")
        process_images(images, name, out_root, args)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure fish length along a curved midline.")
    ap.add_argument("input", help="Video, single image, or a directory with videos and/or images.")
    ap.add_argument("--out", default="output", help="Output root directory (default: output).")
    ap.add_argument("--keypoints", "-k", type=int, default=9, help="Number of midline keypoints.")
    ap.add_argument("--every", type=int, default=15, help="For video: process every Nth frame.")
    ap.add_argument("--px-per-unit", type=float, default=None,
                    help="Pixels per real unit, for calibration (e.g. px per cm).")
    ap.add_argument("--unit", default="cm", help="Unit label when --px-per-unit is set.")
    ap.add_argument("--no-refine", action="store_true", help="Skip GrabCut refinement (faster).")
    ap.add_argument("--weights", default="model.pt", help="YOLO detector weights (default: model.pt).")
    ap.add_argument("--pad", type=float, default=0.2, help="Bbox padding fraction for segmentation (default: 0.2).")
    ap.add_argument("--no-detect", action="store_true",
                    help="Disable YOLO detection; segment the whole frame.")
    ap.add_argument("--segmenter", choices=["grabcut", "sam"], default="sam",
                    help="Segmentation backend inside the box (default: sam).")
    ap.add_argument("--sam-weights", default="sam2_b.pt",
                    help="SAM/SAM2 weights when --segmenter sam (default: sam2_b.pt).")
    ap.add_argument("--recursive", action="store_true",
                    help="For a directory: also scan subfolders for videos/images.")
    args = ap.parse_args(argv)

    src = Path(args.input)
    if not src.exists():
        print(f"Input not found: {src}", file=sys.stderr)
        return 2

    if src.is_file() and src.suffix.lower() in VIDEO_EXTS:
        print(f"Video: {src.name} (every {args.every} frame(s))")
        process_video(src, out_root=args.out, every=args.every, k=args.keypoints,
                      refine=not args.no_refine, px_per_unit=args.px_per_unit,
                      unit=args.unit, weights=args.weights,
                      use_detector=not args.no_detect, pad=args.pad,
                      segmenter=args.segmenter, sam_weights=args.sam_weights)
    elif src.is_dir():
        return process_directory(src, args.out, args)
    elif src.suffix.lower() in IMAGE_EXTS:
        print(f"Image: {src.name}")
        process_images([src], src.stem, args.out, args)
    else:
        print(f"Unsupported input type: {src.suffix}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
