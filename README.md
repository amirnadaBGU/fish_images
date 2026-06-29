# fishmeasure

Measure a fish's length from a photo or video by segmenting it and measuring
along its **curved body midline** (not a straight line).

## Pipeline

0a. **Preprocess** (`fishmeasure/preprocess.py`) — every frame is enhanced with
   CLAHE (local contrast on the L channel) + an unsharp-mask sharpen before
   anything else. Filtering preserves pixel geometry, so all coordinates stay in
   full-resolution pixels. The saved `frames/` and overlays are the filtered
   image (detection, segmentation and review all see the same picture).
0b. **Detection** (`fishmeasure/detector_yolo.py`) — a YOLO model (`model.pt`)
   locates the fish. The image is square-**letterboxed to 640×640** (matching
   `model.val()`: aspect-ratio preserved, no upscale, grey padding); detections
   are mapped back to full-resolution coordinates. Segmentation then runs **only
   inside the fish bounding box + 20% padding**, so rings/glare elsewhere can't
   corrupt the mask or the centre of mass. No fish → the frame is flagged
   `no_fish`. If `model.pt` is absent it falls back to whole-frame segmentation.
1. **Segmentation** — two backends inside the detected box:
   - `grabcut` (default, `segmentation.py`): LAB colour separation + GrabCut.
   - `sam` (`segmenter_sam.py`, recommended when GrabCut struggles): SAM /
     MobileSAM prompted with the YOLO box. Because it segments the object
     itself, it fixes both classical failure modes — grabbing background and
     missing low-contrast parts (translucent fins). Select with `--segmenter sam`
     (needs ultralytics; `mobile_sam.pt` auto-downloads on first use). Falls
     back to GrabCut if SAM is unavailable.
2. **Skeleton** (`fishmeasure/skeleton.py`) — morphological thinning to the
   medial axis, then the **geodesic longest path** (double-BFS) gives the body
   axis from snout to tail, ignoring fin branches.
3. **Measurement** (`fishmeasure/measurement.py`) — extend the axis to the true
   mask boundary (snout + tail-fin tip), resample into *k* evenly spaced
   keypoints, and sum the segment lengths = curved length.
4. **Visualization** (`fishmeasure/visualize.py`) — overlay of mask, midline,
   keypoints and the measured length.

## Install

```bash
pip install -r requirements.txt
pip install -r requirements-yolo.txt   # YOLO detector backend (installs torch)
```

Put the detector weights at `model.pt` in the project root. The detector is the
default; use `--no-detect` to run the classical whole-frame pipeline, `--weights
path.pt` for a different model, and `--pad 0.2` to change the box padding.

## Usage

```bash
# Single image
python measure.py vlcsnap-2026-06-28-10h30m40s479.png

# A video, every 15th frame
python measure.py clip.mp4 --every 15 --out output

# Use SAM segmentation (better masks) instead of GrabCut
python measure.py clip.mp4 --segmenter sam --out output

# A directory with videos AND/OR still images (mixed) — process all together
python measure.py path/to/folder --every 15 --out output
#   each video -> its own subfolder (frames sampled every N)
#   all still images -> one "<folder>_stills" group
#   add --recursive to also scan subfolders

# Calibrate px -> cm (reference object 2 cm wide measured at 180 px -> 90 px/cm)
python measure.py clip.mp4 --px-per-unit 90 --unit cm

# More/fewer keypoints, or skip GrabCut for speed
python measure.py fish.png -k 15
python measure.py clip.mp4 --no-refine
```

### Outputs

Each video (or image folder) named `clip` produces:

```
output/clip/
    frames/      raw extracted frames
    overlays/    measurement overlays (mask, midline, keypoints, CoM)
    measure.json editable source of truth (keypoints + metrics per frame)
output/clip.xlsx report regenerated from measure.json
```

The Excel report has one row per fish/frame with: curved length, straight
length, **center of mass (x, y)**, **top-left tip (x, y)** and **bottom-right
tip (x, y)** — plus `length (<unit>)` when calibrated. `measure.json` is
canonical so the planned review UI can edit keypoints and the xlsx is rebuilt
from it (avoids Excel file-lock issues).

### Problematic frames (auto-flagged)

Frames are checked automatically and flagged for review instead of being
silently dropped:

- `no_fish` — nothing segmented.
- `tiny` — blob too small to trust.
- `oversegmented` — blob covers most of the frame.
- `cut_off:<edges>` — the fish touches an image border (partially out of frame).

Flagged rows get a `problematic` flag and a `flags` description in both
`measure.json` and the Excel report (highlighted orange), and their overlay
carries a red `PROBLEM`/`REVIEW` banner. Summary statistics ignore problematic
frames. Thresholds live in `fishmeasure/quality.py`.

## Review / edit web app

After measuring, launch the local review app to browse frames and correct
mis-placed keypoints:

```bash
python review.py --out output          # open http://127.0.0.1:5000
```

- Pick a video; the sidebar lists every frame (⚠ = problematic, ✓ = edited).
- Drag the yellow handles to fix the body line; pink handles are the head/tail
  tips. The curved length updates live.
- The cyan ✛ marks the center of mass (shown, not editable).
- "Next problem" jumps to the next flagged frame; arrow keys move between
  frames; Ctrl/Cmd+S saves.
- For a `no_fish` frame, "Add keypoints" drops a default line you can drag.
- Saving rewrites `measure.json`, regenerates that frame's overlay, and rebuilds
  the `.xlsx` (edited rows highlighted yellow). Tick "mark resolved" to clear a
  frame's problematic flag once you've fixed it.
- "🗑 Delete frame" removes that frame's image and its row from the Excel
  (with a confirmation).
- Every action gives feedback: buttons show a busy state while working and a
  toast confirms the result (e.g. "Saved ✓", "Deleted ✓").
- "Show mask" overlays the segmentation mask used for the measurement (green
  tint) so you can see exactly what the length was based on; untick to hide it.
  Masks are saved under `masks/` during processing; for videos measured before
  this feature the mask is recomputed on demand the first time it's shown.
- "Show YOLO box" draws the fish detection rectangle (dashed orange) that the
  segmentation was run inside; untick to hide. Stored per frame in measure.json.
- Mouse wheel zooms in/out centered on the cursor; drag to pan while zoomed;
  double-click resets the zoom. Keypoint handles stay the same size on screen.

## Programmatic use

```python
import cv2
from fishmeasure import measure_image

img = cv2.imread("fish.png")
res = measure_image(img, k=9, px_per_unit=90, unit="cm")
print(res.curved_length_px, res.curved_length_unit)
```

## Roadmap

- **Calibration**: auto-detect a reference object (the marker rings visible in
  the sample frame) to compute `px_per_unit` automatically.
- **SAM segmentation**: drop-in replacement for `segmentation.segment` using
  MobileSAM/SAM once PyTorch is available (see `CLAUDE.md`).
- **Tracking**: associate the same fish across video frames to track growth /
  motion over time.

## Notes

See `CLAUDE.md` for project history and the reasoning behind the classical
(non-SAM) segmentation choice.
