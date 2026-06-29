# Project context — fish length measurement

Persistent context for future sessions (read this first).

## Goal
Measure fish length from images and, going forward, **batch-process video frames**
with a convenient CLI. Length must be measured along the **curved body midline**
using keypoints, not a straight line. User: Amir.

## What exists
A Python package `fishmeasure/` + `measure.py` CLI. Pipeline:
segment (LAB color + GrabCut) -> skeleton (medial axis + geodesic longest path)
-> extend to tips -> resample K keypoints -> curved length -> overlay.
Modules: `segmentation.py`, `skeleton.py`, `measurement.py`, `visualize.py`,
`pipeline.py`. CLI handles single image, folder, or video (`--every N`).

## Key decisions / history
- **SAM was requested but could not run in the sandbox.** MobileSAM weights
  (40 MB) WERE successfully downloaded via `git clone
  https://github.com/ChaoningZhang/MobileSAM.git` (github.com is reachable;
  raw/release CDNs and huggingface are blocked by the proxy allowlist).
  Running them needs PyTorch, which could NOT be installed: the CPU wheel index
  (download.pytorch.org) is blocked, and the default PyPI torch (532 MB + multi-GB
  CUDA deps) exceeds the ~4 GB disk and the ~45 s per-command limit, and
  background installs don't persist across shell calls. So we used a classical
  GrabCut pipeline instead. To add SAM later: install torch, load
  `mobile_sam.pt`, prompt with the fish bbox, and swap into
  `segmentation.segment` (same signature: image_bgr -> 0/255 mask).
- GrabCut must run at reduced resolution (~1/3) — full 5312x2988 GrabCut was
  OOM-killed.
- Skeleton needs the mask Gaussian-smoothed first, else fins create many
  spurious branches. Main axis = geodesic longest path via double-BFS.

## Sample result (sanity check)
Image `vlcsnap-2026-06-28-10h30m40s479.png` (5312x2988):
curved length ≈ **1944 px** (9 keypoints), straight tip-to-tip ≈ 1917 px,
raw skeleton arc ≈ 2144 px. Use this to verify regressions.

## Measurements are in PIXELS
To get cm, calibrate with `--px-per-unit` using a known-size reference object.
The sample frame contains circular marker rings that could serve as references
(roadmap: auto-detect them).

## Target end product (agreed with Amir)
Video in -> cut to frames -> detect & measure ONE fish per frame -> write to
Excel. Plus a **review UI** (chosen: local web app, Flask + canvas) to browse a
video's overlay images, drag mis-placed keypoints, and have the length update.
Excel per fish must include: curved length, center-of-mass position, top-left
tip position, bottom-right tip position.

## Phase 1 — DONE
`fishmeasure/video.py` + `report.py` + updated `pipeline.py`/`measurement.py`/
`visualize.py`. Video/folder -> `output/<name>/{frames,overlays,measure.json}`
+ `output/<name>.xlsx`. measure.json is the canonical store; xlsx is regenerated
from it (so the editor can rewrite json then rebuild xlsx; avoids Excel locks).
Metrics per frame: curved_length_px, straight_tip_to_tip_px, com_x/com_y (mask
centroid), tip_tl_x/y (smaller x+y), tip_br_x/y (larger x+y). Tips ARE the first
& last keypoints, so editing them updates length + tip columns; CoM only changes
if the MASK is edited (out of scope for keypoint editor). `measurement.curved_metrics(keys, mask)`
recomputes all keypoint-derived metrics — reuse it in the editor backend.
Verified end-to-end on a half-res synthetic video: ~975 px = half of 1944 px. OK.

## Phase 1b — DONE: auto-flagging of bad frames
`fishmeasure/quality.py::assess(mask, image_shape)` returns
(problematic, flags, measurable). Flags: `no_fish` (empty), `tiny`,
`oversegmented` (>55% of frame), `cut_off:<edges>` (fish touches a border ->
partially out of frame). `measure_image` never raises now: bad frames return a
MeasureResult with problematic=True, NaN metrics, empty keypoints. measure.json
+ xlsx carry `problematic` + `flags`; problem rows are highlighted ORANGE
(F8CBAD) in xlsx; overlays get a red PROBLEM banner (orange REVIEW banner if
measurable but flagged). Summary stats skip problematic frames. CoM is NOT
user-editable per Amir. Thresholds in quality.py are tunable.

## Phase 2 — DONE: review/edit web app
`review.py` (Flask) + `web/index.html` (canvas, single file, no build) +
`fishmeasure/editing.py`. Run `python review.py --out output` -> open
http://127.0.0.1:5000. Routes: GET /api/videos, GET /api/videos/<v> (meta),
GET/POST /api/videos/<v>/frames/<frame>. On POST, `editing.recompute_record`
recomputes length/straight/tips from the edited keypoints (CoM PRESERVED, not
recomputed — per Amir), rewrites measure.json, regenerates the frame overlay
(draw_overlay with an empty mask so no tint, keypoints define the line), and
rebuilds the xlsx. edited=True -> yellow row. Optional clear_problem flag.
Frontend: drag yellow handles (pink=tips), live length, cyan +=CoM (read-only),
prev/next + "next problem", "Add keypoints" for no_fish frames, Ctrl/Cmd+S.
Verified: POSTing 5 keypoints gave length 282.84 px (== 4*hypot(50,50)),
edited=True propagated to json + xlsx, CoM unchanged. Needs `flask`.
DELETE /api/videos/<v>/frames/<frame> removes the frame from measure.json,
deletes its frames/ + overlays/ PNGs, and rebuilds xlsx (verified 5->4).
Frontend has a 🗑 Delete button (confirm) + toast feedback + busy button states
on every action. NOTE: frontend img.src must NOT append .png (route adds it) —
double .png caused black canvases; fixed.
Masks: processing now also saves `masks/<frame>.png` (the 0/255 segmentation
mask). GET /api/videos/<v>/frames/<frame>/mask serves it, recomputing via
`segment()` on demand for videos measured before this existed. Frontend "Show
mask" checkbox tints the mask green over the canvas (toggle to hide); tint built
client-side from the white mask. Verified: saved + on-demand recompute + empty
mask for no_fish all return valid PNGs.
Canvas zoom/pan: mouse wheel zooms about the cursor (zoom 1..8), drag-to-pan
when zoomed, double-click resets. Implemented via ctx.setTransform(zoom,..,panX,
panY) in fit-space; handle/line sizes multiplied by 1/zoom to stay constant on
screen; evtPt undoes pan+zoom+fit scale; clampPan keeps the image covering the
canvas. Mask-display race fixed earlier with a request token + maskForName guard.

## ENV GOTCHA (important)
The Windows project folder is mounted into the Linux sandbox, but host writes
(Edit/Write) sync to the Linux mount with LATENCY and can appear TRUNCATED there
for a while. Host files (via Read) are correct. To run/test, copy the project to
a pure-Linux dir (e.g. /tmp/fishproj) first, or you'll hit phantom SyntaxErrors.

## Phase 3 — DONE: YOLO detection-first pipeline
Per Amir: the pipeline now STARTS with YOLO detection. `fishmeasure/detector_yolo.py`
loads `model.pt` (ultralytics, lazy import so package works without torch),
`detect_fish()` returns the best whole-fish bbox (class name contains 'fish',
not a part keyword; else largest box). `measure_image` flow: detect -> `pad_bbox`
(+20%) -> `segmentation.segment_in_bbox` (crop, segment, place back) -> assess ->
skeleton -> measure. No detection -> `no_fish`. If `model.pt` missing -> one-time
warning + whole-frame fallback (so old behavior still works). CLI flags:
`--weights model.pt`, `--pad 0.2`, `--no-detect`. video.py saves masks/ and
threads weights/pad/use_detector. review.py recomputes on-demand masks via the
same detect->segment_in_bbox path (`--weights`). ultralytics in
requirements-yolo.txt (needs torch; NOT installable in THIS sandbox, so YOLO
code is compile-checked + logic-tested with a MOCK detect_fish only).
Verified (mock bbox): mask + CoM confined to padded box, length ~1950 px; and
fallback path still gives ~1944/halved. NOTE: choosing the fish class is
heuristic (FISH_KEYWORDS/PART_KEYWORDS) — adjust if model.names differ; head/tail
part centres can later seed tip keypoints (detect_parts already returns them).

## Phase 3b — DONE: mixed-directory input
`measure.py process_directory()` accepts a directory containing videos and/or
still images. Each video -> its own subfolder (frames every --every); all stills
-> one group named `<dir>_stills`. `--recursive` scans subfolders. Verified on a
dir with 1 video + 2 stills: outdir/mix(.xlsx) + outdir/testdir_stills(.xlsx),
each with frames/masks/overlays/measure.json.

## Phase 3c — DONE: show/hide YOLO box in UI
measure_image now stores the detection box on MeasureResult (`bbox` =
[x0,y0,x1,y1], `bbox_pad` = padded box segmented); video._result_to_record
writes `bbox` into each measure.json record (None if no detection/fallback).
editing.recompute_record leaves bbox untouched on save. Frontend "Show YOLO box"
checkbox draws a dashed orange rect (rec.bbox*scale, line width /zoom) inside the
zoom/pan transform; toast feedback; "No YOLO box" if absent. Verified bbox in
measure.json via mock detector; JS additions node-checked OK.

## Phase 3d — DONE: preprocessing (filters + val-style letterbox)
Per Amir, before detection+segmentation each frame is filtered:
`fishmeasure/preprocess.py::apply_filters` = CLAHE on L (clip 2.0, tiles 8x8) +
unsharp mask (addWeighted clahed*1.8 + GaussianBlur(55,55)*-0.8). NOTE: Amir's
snippet returned undefined `resized`; corrected to return `sharpened` (no resize
inside). YOLO input is square-letterboxed to 640 (`preprocess.letterbox`,
auto=False/scaleup=False, grey 114) replicating model.val(); `detect_boxes`
predicts on the in-memory letterboxed array (no temp-dir/PNG disk trick — array
is pixel-identical and faster) and maps boxes back via `scale_box_back`
(ratio+pad). measure_image gains `prefilter` (default True); detection AND
segmentation run on the filtered image. video.py/measure.py filter ONCE, save
the FILTERED frame to frames/ (+overlay on filtered), and call measure_image
with prefilter=False (no double filter). review.py recompute uses the already-
filtered saved frame. Filters preserve geometry so coords stay full-res.
Verified (synthetic + mock model): apply_filters keeps shape; letterbox 640
round-trips exactly; detect_boxes maps predict-space->full-res; pipeline mask
confined to bbox, CoM at fish centre; prefilter False==True result.
VAL_IMGSZ assumed 640.

## Phase 3e — DONE: SAM 2 box-prompt segmentation backend
Per Amir, GrabCut sometimes fails (grabs background AND misses parts) even with
good detection. `fishmeasure/segmenter_sam.py::segment_with_box(image, bbox)`
uses ultralytics SAM/SAM2 (lazy import) prompted with the YOLO box -> full-frame
0/255 mask (picks largest mask, resizes to frame if needed). Default weights:
`sam2_b.pt` (SAM 2 base, ~80 MB, ultralytics >= 8.3 auto-downloads on first use).
Supports sam2_t/s/b/l.pt and sam2.1_*.pt; MobileSAM (mobile_sam.pt) still works
as a lighter fallback. measure_image gains `segmenter` ('grabcut' default | 'sam')
+ `sam_weights`; on 'sam' it segments with the tight detection bbox (no pad), and
FALLS BACK to grabcut on any SAM error. Threaded through measure.py
(`--segmenter sam`, `--sam-weights`), video.py, review.py (on-demand recompute).
`sam_info(weights)` returns a human-readable label for the active model.
Verified with a MOCK SAM: full-frame placement, resize of differing-size mask,
pipeline len/CoM correct, and grabcut fallback.

## Roadmap (later)
1. Use detected head/tail part boxes to seed/lock tip keypoints.
2. Auto-calibration from reference object (marker rings in frame).
3. Per-fish tracking across video frames.
