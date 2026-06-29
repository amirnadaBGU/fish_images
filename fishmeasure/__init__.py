"""fishmeasure — segment a fish and measure its length along a curved midline.

Pipeline: image -> segmentation (LAB color + GrabCut) -> skeleton (medial axis +
geodesic longest path) -> measurement (extend to tips, resample keypoints,
curved length) -> visualization.

See README.md for usage and CLAUDE.md for project context/history.
"""

from .pipeline import measure_image, MeasureResult

__all__ = ["measure_image", "MeasureResult"]
__version__ = "0.1.0"
