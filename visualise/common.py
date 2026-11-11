"""Colours and small helpers shared by the plots, videos and overlays."""

from __future__ import annotations

import re

import matplotlib
import numpy as np

from common.datasets import Frame, iter_frame_images

GT_COLOUR = (0.1, 0.8, 0.2)     # green
PRED_COLOUR = (0.9, 0.1, 0.6)   # magenta
EXTRA_COLOUR = (1.0, 0.55, 0.0)  # orange: extra points, e.g. factor-graph boxes inserted by tracking3d
# tab10 without green: green is reserved for ground truth, as in the 2D plots.
TRACK_COLOURS = tuple(c for i, c in enumerate(matplotlib.colormaps["tab10"].colors) if i != 2)
ANCHOR_MARKERS = {"start": "P", "end": "X", "correction": "D"}   # annotated start / end, hand corrections
# Hand-corrected tracks (step 10): a solid dark green line, unlike the wide pale band of the ground truth.
CORRECTED_COLOUR = (0.0, 0.55, 0.1)


def to_bgr(colour_rgb) -> tuple[int, int, int]:
    """A matplotlib RGB colour (0-1) as an OpenCV BGR colour (0-255)."""
    return tuple(int(255 * c) for c in colour_rgb[::-1])


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s)


def first_image(frame: Frame) -> np.ndarray | None:
    try:
        return next(iter_frame_images([frame]))[1]
    except Exception:
        return None
