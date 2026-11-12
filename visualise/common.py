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
# A corrected track counts as moved (drawn green) where it is this far from the source track (px).
MOVED_PX = 3.0


def _visible(camera, xyz: np.ndarray, size=None) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2) distorted pixel positions and (N,) mask: in front of the camera and, if size, in the image."""
    uv = camera.project(xyz, distort=True)
    ok = camera.depth(xyz) > 0
    if size:
        w, h = size
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, ok


def moved_in_view(track, source, camera, size=None, px: float = MOVED_PX) -> np.ndarray:
    """(T,) per frame of a corrected track: True where, in this camera, it is visible and either over
    px from the source track or the source track has no visible point there (the videos' rule).
    Every visible frame without a source track."""
    uv, ok = _visible(camera, track.xyz, size)
    if source is None:
        return ok
    suv, sok = _visible(camera, source.xyz, size)
    ref = {int(f): p for f, p, o in zip(source.frames, suv, sok) if o}
    far = np.array([int(f) not in ref or np.hypot(*(p - ref[int(f)])) > px for f, p in zip(track.frames, uv)],
                   dtype=bool).reshape(-1)
    return ok & far


def moved_frames(track, source, cameras, px: float = MOVED_PX) -> np.ndarray:
    """(T,) per frame of a corrected track: True where it is moved (moved_in_view) in any camera."""
    out = np.zeros(len(track.frames), dtype=bool)
    for cam in cameras:
        out |= moved_in_view(track, source, cam, getattr(cam, "image_size", None), px)
    return out


def moved_segments(mask: np.ndarray) -> list[slice]:
    """Runs of True in mask, each widened by one point either side so the green joins the track it left."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [slice(max(a - 1, 0), min(b + 1, len(mask))) for a, b in zip(idx[::2], idx[1::2])]


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
