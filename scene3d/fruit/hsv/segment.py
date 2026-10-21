"""Find the red fake fruit in one camera view by HSV thresholding.

Pipeline per view:
    1. The view's median frame (scene3d/frames.py): flies vanish, the fruit stays.
    2. Red mask in HSV. Red wraps around hue 0, so it is a union of hue ranges
       (OpenCV hue is 0-179). The saturation floor keeps out the pale pink walls,
       the value floor the dark red-brown soil. If no fruit triangulates, locate.py
       retries with each of the lower fallback_min_values (dimly lit recordings).
    3. Open (removes soil speckle), then close (joins the fruit across the white
       pattern on its face); each outer contour, filled, is a candidate blob.
    4. Blobs that are too small or not round (area / enclosing-circle area) are
       dropped; the largest `max_candidates` are kept. Which blob is the fruit is
       decided across views by mesh/reconstruct.py (the one that triangulates).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_HSV = {
    "hue_ranges": [[0, 8], [170, 179]],   # OpenCV hue, 0-179
    "min_saturation": 120,
    "min_value": 60,
    "fallback_min_values": [40, 30],   # tried in turn only if min_value finds no fruit (dark videos)
    "open_px": 3,
    "close_px": 15,
    "min_area_px": 200,
    "min_fill": 0.5,          # blob area / its enclosing circle's area (1 = a disc)
    "max_candidates": 5,
}


@dataclass
class Blob:
    xy: np.ndarray            # (2,) centroid, raw (distorted) pixels
    radius_px: float          # enclosing-circle radius
    area: float
    fill: float
    contour: np.ndarray


def red_mask(bgr: np.ndarray, s: dict) -> np.ndarray:
    """uint8 0/1 mask of saturated red pixels, cleaned by an open and a close."""
    h, sat, val = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    hue = np.zeros(h.shape, bool)
    for lo, hi in s["hue_ranges"]:
        hue |= (h >= lo) & (h <= hi)
    mask = (hue & (sat >= s["min_saturation"]) & (val >= s["min_value"])).astype(np.uint8)
    if s["open_px"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((s["open_px"],) * 2, np.uint8))
    if s["close_px"]:
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s["close_px"],) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disc)
    return mask


def find_blobs(mask: np.ndarray, s: dict) -> list[Blob]:
    """Round red blobs in the mask, largest first (at most s["max_candidates"])."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < s["min_area_px"]:
            continue
        _, r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r)
        if fill < s["min_fill"]:
            continue
        m = cv2.moments(c)
        blobs.append(Blob(np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), r, area, fill, c))
    blobs.sort(key=lambda b: -b.area)
    return blobs[:s["max_candidates"]]


def save_debug_image(bgr: np.ndarray, mask: np.ndarray, blobs: list[Blob], chosen: Blob | None,
                     path: Path, label: str) -> None:
    """Median frame with the red mask tinted cyan, candidates outlined yellow, the chosen one green."""
    img = bgr.copy()
    img[mask > 0] = (0.5 * img[mask > 0] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
    for b in blobs:
        colour = (0, 255, 0) if b is chosen else (0, 255, 255)
        cv2.drawContours(img, [b.contour], -1, colour, 2)
        cv2.putText(img, f"r={b.radius_px:.0f}px fill={b.fill:.2f}", (int(b.xy[0] + b.radius_px + 6), int(b.xy[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
