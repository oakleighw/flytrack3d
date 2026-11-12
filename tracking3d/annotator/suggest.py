"""Suggested START / END points for one view from the annotations in the other views.

For each object annotated (kind = start or end) in other views, at the same aligned
frame (frame numbers + each camera's frame_offset):
    - 2+ other views at that frame: triangulate, project into this view, then snap to
      the nearest detection there if it is within gate_px.
    - 1 other view: the detection in this view nearest to that point's epipolar line,
      within gate_px and in front of both cameras. No such detection = no suggestion
      (the depth along the line is unknown).
If an object is annotated at several frames, the frame seen by the most views wins
(ties: earliest for a start, latest for an end, as in tracking3d/guided.py).

Positions are raw (distorted) pixels and frames label frame numbers, as in the GUI.
"""

from __future__ import annotations

import cv2
import numpy as np

from tracking3d.camera import fundamental, triangulate_dlt


def distort(cam, xy: np.ndarray) -> np.ndarray:
    """(N, 2) ideal pinhole pixels -> (N, 2) raw pixels (inverse of Camera.undistort)."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    if not len(xy):
        return np.zeros((0, 2))
    rays = np.hstack([xy, np.ones((len(xy), 1))]) @ np.linalg.inv(cam.K).T
    zero = np.zeros(3)
    if cam.model == "fisheye":
        out = cv2.fisheye.projectPoints(rays.reshape(-1, 1, 3), zero, zero, cam.K, cam.dist)[0]
    else:
        out = cv2.projectPoints(rays, zero, zero, cam.K, cam.dist)[0]
    return out.reshape(-1, 2)


def epipolar_polyline(cam_a, xy_a_raw: tuple[float, float], cam_b, size: tuple[int, int],
                      n: int = 60) -> np.ndarray:
    """The epipolar line in view b of raw pixel xy_a_raw in view a, as raw pixels in b
    (a polyline: distortion bends it). size = (w, h) of view b."""
    xa = cam_a.undistort(np.array([xy_a_raw]))[0]
    a, b, c = fundamental(cam_a, cam_b) @ [xa[0], xa[1], 1.0]
    w, h = size
    if abs(b) >= abs(a):     # mostly horizontal: sample x
        xs = np.linspace(-0.1 * w, 1.1 * w, n)
        pts = np.stack([xs, -(a * xs + c) / b], axis=1)
    else:
        ys = np.linspace(-0.1 * h, 1.1 * h, n)
        pts = np.stack([-(b * ys + c) / a, ys], axis=1)
    return distort(cam_b, pts)


def epipolar_polylines(cam_a, xy_a_raw: np.ndarray, cam_b, size: tuple[int, int],
                       n: int = 60) -> np.ndarray:
    """epipolar_polyline for many points at once: (N, 2) raw pixels in view a ->
    (N, n, 2) raw pixels in view b (one undistort, one F and one distort call for all)."""
    xy = np.asarray(xy_a_raw, np.float64).reshape(-1, 2)
    if not len(xy):
        return np.zeros((0, n, 2))
    xa = cam_a.undistort(xy)
    lines = np.hstack([xa, np.ones((len(xa), 1))]) @ fundamental(cam_a, cam_b).T   # (N, 3): a, b, c
    a, b, c = lines[:, :1], lines[:, 1:2], lines[:, 2:]
    w, h = size
    xs = np.linspace(-0.1 * w, 1.1 * w, n)[None, :]
    ys = np.linspace(-0.1 * h, 1.1 * h, n)[None, :]
    horiz = np.abs(b) >= np.abs(a)                     # mostly horizontal: sample x
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.where(horiz, xs, -(b * ys + c) / a)
        py = np.where(horiz, -(a * xs + c) / b, ys)
    pts = np.stack([px, py], axis=2)                   # (N, n, 2)
    return distort(cam_b, pts.reshape(-1, 2)).reshape(len(xy), n, 2)


def aligned(frame: int, cam) -> int:
    return int(frame) + cam.frame_offset


def _pick_frame(anchors: list[tuple[int, dict]], kind: str) -> list[tuple[int, dict]]:
    """anchors: (view index, annotation). The ones at the best-supported aligned frame."""
    by_frame: dict[int, list] = {}
    for vi, a in anchors:
        by_frame.setdefault(a["_aligned"], []).append((vi, a))
    best = max(by_frame, key=lambda f: (len(by_frame[f]), -f if kind == "start" else f))
    return by_frame[best]


def suggest(views: list, target: int, annotations: list[dict], kind: str,
            gate_px: float) -> tuple[list[dict], list[str]]:
    """Suggested `kind` annotations for views[target]. views: annotator ViewInputs with .camera.
    Objects that already have a hand-placed `kind` point in the target view are skipped.
    Returns (new annotation dicts, one note per object)."""
    tv = views[target]
    tcam = tv.camera
    index = {v.name: i for i, v in enumerate(views)}
    out, notes = [], []
    objects = sorted({a["object"] for a in annotations if a["kind"] == kind})
    for obj in objects:
        if any(a["object"] == obj and a["kind"] == kind and a["view"] == tv.name and not a.get("suggested")
               for a in annotations):
            continue
        others = [(index[a["view"]], {**a, "_aligned": aligned(a["frame"], views[index[a["view"]]].camera)})
                  for a in annotations
                  if a["object"] == obj and a["kind"] == kind and a["view"] in index and a["view"] != tv.name
                  and views[index[a["view"]]].camera is not None]
        if not others:
            continue
        group = _pick_frame(others, kind)
        frame = group[0][1]["_aligned"] - tcam.frame_offset
        if frame not in tv.frame_numbers:
            notes.append(f"#{obj}: frame {frame} is not in this video")
            continue
        cams = [views[vi].camera for vi, _ in group]
        xys = [c.undistort(np.array([[a["x"], a["y"]]]))[0] for c, (_, a) in zip(cams, group)]
        d = tv.detections.get(frame)
        d_raw = d[:, :2] if d is not None and len(d) else np.zeros((0, 2))
        d_und = tcam.undistort(d_raw)
        names = "+".join(views[vi].name for vi, _ in group)

        if len(group) >= 2:
            X = triangulate_dlt(cams, xys)
            if tcam.depth(X)[0] <= 0 or any(c.depth(X)[0] <= 0 for c in cams):
                notes.append(f"#{obj}: triangulated point is behind a camera")
                continue
            p_und = tcam.project(X)[0]
            dist = np.linalg.norm(d_und - p_und, axis=1) if len(d_und) else np.zeros(0)
            if len(dist) and dist.min() <= gate_px:
                j = int(dist.argmin())
                xy, how = d_raw[j], f"triangulated from {names}, snapped to a detection {dist[j]:.0f} px away"
            else:
                xy, how = tcam.project(X, distort=True)[0], f"triangulated from {names}, no detection near"
        else:
            cam_a, xa = cams[0], xys[0]
            line = fundamental(cam_a, tcam) @ [xa[0], xa[1], 1.0]
            ok = []
            for j, xb in enumerate(d_und):
                dist = abs(line @ [xb[0], xb[1], 1.0]) / np.hypot(line[0], line[1])
                X = triangulate_dlt([cam_a, tcam], [xa, xb])
                if dist <= gate_px and cam_a.depth(X)[0] > 0 and tcam.depth(X)[0] > 0:
                    ok.append((dist, j))
            if not ok:
                notes.append(f"#{obj}: no detection on the epipolar line of {names} at frame {frame}")
                continue
            dist, j = min(ok)
            xy = d_raw[j]
            how = (f"epipolar line of {names}, detection {dist:.1f} px off"
                   + (f" ({len(ok)} candidates - check it)" if len(ok) > 1 else ""))
        out.append({"object": obj, "view": tv.name, "kind": kind, "frame": int(frame),
                    "x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2), "box": None, "suggested": True})
        notes.append(f"#{obj}: frame {frame}, {how}")
    return out, notes
