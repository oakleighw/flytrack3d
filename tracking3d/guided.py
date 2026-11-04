"""Factor-graph tracking guided by hand-annotated start / end points.

Each annotated object becomes exactly one track; nothing else is tracked. Any mix of
annotations works: a start in one view, an end in another, both in all three, etc.

How the annotations are used:
    - Frame range. The object's track runs from its earliest START to its latest END
      (any view). A side without an annotation is found as in factor_graph.py:
      extend by extend_frames per round, then trim to the detections found.
    - Anchor factors. Every annotation is a reprojection factor at its frame, plain
      Gaussian with anchor_pixel_sigma (not Huber: it is trusted). Two views annotated
      at the same frame pin that frame in 3D; one view pins a ray, and the motion
      prior and detections fix the depth.
    - Seeding. The unguided factor-graph tracks (factor_graph.build_tracks, static
      false positives already dropped), then the raw triangulated tracklets, are
      candidates. For the start and for the end, the candidate passing closest to the
      annotation is used: within anchor_gate_px at the annotated frame, or starting /
      ending up to anchor_frame_window frames away, allowing for movement of up to
      anchor_max_px_per_frame meanwhile. Frames annotated in 2+ views are triangulated;
      single-view annotations start on their camera ray at the depth of the nearest
      seed point. Start and end seeds are joined by interpolation, which lets the
      track bridge stretches where the detector lost the object.
    - Association. As in factor_graph.py (nearest detection within
      association_gate_px of the projected track), except that detections used by an
      earlier object are skipped, so two objects never share a detection. Objects
      with both a start and an end go first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from .factor_graph import (MIN_MULTIVIEW_FRAMES, Track3D, _associate, _extend, _main_run, _solve, build_tracks,
                           gtsam_camera)
from .multiview import View
from .camera import triangulate_dlt
from .triangulate import Point3D


@dataclass
class Anchor:
    view: int                 # index into views
    frame: int                # aligned frame number
    kind: str                 # "start" or "end"
    xy: np.ndarray            # undistorted pixels
    raw: dict                 # the annotation as saved


@dataclass
class GuidedObject:
    id: int
    anchors: list[Anchor]
    notes: list[str] = field(default_factory=list)

    def _frames(self, kind: str) -> list[int]:
        return [a.frame for a in self.anchors if a.kind == kind]

    @property
    def start(self) -> int | None:
        f = self._frames("start")
        return min(f) if f else None

    @property
    def end(self) -> int | None:
        f = self._frames("end")
        return max(f) if f else None


def objects_from_annotations(annotations: list[dict], views: list[View]) -> tuple[list[GuidedObject], list[str]]:
    """Group the GUI's annotation dicts per object, in aligned frames and undistorted pixels."""
    index = {v.name: i for i, v in enumerate(views)}
    by_obj: dict[int, list[Anchor]] = {}
    warnings = []
    for a in annotations:
        v = index.get(a["view"])
        if v is None:
            warnings.append(f"annotation for unknown view '{a['view']}' ignored")
            continue
        cam = views[v].camera
        xy = cam.undistort(np.array([[a["x"], a["y"]]]))[0]
        by_obj.setdefault(int(a["object"]), []).append(
            Anchor(v, int(a["frame"]) + cam.frame_offset, a["kind"], xy, a))
    objects = [GuidedObject(k, anchors) for k, anchors in sorted(by_obj.items())]
    for o in objects:
        if o.start is not None and o.end is not None and o.start > o.end:
            raise SystemExit(f"Object #{o.id}: its START (frame {o.start}) is after its END (frame {o.end}). "
                             f"Fix the annotations and run again.")
    return objects, warnings


def _anchor_score(frames: np.ndarray, xyz: np.ndarray, a: Anchor, views: list[View], s: dict) -> float:
    """How well a candidate trajectory passes through an annotation; <= 1 is a match.
    Pixel distance in the annotated view at the annotated frame or, if the candidate starts
    or ends up to anchor_frame_window frames away, at its nearest end - allowing for the
    object moving up to anchor_max_px_per_frame in between."""
    if (a.kind == "start" and frames[-1] < a.frame) or (a.kind == "end" and frames[0] > a.frame):
        return np.inf   # a start's seed must continue after it, an end's seed must lead up to it
    if frames[0] <= a.frame <= frames[-1]:
        k, dt = int(np.abs(frames - a.frame).argmin()), 0
    else:
        k = 0 if a.frame < frames[0] else len(frames) - 1
        dt = abs(a.frame - frames[k])
        if dt > int(s["anchor_frame_window"]):
            return np.inf
    cam = views[a.view].camera
    if cam.depth(xyz[k])[0] <= 0:
        return np.inf
    d = np.linalg.norm(cam.project(xyz[k])[0] - a.xy)
    return float(d / (s["anchor_gate_px"] + s["anchor_max_px_per_frame"] * dt))


def _back_project(cam, xy: np.ndarray, depth: float) -> np.ndarray:
    """World point on the ray through undistorted pixel xy, at camera depth `depth`."""
    x_cam = depth * np.linalg.solve(cam.K, [xy[0], xy[1], 1.0])
    return cam.R.T @ (x_cam - cam.t)


def _seed(obj: GuidedObject, candidates: list[tuple[str, np.ndarray, np.ndarray]], views: list[View],
          s: dict) -> np.ndarray | None:
    """(K, 4) frame, x, y, z rows to initialise the object's trajectory, or None.
    candidates: (description, frames, xyz) - unguided tracks first, then raw tracklets."""
    lo = obj.start if obj.start is not None else -np.inf
    hi = obj.end if obj.end is not None else np.inf
    rows: list[np.ndarray] = []
    used = set()
    for kind in ("start", "end"):
        anchors = [a for a in obj.anchors if a.kind == kind]
        if not anchors:
            continue
        # Matching unguided tracks are preferred over raw tracklets, then the best score wins.
        score, i = min(((min(_anchor_score(f, x, a, views, s) for a in anchors), i)
                        for i, (_, f, x) in enumerate(candidates)), default=(np.inf, -1),
                       key=lambda c: (c[0] > 1, not candidates[c[1]][0].startswith("unguided"), c[0]))
        if score > 1:
            obj.notes.append(f"{kind}: no track or tracklet passes near the annotation")
            continue
        name, f, x = candidates[i]
        obj.notes.append(f"{kind}: seeded by {name}, frames {int(f[0])}-{int(f[-1])} (match score {score:.2f})")
        if i not in used:
            keep = (f >= lo) & (f <= hi)
            rows += [np.array([fr, *p]) for fr, p in zip(f[keep], x[keep])]
            used.add(i)

    # Frames annotated in 2+ views: triangulate them.
    for f in sorted({a.frame for a in obj.anchors}):
        same = {a.view: a for a in obj.anchors if a.frame == f}
        if len(same) >= 2:
            cams = [views[v].camera for v in same]
            X = triangulate_dlt(cams, [a.xy for a in same.values()])
            if all(c.depth(X)[0] > 0 for c in cams):
                rows.append(np.array([f, *X]))
    if not rows:
        return None

    # Frames annotated in one view only: on that camera's ray, at the depth of the nearest seed point.
    seeded = np.array(rows)
    covered = {int(r[0]) for r in rows}
    for a in obj.anchors:
        if a.frame not in covered:
            cam = views[a.view].camera
            near = seeded[np.abs(seeded[:, 0] - a.frame).argmin(), 1:4]
            rows.append(np.array([a.frame, *_back_project(cam, a.xy, cam.depth(near)[0])]))
            covered.add(a.frame)

    rows = np.array(sorted(rows, key=lambda r: r[0]))
    frames, inv = np.unique(rows[:, 0], return_inverse=True)
    xyz = np.array([rows[inv == i, 1:4].mean(axis=0) for i in range(len(frames))])
    seed = np.hstack([frames[:, None], xyz])
    if len(seed) == 1:
        seed = np.vstack([seed, seed + [1, 0, 0, 0]])
    return seed


def optimise_guided(obj: GuidedObject, seed: np.ndarray, views: list[View], gcams: list, s: dict,
                    frame_range: tuple[int, int], claimed: set, report=None) -> Track3D:
    annotated = [a.frame for a in obj.anchors]
    lo = obj.start if obj.start is not None else int(min(seed[0, 0], *annotated))
    hi = obj.end if obj.end is not None else int(max(seed[-1, 0], *annotated))
    frames = np.arange(lo, hi + 1)
    xyz = np.stack([np.interp(frames, seed[:, 0], seed[:, i]) for i in (1, 2, 3)], axis=1)
    n_ext, gap = int(s["extend_frames"]), int(s["link_max_frame_gap"])
    iterations = max(1, int(s["guided_iterations"]))
    obs: list = []
    sigma = np.full(len(frames), np.nan)
    for it in range(iterations):
        # Only extend sides that were not annotated.
        frames, xyz = _extend(frames, xyz, n_ext, frames[0] if obj.start is not None else frame_range[0],
                              frames[-1] if obj.end is not None else frame_range[1])
        obs = _associate(frames, xyz, views, s["association_gate_px"], claimed)
        k0, k1 = _main_run([o[0] for o in obs], gap) if obs else (0, len(frames) - 1)
        anchor_ks = [a.frame - frames[0] for a in obj.anchors]
        k0 = 0 if obj.start is not None else min(k0, *anchor_ks)
        k1 = len(frames) - 1 if obj.end is not None else max(k1, *anchor_ks)
        frames, xyz = frames[k0:k1 + 1], xyz[k0:k1 + 1]
        obs = [(k - k0, v, j) for k, v, j in obs if k0 <= k <= k1]
        anchors = [(a.frame - frames[0], a.view, a.xy) for a in obj.anchors]
        if report:
            report(f"round {it + 1}/{iterations}: solving {len(frames)} frames, {len(obs)} dets, "
                   f"{len(anchors)} annotations" + (" + sigma" if it == iterations - 1 else ""))
        xyz, sigma = _solve(frames, xyz, obs, views, gcams, s, with_sigma=(it == iterations - 1), anchors=anchors)

    n_views = np.bincount([k for k, _, _ in obs], minlength=len(frames)) if obs else np.zeros(len(frames), int)
    reproj = np.array([np.linalg.norm(views[v].camera.project(xyz[k])[0] - views[v].dets[int(frames[k])][j, :2])
                       for k, v, j in obs])
    return Track3D(id=obj.id, frames=frames, xyz=xyz, sigma=sigma, views=n_views,
                   observations=[(int(frames[k]), v, j) for k, v, j in obs], reproj_px=reproj)


def anchor_residuals(track: Track3D, obj: GuidedObject, views: list[View]) -> list[float]:
    """Pixel distance between each annotation and the track projected at its frame."""
    out = []
    for a in obj.anchors:
        k = int(a.frame - track.frames[0])
        out.append(float(np.linalg.norm(views[a.view].camera.project(track.xyz[k])[0] - a.xy)))
    return out


def build_guided_tracks(views: list[View], points: dict[int, list[Point3D]], annotations: list[dict],
                        s: dict) -> tuple[list[Track3D], list[GuidedObject], list[str]]:
    """One track per annotated object. Returns (tracks, objects, warnings)."""
    objects, warnings = objects_from_annotations(annotations, views)
    # Candidate seeds: the unguided factor-graph tracks (static false positives already dropped),
    # then the raw triangulated tracklets.
    # Steps 3-4 of the unguided pipeline, run only to produce candidate seeds.
    unguided, tracklets = build_tracks(views, points, {**s, "n_tracks": None})
    candidates = ([(f"unguided track {t.id}", t.frames, t.xyz) for t in unguided]
                  + [(f"a {len(t)}-frame tracklet", t[:, 0], t[:, 1:4]) for t in tracklets])
    gcams = [gtsam_camera(v.camera) for v in views]
    all_frames = [f for v in views for f in v.dets] + [a.frame for o in objects for a in o.anchors]
    frame_range = (min(all_frames), max(all_frames)) if all_frames else (0, 0)

    # Most constrained objects first, so they claim their detections before the others.
    order = sorted(objects, key=lambda o: (-((o.start is not None) + (o.end is not None)), -len(o.anchors), o.id))
    claimed: set = set()
    tracks = []
    bar = tqdm(order, desc="Guided factor graph", unit="object", dynamic_ncols=True)
    for obj in bar:
        # Starting guess from the clicks: nearest candidate, triangulated / ray-placed clicks, interpolated.
        seed = _seed(obj, candidates, views, s)
        if seed is None:
            warnings.append(f"object #{obj.id}: no 3D starting point - no track or tracklet passes near its "
                            f"annotations and no frame is annotated in 2+ views. Annotate it in a second view "
                            f"at the same frame, or nearer to where it is detected.")
            continue
        # Fresh factor graph with the clicks as trusted anchor factors, then claim its detections.
        tr = optimise_guided(obj, seed, views, gcams, s, frame_range, claimed,
                             report=lambda m: bar.set_postfix_str(f"object #{obj.id} | {m}"))
        claimed |= {(f, v, j) for f, v, j in tr.observations}
        if (tr.views >= 2).sum() < MIN_MULTIVIEW_FRAMES:
            obj.notes.append(f"only {(tr.views >= 2).sum()} frames seen by 2+ cameras: depth mostly from the "
                             f"motion prior and annotations")
        tracks.append(tr)
    tracks.sort(key=lambda t: t.id)
    return tracks, objects, warnings
