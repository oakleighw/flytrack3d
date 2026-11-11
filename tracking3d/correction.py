"""Hand corrections of a past factor-graph result (step 10, main_human_correction.py).

A guided (or unguided) result usually follows its fly, but a track can jump to another
detection or drift where the detector lost the fly. In the correction annotator
(annotator/correction_tool.py) the user clicks where the fly really is: correction points,
in any view, at any frame, for any track. Here they are used exactly like the guided
start / end annotations (guided.py), and only the tracks that have corrections are solved
again; the others are kept as they are.

    Anchors      the source result's START / END annotations (guided results) plus the
                 corrections: plain Gaussian reprojection factors, anchor_pixel_sigma. A frame
                 corrected in 2+ views pins the fly in 3D; one view pins its camera ray.
    Seed         the source track, except where a correction says it is wrong: within
                 correction_seed_window frames of a correction where the source had a detection
                 (maybe the wrong one), and over the whole detection-free gap around a correction
                 where it had none (there the source is only the motion prior; at least
                 correction_seed_window frames either side). There the corrections take over
                 (triangulated where a frame is corrected in 2+ views, else on the camera ray at the
                 depth of the rest of the seed), interpolated in between. The seed matters beyond
                 the start: each frame has a weak prior towards it, which in a long gap without
                 detections is what pulls the solved track back to it between corrections.
    Frame range  as in guided.py: fixed by an annotated START / END, else extended / trimmed to
                 the detections found, always covering every correction. Corrections before a
                 track's START or after its END are left out (with a warning).
    Association  as in guided.py, nearest detection within association_gate_px, rounds of
                 guided_iterations. Corrected tracks claim their detections first; the tracks
                 kept as they are get theirs re-associated (needed by the evaluation and videos),
                 skipping those claimed.

Corrections are kept per source result in <run>/3d/corrections.json, so running the step
again loads them. Low confidence: a track frame whose 1-sigma 3D uncertainty is above
correction_low_sigma (world units).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .annotator.correction_tool import KIND, TrackView
from .camera import triangulate_dlt
from .factor_graph import Track3D, _associate, gtsam_camera
from .guided import GuidedObject, _back_project, objects_from_annotations, optimise_guided
from .multiview import View
from .past_results import PastResult

CORRECTIONS_FILE = "corrections.json"


# ----------------------------------------------------------------------------- saved corrections

def corrections_path(run_path: Path) -> Path:
    return Path(run_path) / "3d" / CORRECTIONS_FILE


def source_key(result: PastResult) -> str:
    """How a source result is named in corrections.json, e.g. yolo_s/factor_graph/guided_20261009-133218."""
    return f"{result.label}/{result.name}"


def read_corrections(run_path: Path) -> dict[str, dict]:
    """{source key: {"corrections": [...], "updated": ..., ...}} of a run ({} if it has none)."""
    path = corrections_path(run_path)
    return json.loads(path.read_text()).get("sources", {}) if path.is_file() else {}


def write_corrections(run_path: Path, key: str, corrections: list[dict], offsets: dict[str, int],
                      source_path: Path) -> Path:
    """Store a source result's corrections in <run>/3d/corrections.json (the other sources' are kept)."""
    path = corrections_path(run_path)
    data = json.loads(path.read_text()) if path.is_file() else {}
    data["_readme"] = ("Hand corrections of past factor-graph results (main_human_correction.py), per source result "
                       "(<detections>/factor_graph/<result>). frame = label frame number of the view, x / y raw "
                       "pixels, object = track id in that result.")
    data.setdefault("sources", {})[key] = {
        "updated": datetime.now().isoformat(timespec="seconds"), "source": str(source_path),
        "frame_offsets": offsets, "corrections": corrections}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


# ----------------------------------------------------------------------------- reading results

def load_result_tracks(csv: Path) -> list[Track3D]:
    """A tracks_3d.csv as Track3D (no observations: see reassociate)."""
    rows = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)   # track, frame, x, y, z, sigma, n_views
    out = []
    for tid in np.unique(rows[:, 0]):
        r = rows[rows[:, 0] == tid]
        out.append(Track3D(int(tid), r[:, 1].astype(int), np.ascontiguousarray(r[:, 2:5]), r[:, 5].copy(),
                           r[:, 6].astype(int)))   # contiguous: cv2.projectPoints refuses strided arrays
    return out


def fixed_annotations(result: PastResult) -> dict[int, list[dict]]:
    """The START / END annotations a guided result was made with, per track id ({} for unguided)."""
    objs = result._json("tracks.json").get("objects") or {}
    return {int(k): [a for a in v.get("annotations", []) if a.get("kind") in ("start", "end")]
            for k, v in objs.items()}


def corrected_results(results: list[PastResult], key: str) -> list[PastResult]:
    """The corrected results made from the source `key`, oldest first."""
    return [r for r in results if r.kind == "corrected" and r._json("params.json").get("source_key") == key]


# ----------------------------------------------------------------------------- display

def reassociate(track: Track3D, views: list[View], gate: float, claimed: set | None = None) -> Track3D:
    """The track with its detections found again: nearest within gate px of its projection, per
    frame and view (skipping `claimed`). Its frames and positions are not changed."""
    obs = _associate(track.frames, track.xyz, views, gate, claimed)
    n_views = np.bincount([k for k, _, _ in obs], minlength=len(track.frames))
    reproj = np.array([np.linalg.norm(views[v].camera.project(track.xyz[k])[0]
                                      - views[v].dets[int(track.frames[k])][j, :2]) for k, v, j in obs])
    return Track3D(track.id, track.frames, track.xyz, track.sigma, n_views,
                   [(int(track.frames[k]), v, j) for k, v, j in obs], reproj)


def track_views(track: Track3D, views: list[View], gate: float) -> dict[str, TrackView]:
    """The track as the correction annotator draws it in each view: projected position (raw
    pixels), married (a detection within gate px in that view) and sigma, per label frame."""
    married = {(k, v) for k, v, _ in _associate(track.frames, track.xyz, views, gate)}
    out = {}
    for v, view in enumerate(views):
        cam, (w, h) = view.camera, view.size
        uv = cam.project(track.xyz, distort=True)
        ok = (cam.depth(track.xyz) > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv[~ok] = np.nan
        out[view.name] = TrackView(track.frames - cam.frame_offset, uv,
                                   np.array([(k, v) in married for k in range(len(track.frames))]),
                                   np.asarray(track.sigma, float))
    return out


# ----------------------------------------------------------------------------- solving

def untrusted_frames(track: Track3D, cframes: np.ndarray, window: int) -> np.ndarray:
    """(K,) bool: the source track's frames that corrections at `cframes` override in the seed.

    A correction where the source track had a detection (it may be the wrong one) overrides it for
    `window` frames either side. One where it had none - the source there is only the motion prior
    bridging a gap - overrides the whole gap: back and forward to the nearest frames with a
    detection, and never less than `window` frames either side."""
    frames, seen = track.frames, np.asarray(track.views) > 0
    det = frames[seen]
    out = np.zeros(len(frames), bool)
    for f in cframes:
        lo, hi = f - window, f + window
        k = int(np.searchsorted(frames, f))
        if not (k < len(frames) and frames[k] == f and seen[k]):   # no detection at the correction: the gap
            i = int(np.searchsorted(det, f))
            lo = min(lo, det[i - 1] + 1 if i > 0 else frames[0])
            hi = max(hi, det[i] - 1 if i < len(det) else frames[-1])
        out |= (frames >= lo) & (frames <= hi)
    return out


def correction_seed(track: Track3D, obj: GuidedObject, views: list[View], window: int) -> np.ndarray:
    """(K, 4) frame, x, y, z: the starting trajectory of a corrected track (see the module doc)."""
    corr = [a for a in obj.anchors if a.kind == KIND]
    cframes = np.array(sorted({a.frame for a in corr}))
    near = untrusted_frames(track, cframes, window)
    rows = [np.array([f, *p]) for f, p in zip(track.frames[~near], track.xyz[~near])]
    single = []
    for f in cframes:   # frames corrected in 2+ views: triangulated
        same = {a.view: a for a in corr if a.frame == f}
        X = None
        if len(same) >= 2:
            cams = [views[v].camera for v in same]
            X = triangulate_dlt(cams, [a.xy for a in same.values()])
            if not all(c.depth(X)[0] > 0 for c in cams):
                X = None
        if X is not None:
            rows.append(np.array([f, *X]))
        else:
            single.append(next(iter(same.values())))
    base = np.array(sorted(rows, key=lambda r: r[0])) if rows else np.hstack([track.frames[:, None], track.xyz])
    for a in single:   # one view: on its camera ray, at the depth of the rest of the seed there
        cam = views[a.view].camera
        guess = np.array([np.interp(a.frame, base[:, 0], base[:, i]) for i in (1, 2, 3)])
        depth = cam.depth(guess)[0]
        if depth <= 0:
            depth = float(np.median(cam.depth(track.xyz)))
        rows.append(np.array([a.frame, *_back_project(cam, a.xy, depth)]))
    rows = np.array(sorted(rows, key=lambda r: r[0]))
    frames, inv = np.unique(rows[:, 0], return_inverse=True)
    xyz = np.array([rows[inv == i, 1:4].mean(axis=0) for i in range(len(frames))])
    seed = np.hstack([frames[:, None], xyz])
    return seed if len(seed) > 1 else np.vstack([seed, seed + [1, 0, 0, 0]])


def correct_tracks(source: list[Track3D], fixed: dict[int, list[dict]], corrections: list[dict],
                   views: list[View], s: dict) -> tuple[list[Track3D], dict[int, GuidedObject], list[str]]:
    """Re-solve every source track that has corrections; keep the others (re-associated).
    Returns (every track by id, {corrected track id: its GuidedObject (anchors, notes)}, warnings)."""
    by_id = {t.id: t for t in source}
    warnings = []
    unknown = sorted({c["object"] for c in corrections} - set(by_id))
    if unknown:
        warnings.append(f"corrections for track(s) {unknown} ignored: the source result has no such track")
    todo = sorted({c["object"] for c in corrections} & set(by_id))
    annotations = [a for tid in todo for a in fixed.get(tid, [])] + [c for c in corrections if c["object"] in todo]
    objects, w = objects_from_annotations(annotations, views)
    warnings += w
    objects = {o.id: o for o in objects}
    for o in objects.values():   # a fixed START / END bounds the track: corrections outside it can't apply
        out = [a for a in o.anchors if a.kind == KIND and ((o.start is not None and a.frame < o.start)
                                                          or (o.end is not None and a.frame > o.end))]
        if out:
            o.anchors = [a for a in o.anchors if a not in out]
            o.notes.append(f"{len(out)} correction(s) outside its START-END ({o.start}-{o.end}) left out")
            warnings.append(f"track #{o.id}: {len(out)} correction(s) before its START or after its END left out "
                            f"(change the START / END in step 8 to move them)")
    objects = {k: o for k, o in objects.items() if any(a.kind == KIND for a in o.anchors)}

    gcams = [gtsam_camera(v.camera) for v in views]
    all_frames = [f for v in views for f in v.dets] + [a.frame for o in objects.values() for a in o.anchors]
    frame_range = (min(all_frames), max(all_frames)) if all_frames else (0, 0)
    window = int(s["correction_seed_window"])
    claimed: set = set()
    out: dict[int, Track3D] = {}
    bar = tqdm(list(objects.values()), desc="Corrected factor graph", unit="track", dynamic_ncols=True)
    for obj in bar:
        n_corr = sum(a.kind == KIND for a in obj.anchors)
        bar.set_postfix_str(f"track #{obj.id} | {n_corr} correction(s) | seeding")
        seed = correction_seed(by_id[obj.id], obj, views, window)
        tr = optimise_guided(obj, seed, views, gcams, s, frame_range, claimed,
                             report=lambda m, obj=obj: bar.set_postfix_str(f"track #{obj.id} | {m}"))
        claimed |= set(tr.observations)
        obj.notes.append(f"{n_corr} correction(s); re-solved over frames {int(tr.frames[0])}-{int(tr.frames[-1])} "
                         f"(was {int(by_id[obj.id].frames[0])}-{int(by_id[obj.id].frames[-1])})")
        out[obj.id] = tr
    for t in tqdm([t for t in source if t.id not in out], desc="Re-associating kept tracks", unit="track",
                  dynamic_ncols=True, leave=False):
        out[t.id] = tr = reassociate(t, views, s["association_gate_px"], claimed)
        claimed |= set(tr.observations)
    return [out[k] for k in sorted(out)], objects, warnings
