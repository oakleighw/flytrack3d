"""Factor-graph tracks evaluated against the run's ground truth, in 3D and back in 2D.

3D: every track position is matched, frame by frame, to the ground-truth 3D points (the
    run's GT labels triangulated in 2+ views and linked - the green overlay) with the
    Hungarian method on distance, within eval_match_distance. Those points are
    triangulated themselves, so they carry a small error of their own.

2D: each track is projected back into every camera (full camera model, with distortion).
    In frames where the track has no detection in a view - the detector missed the object,
    or it fell outside the association gate - a box is inserted at the projection, with
    score backproj_score. Its size is the object's physical size (box size x depth / focal
    length, from the track's own detections in that view, interpolated in time; the median
    over all views if this view has none) projected at the track's depth in that frame, so
    it shrinks and grows as the object moves away and back. The run's 2D metrics are then
    recomputed (same settings as main_evaluate2d.py) for three prediction sets:
        detector         the saved predictions, as evaluated by main_evaluate2d.py
        + factor graph   plus the inserted boxes
        on tracks only   only the predictions a track used, plus the inserted boxes
                         (everything the factor graph rejected is dropped)
    P/R/F1 are given at every pr_iou_threshold; AP for the detector set only (the inserted boxes
    all share backproj_score, so an AP over them would follow that constant). How well the boxes
    sit is measured on the true positives: IoU and centre error (px) against the ground-truth box
    they matched, for detector and inserted boxes separately. Last, with no IoU or box size at all,
    point sets are matched one-to-one to the ground-truth box centres by distance (Hungarian,
    within eval_match_px): the detector's box centres, those plus the inserted ones, and the tracks
    themselves projected into every frame (split into frames where the track had a detection in that
    view and frames it bridged) - the 2D counterpart of the 3D evaluation, without its triangulated
    ground truth.

With visualise_backproj_graph or visualise_backproj_video, the inserted boxes are also saved as
YOLO labels, laid out like the run's predictions (one folder per input, one .txt per frame,
"class xc yc w h score", plus data.yaml): reprojection/labels/<input>/<sequence>/<frame>.txt.
visualise_backproj_video renders reprojection/<camera>.mp4 per camera: detector boxes in blue,
ground truth in green, every track's reprojected position as a red dot (also where the detector
has a box, to show how well the track follows it) and the inserted boxes in red.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.datasets import Dataset, frame_number, load_dataset, read_class_names
from detection.evaluate_models import eval_settings, evaluate
from detection.metrics import box_iou, match_boxes
from detection.predictions import PRED_DIR, load_predictions, save_predictions
from visualise.boxes.video import save_backproj_video
from visualise.charts.evaluation import save_2d_charts, save_3d_charts
from visualise.common import CORRECTED_COLOUR, MOVED_PX, safe_name
from visualise.metrics.tracking import format_2d, format_3d
from visualise.plots2d.track_plots import save_track_plots

from .factor_graph import Track3D
from .multiview import View

VARIANTS = ("detector", "+ factor graph", "on tracks only")


# ----------------------------------------------------------------------------- 3D

def _error_stats(err: np.ndarray) -> dict:
    """err: (N, 3) track - ground truth."""
    d = np.linalg.norm(err, axis=1)
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "rmse": float(np.sqrt((d ** 2).mean())), "mean": float(d.mean()),
            "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max()),
            "rmse_xyz": np.sqrt((err ** 2).mean(axis=0)).tolist()}


def evaluate_3d(tracks: list[Track3D], gt_tracks: list[np.ndarray], gate: float) -> dict:
    """gt_tracks: (K, 4) frame, x, y, z arrays (multiview.ground_truth_tracks)."""
    gt: dict[int, list] = {}
    for g, t in enumerate(gt_tracks, 1):
        for row in t:
            gt.setdefault(int(row[0]), []).append((g, row[1:4]))
    pred: dict[int, list] = {}
    for t in tracks:
        for f, p, n in zip(t.frames, t.xyz, t.views):
            pred.setdefault(int(f), []).append((t.id, p, int(n)))

    matches, in_gt_frames = [], 0   # (track id, gt id, error xyz, views of the track in that frame)
    for f, ps in pred.items():
        gs = gt.get(f)
        if not gs:
            continue
        in_gt_frames += len(ps)
        cost = np.linalg.norm(np.array([p[1] for p in ps])[:, None] - np.array([g[1] for g in gs])[None], axis=2)
        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] <= gate:
                matches.append((ps[r][0], gs[c][0], ps[r][1] - gs[c][1], ps[r][2]))

    def stats(ms):
        return _error_stats(np.array([m[2] for m in ms]).reshape(-1, 3))

    n_gt = sum(len(v) for v in gt.values())
    per_track = []
    for t in tracks:
        ms = [m for m in matches if m[0] == t.id]
        main, n_main = Counter(m[1] for m in ms).most_common(1)[0] if ms else (None, 0)
        per_track.append({"id": t.id, "frames": len(t.frames), "matched": len(ms), "gt_track": main,
                          "matched_to_gt_track": n_main, "error": stats(ms)})
    per_gt = []
    for g, t in enumerate(gt_tracks, 1):
        ms = [m for m in matches if m[1] == g]
        per_gt.append({"id": g, "start_frame": int(t[0, 0]), "end_frame": int(t[-1, 0]), "points": len(t),
                       "matched": len(ms), "tracks": sorted({m[0] for m in ms})})
    return {
        "match_distance": gate, "gt_tracks": len(gt_tracks), "gt_points": n_gt,
        "track_points": sum(len(t.frames) for t in tracks), "track_points_in_gt_frames": in_gt_frames,
        "matched": len(matches),
        "recall": len(matches) / n_gt if n_gt else None,
        "precision": len(matches) / in_gt_frames if in_gt_frames else None,
        "error": stats(matches),
        "error_by_views": {"0-1 views": stats([m for m in matches if m[3] <= 1]),
                           "2 views": stats([m for m in matches if m[3] == 2]),
                           "3 views": stats([m for m in matches if m[3] >= 3])},
        "per_track": per_track, "per_gt_track": per_gt,
    }


# ----------------------------------------------------------------------------- 2D

@dataclass
class ViewData:
    ds: Dataset
    preds: dict[str, np.ndarray]     # key -> (M, 6) x1, y1, x2, y2, score, class (raw pixels)
    keys: dict[int, str]             # aligned frame number -> dataset frame key
    conf: float                      # confidence for P/R (as in main_evaluate2d.py)
    vis_conf: float                  # boxes drawn in the videos / track plots (as in main_evaluate2d.py)


def load_view_data(session, view: View, es: dict, vis: dict) -> ViewData:
    meta = session.run.metadata["inputs"][view.input_name]
    ds = load_dataset(meta["dataset"]["source"])
    preds = load_predictions(session.run.path / view.input_name / session.model / PRED_DIR, ds)
    keys = {frame_number(f, i) + view.camera.frame_offset: f.key for i, f in enumerate(ds.frames)}
    conf = es["pr_conf_threshold"]
    if conf is None:
        conf = meta["models"][session.model]["model_config"].get("confidence_threshold", 0.0)
    vis_conf = vis.get("conf_threshold") if vis.get("conf_threshold") is not None else conf
    return ViewData(ds, preds, keys, float(conf), float(vis_conf))


def backproject(tracks: list[Track3D], views: list[View], data: list[ViewData],
                score: float) -> tuple[list[dict], list[dict]]:
    """Per view: ({key: (M, 6) inserted boxes}, {key: bool mask of the predictions a track used})."""
    inserted: list[dict] = [{} for _ in views]
    on_track: list[dict] = [{} for _ in views]
    for t in tracks:
        observed, sizes = set(), [[] for _ in views]   # sizes: (frame, physical w, physical h, class)
        for f, v, j in t.observations:
            key = data[v].keys.get(f)
            boxes = data[v].preds.get(key, np.zeros((0, 6)))
            if not len(boxes):
                continue
            # The track's detection is a box centre; find its box in the saved predictions.
            dist = np.linalg.norm((boxes[:, :2] + boxes[:, 2:4]) / 2 - views[v].raw[f][j, :2], axis=1)
            i = int(dist.argmin())
            if dist[i] > 0.5:
                continue
            observed.add((v, f))
            on_track[v].setdefault(key, np.zeros(len(boxes), bool))[i] = True
            cam, b = views[v].camera, boxes[i]
            z = cam.depth(t.xyz[f - t.frames[0]])[0]
            sizes[v].append((f, (b[2] - b[0]) * z / cam.K[0, 0], (b[3] - b[1]) * z / cam.K[1, 1], b[5]))
        everywhere = np.array([s for per_view in sizes for s in per_view]).reshape(-1, 4)
        if not len(everywhere):
            continue   # no detection to take the object's size from
        cls = int(np.bincount(everywhere[:, 3].astype(int)).argmax())
        for v, view in enumerate(views):
            cam, (w, h) = view.camera, view.size
            s = np.array(sorted(sizes[v])).reshape(-1, 4)
            depth, uv = cam.depth(t.xyz), cam.project(t.xyz, distort=True)
            for k, f in enumerate(t.frames):
                f = int(f)
                key = data[v].keys.get(f)
                x, y = uv[k]
                if key is None or (v, f) in observed or depth[k] <= 0 or not (0 <= x < w and 0 <= y < h):
                    continue
                if len(s):
                    pw, ph = np.interp(f, s[:, 0], s[:, 1]), np.interp(f, s[:, 0], s[:, 2])
                else:
                    pw, ph = np.median(everywhere[:, 1]), np.median(everywhere[:, 2])
                bw, bh = pw * cam.K[0, 0] / depth[k], ph * cam.K[1, 1] / depth[k]
                inserted[v].setdefault(key, []).append([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2, score, cls])
    return [{k: np.array(b) for k, b in ins.items()} for ins in inserted], on_track


def project_tracks(tracks: list[Track3D], views: list[View], data: list[ViewData],
                   with_observed: bool = False) -> list[dict]:
    """Per view: {frame key: (N, 2) x, y} of every track projected into the image (full camera
    model, with distortion), in every frame it is in front of the camera and inside the image.
    with_observed: (N, 3) x, y, 1 if the track has a detection in that view in that frame else 0."""
    points: list[dict] = [{} for _ in views]
    for t in tracks:
        observed = {(f, v) for f, v, _ in t.observations}
        for v, view in enumerate(views):
            (w, h), cam = view.size, view.camera
            depth, uv = cam.depth(t.xyz), cam.project(t.xyz, distort=True)
            for f, d, (x, y) in zip(t.frames, depth, uv):
                key = data[v].keys.get(int(f))
                if key is not None and d > 0 and 0 <= x < w and 0 <= y < h:
                    points[v].setdefault(key, []).append((x, y, float((int(f), v) in observed)))
    return [{k: np.array(p)[:, :3 if with_observed else 2] for k, p in pts.items()} for pts in points]


def _variants(d: ViewData, ins: dict, on: dict) -> dict[str, tuple[dict, dict]]:
    """{variant: ({key: (M, 6) boxes}, {key: (M,) bool, True = inserted by the factor graph})}."""
    empty = np.zeros((0, 6))
    plus, only, plus_ins, only_ins = {}, {}, {}, {}
    for k, p in d.preds.items():
        add, kept = ins.get(k, empty), p[on.get(k, np.zeros(len(p), bool))]
        plus[k], only[k] = np.vstack([p, add]), np.vstack([kept, add])
        plus_ins[k] = np.r_[np.zeros(len(p), bool), np.ones(len(add), bool)]
        only_ins[k] = np.r_[np.zeros(len(kept), bool), np.ones(len(add), bool)]
    detector = (d.preds, {k: np.zeros(len(p), bool) for k, p in d.preds.items()})
    return dict(zip(VARIANTS, (detector, (plus, plus_ins), (only, only_ins))))


def _loc_stats(iou: list[float], centre: list[float]) -> dict:
    if not iou:
        return {"tp": 0}
    iou, centre = np.array(iou), np.array(centre)
    return {"tp": int(len(iou)), "iou_mean": float(iou.mean()), "iou_median": float(np.median(iou)),
            "centre_px_rmse": float(np.sqrt((centre ** 2).mean())), "centre_px_mean": float(centre.mean()),
            "centre_px_median": float(np.median(centre)), "centre_px_p95": float(np.percentile(centre, 95)),
            "centre_px_max": float(centre.max())}


def localisation(gts: dict, preds: dict, is_inserted: dict, conf: float, es: dict) -> dict:
    """{IoU@t: {"all" / "detector" / "inserted": stats}} over the true positives at conf (matched as
    for P/R): IoU with their ground-truth box and centre distance to it (raw px). "all" = every
    true positive of the set; "detector" / "inserted" = those from the detector / the factor graph."""
    out = {}
    for t in es["pr_iou_thresholds"]:
        acc = {False: ([], []), True: ([], [])}
        for k, g in gts.items():
            p = preds.get(k, np.zeros((0, 6)))
            for di, gi, iou in match_boxes(g, p, t, es["max_dets_per_image"]):
                if p[di, 4] < conf:
                    continue   # below the P/R confidence: not a true positive there
                dc, gc = (p[di, :2] + p[di, 2:4]) / 2, (g[gi, 1:3] + g[gi, 3:5]) / 2
                acc[bool(is_inserted[k][di])][0].append(iou)
                acc[bool(is_inserted[k][di])][1].append(float(np.linalg.norm(dc - gc)))
        out[f"IoU@{t:g}"] = {"all": _loc_stats(acc[False][0] + acc[True][0], acc[False][1] + acc[True][1]),
                             "detector": _loc_stats(*acc[False]), "inserted": _loc_stats(*acc[True])}
    return out


def _scored(name: str, res: dict, gts: dict, preds: dict, is_inserted: dict, conf: float, es: dict) -> dict:
    """A variant's metrics, box count and localisation. AP is kept for the detector alone: the
    inserted boxes all carry the fixed backproj_score, so where they rank (and so AP) is set by
    that constant, not by the tracks. The factor-graph variants are compared by P/R/F1."""
    if name != VARIANTS[0]:
        res = {k: v for k, v in res.items() if k not in ("average_precision", "per_class")}
    return {**res, "boxes": int(sum(map(len, preds.values()))),
            "localisation": localisation(gts, preds, is_inserted, conf, es)}


def _px_stats(d: np.ndarray) -> dict:
    if not len(d):
        return {}
    return {"rmse": float(np.sqrt((d ** 2).mean())), "mean": float(d.mean()), "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)), "max": float(d.max())}


def _size_stats(iou: list[float], ratio: list[float]) -> dict:
    """ratio: predicted / ground-truth box scale (sqrt(w * h)) of each matched box."""
    if not iou:
        return {}
    iou, ratio = np.array(iou), np.array(ratio)
    err = np.abs(ratio - 1)
    return {"iou_mean": float(iou.mean()), "size_err_median": float(np.median(err)),
            "size_err_p95": float(np.percentile(err, 95)), "size_bias_median": float(np.median(ratio - 1))}


def _centre_match(gts: dict, points: dict, tags: dict, gate: float) -> dict:
    """Match each frame's points one-to-one to its ground-truth box centres (Hungarian on distance,
    within gate px; class-agnostic). points: (N, 2) x, y, or (N, 4) xyxy boxes (matched by their
    centre; their size is then scored too). {"gt": n, "all" / tag: {"points", "matched",
    "precision", distance stats of the matched [, size stats]}, "all" also "recall"}."""
    dist = {tag: [] for tag in ("all", *sorted({t for v in tags.values() for t in v}))}
    size = {tag: ([], []) for tag in dist}   # (IoU, predicted / true box scale) of matched boxes
    count = Counter()
    n_gt = 0
    for k, g in gts.items():
        gc = (g[:, 1:3] + g[:, 3:5]) / 2
        p, tg = points.get(k, np.zeros((0, 2))), tags.get(k, np.zeros(0, str))
        n_gt += len(gc)
        count.update(tg.tolist())
        count["all"] += len(p)
        if not len(gc) or not len(p):
            continue
        boxes = p.shape[1] == 4
        pc = (p[:, :2] + p[:, 2:4]) / 2 if boxes else p[:, :2]
        cost = np.linalg.norm(pc[:, None] - gc[None], axis=2)
        if boxes:
            ious = box_iou(p, g[:, 1:5])
            scale_p = np.sqrt(np.prod(p[:, 2:4] - p[:, :2], axis=1))
            scale_g = np.sqrt(np.prod(g[:, 3:5] - g[:, 1:3], axis=1))
        for r, c in zip(*linear_sum_assignment(np.where(cost <= gate, cost, 1e9))):
            if cost[r, c] <= gate:
                for tag in ("all", tg[r]):
                    dist[tag].append(cost[r, c])
                    if boxes:
                        size[tag][0].append(ious[r, c])
                        size[tag][1].append(scale_p[r] / max(scale_g[c], 1e-9))
    out = {"gt": n_gt}
    for tag, d in dist.items():
        d = np.array(d)
        out[tag] = {"points": count[tag], "matched": int(len(d)),
                    "precision": len(d) / count[tag] if count[tag] else None, **_px_stats(d), **_size_stats(*size[tag])}
    out["all"]["recall"] = out["all"]["matched"] / n_gt if n_gt else None
    return out


def centre_errors(views: list[View], data: list[ViewData], inserted: list[dict], on_track: list[dict],
                  tracks: list[Track3D], gate: float) -> dict:
    """{view name or "ALL (pooled)": {point set: _centre_match}} for "detector" (box centres at the P/R
    confidence), "+ factor graph" (plus the inserted box centres; tags detector / inserted),
    "tracked detections" (the raw detector boxes the tracks were built from) and "track
    reprojection" (every track in every frame; tags observed / bridged). The last two cover the
    same objects and frames, so they show what the factor graph's smoothing does to the position."""
    proj = project_tracks(tracks, views, data, with_observed=True)
    sets = ("detector", "+ factor graph", "tracked detections", "track reprojection")
    out, pooled_gt, pooled = {}, {}, {s: ({}, {}) for s in sets}
    for view, d, ins, on, pr in zip(views, data, inserted, on_track, proj):
        gts = {f.key: f.gt for f in d.ds.frames}
        pooled_gt.update(gts)
        per_set = {s: ({}, {}) for s in sets}
        for k in gts:
            p = d.preds.get(k, np.zeros((0, 6)))
            det = p[p[:, 4] >= d.conf, :4]
            add = ins.get(k, np.zeros((0, 6)))[:, :4]
            tp = pr.get(k, np.zeros((0, 3)))
            per_set["detector"][0][k], per_set["detector"][1][k] = det, np.full(len(det), "detector")
            per_set["+ factor graph"][0][k] = np.vstack([det, add])
            per_set["+ factor graph"][1][k] = np.r_[np.full(len(det), "detector"), np.full(len(add), "inserted")]
            used = p[on.get(k, np.zeros(len(p), bool)), :4]
            per_set["tracked detections"][0][k] = used
            per_set["tracked detections"][1][k] = np.full(len(used), "detector")
            per_set["track reprojection"][0][k] = tp[:, :2]
            per_set["track reprojection"][1][k] = np.where(tp[:, 2] > 0, "observed", "bridged")
        out[view.name] = {}
        for s, (pts, tags) in per_set.items():
            out[view.name][s] = _centre_match(gts, pts, tags, gate)
            pooled[s][0].update(pts)
            pooled[s][1].update(tags)
    out["ALL (pooled)"] = {s: _centre_match(pooled_gt, pts, tags, gate) for s, (pts, tags) in pooled.items()}
    return out


def evaluate_2d(views: list[View], data: list[ViewData], inserted: list[dict], on_track: list[dict],
                es: dict) -> dict:
    """{view name or "ALL (pooled)": {variant: metrics + box counts + localisation}} (AP for the
    detector variant only)."""
    out, pooled_gt, pooled = {}, {}, {v: ({}, {}) for v in VARIANTS}
    for view, d, ins, on in zip(views, data, inserted, on_track):
        gts = {f.key: f.gt for f in d.ds.frames}
        pooled_gt.update(gts)
        out[view.name] = {}
        for name, (preds, is_ins) in _variants(d, ins, on).items():
            pooled[name][0].update(preds)
            pooled[name][1].update(is_ins)
            out[view.name][name] = _scored(name, evaluate(gts, preds, d.conf, es), gts, preds, is_ins, d.conf, es)
    conf = data[0].conf
    out["ALL (pooled)"] = {name: _scored(name, evaluate(pooled_gt, preds, conf, es), pooled_gt, preds, is_ins, conf, es)
                           for name, (preds, is_ins) in pooled.items()}
    return out


# ----------------------------------------------------------------------------- entry

HAND_POINT_HOLD = 8   # frames either side a hand-placed correction point stays drawn in the videos
# MOVED_PX (visualise.common): a corrected track is drawn green where it is this far from the source track (px, that view)


def _split_moved(corrected: list[Track3D], source: dict[int, Track3D], views: list[View],
                 data: list[ViewData]) -> tuple[list[dict], list[dict]]:
    """Per view, the corrected tracks' points split into (unchanged, moved): moved where the
    correction put the track over MOVED_PX from the source track in that view, or where the source
    track has no point there (outside its frames, behind the camera or out of the image)."""
    same: list[dict] = [{} for _ in views]
    moved: list[dict] = [{} for _ in views]
    for t in corrected:
        src = source.get(t.id)
        for v, view in enumerate(views):
            (w, h), cam = view.size, view.camera
            uv, depth = cam.project(t.xyz, distort=True), cam.depth(t.xyz)
            ref = {}
            if src is not None:
                for f, d, p in zip(src.frames, cam.depth(src.xyz), cam.project(src.xyz, distort=True)):
                    if d > 0 and 0 <= p[0] < w and 0 <= p[1] < h:
                        ref[int(f)] = p
            for f, d, (x, y) in zip(t.frames, depth, uv):
                key = data[v].keys.get(int(f))
                if key is None or d <= 0 or not (0 <= x < w and 0 <= y < h):
                    continue
                r = ref.get(int(f))
                out = moved if r is None or np.hypot(x - r[0], y - r[1]) > MOVED_PX else same
                out[v].setdefault(key, []).append((x, y))
    as_arrays = lambda pts: [{k: np.array(p) for k, p in d.items()} for d in pts]
    return as_arrays(same), as_arrays(moved)


def _hand_points(hand: dict[int, list[tuple[float, float]]], d: ViewData) -> dict[str, np.ndarray]:
    """{aligned frame: [(x, y) raw px]} -> {frame key: (N, 3) x, y, exact}, held HAND_POINT_HOLD frames."""
    out: dict[str, list] = {}
    for f, pts in hand.items():
        for df in range(-HAND_POINT_HOLD, HAND_POINT_HOLD + 1):
            key = d.keys.get(int(f) + df)
            if key is not None:
                out.setdefault(key, []).extend((x, y, float(df == 0)) for x, y in pts)
    return {k: np.array(v) for k, v in out.items()}


def _correction_layers(corrected: list[Track3D], source_tracks: list[Track3D] | None, moved, hand_points,
                       views: list[View], data: list[ViewData]) -> list[list]:
    """Per view, step 10's extra layers for the track plots: the corrected tracks as they were
    (grey), where the correction moved them (dark green) and the correction clicks (dark green
    rings). [] per view outside step 10."""
    if not corrected:
        return [[] for _ in views]
    ids = {t.id for t in corrected}
    before = project_tracks([t for t in source_tracks or [] if t.id in ids], views, data)
    green = moved[1] if moved is not None else project_tracks(corrected, views, data)
    out = []
    for v, d, b, g in zip(views, data, before, green):
        clicks = {k: p[p[:, 2] > 0, :2] for k, p in _hand_points((hand_points or {}).get(v.name, {}), d).items()}
        out.append([(b, {"marker": ".", "color": "0.35", "ms": 2, "alpha": 0.7}, "track before correction"),
                    (g, {"marker": "o", "color": CORRECTED_COLOUR, "ms": 2.5, "alpha": 0.9},
                     "corrected track, where the correction moved it"),
                    ({k: c for k, c in clicks.items() if len(c)},
                     {"marker": "o", "mfc": "none", "mec": CORRECTED_COLOUR, "mew": 1.5, "ms": 11},
                     "correction clicks")])
    return out


def evaluate_tracks(session, config: dict, tracks: list[Track3D], gt_tracks: list[np.ndarray] | None,
                    out_dir: Path, corrected_ids=frozenset(),
                    hand_points: dict[str, dict[int, list[tuple[float, float]]]] | None = None,
                    source_tracks: list[Track3D] | None = None) -> list[str]:
    """Writes evaluation_3d.txt, evaluation_2d.txt, evaluation.json, track_plots_backproj/ and the
    back-projected labels and videos in reprojection/ (as enabled in tracking3d). Returns the lines
    to print. Step 10: the tracks in corrected_ids are drawn green in the videos where the correction
    moved them (over MOVED_PX from the same track in source_tracks, the result before correction) and
    red like any other track where it did not (every frame green without source_tracks); hand_points
    (view name -> {aligned frame: [(x, y) raw px]}, the correction clicks) as green rings."""
    s, msgs, results = session.settings, [], {}
    if session.source != "pred":
        return ["Evaluation skipped: tracks built from the ground-truth labels."]
    if gt_tracks:
        results["3d"] = evaluate_3d(tracks, gt_tracks, float(s["eval_match_distance"]))
        text = format_3d(results["3d"])
        (out_dir / "evaluation_3d.txt").write_text(text + "\n")
        save_3d_charts(out_dir / "evaluation_3d.png", results["3d"])
        msgs.append(text)
    else:
        msgs.append("3D evaluation skipped: no ground-truth 3D points for this run.")

    if s["evaluate_2d"] or s["visualise_backproj_graph"] or s["visualise_backproj_video"]:
        es = eval_settings(config)
        vis = config.get("visualisation", {})
        data = [load_view_data(session, v, es, vis) for v in session.views]
        inserted, on_track = backproject(tracks, session.views, data, float(s["backproj_score"]))
        n_ins = sum(len(b) for ins in inserted for b in ins.values())
        has_gt = any(len(f.gt) for d in data for f in d.ds.frames)
        if s["evaluate_2d"] and has_gt:
            results["2d"] = evaluate_2d(session.views, data, inserted, on_track, es)
            results["2d_centres"] = centre_errors(session.views, data, inserted, on_track, tracks, float(s["eval_match_px"]))
            text = format_2d(results["2d"], data, float(s["backproj_score"]), s["conf_threshold"],
                             results["2d_centres"], float(s["eval_match_px"]))
            (out_dir / "evaluation_2d.txt").write_text(text + "\n")
            save_2d_charts(out_dir / "evaluation_2d.png", results["2d"], results["2d_centres"])
            msgs.append(text)
        elif s["evaluate_2d"]:
            msgs.append("2D evaluation skipped: this run has no ground-truth labels.")
        if s["visualise_backproj_graph"] or s["visualise_backproj_video"]:
            label_dir = out_dir / "reprojection" / "labels"
            for v, d, ins in zip(session.views, data, inserted):
                pred_dir = session.run.path / v.input_name / session.model / PRED_DIR
                save_predictions(label_dir / v.input_name, d.ds, ins,
                                 d.ds.class_names or read_class_names(pred_dir / "data.yaml"))
            msgs.append(f"Back-projected labels: {label_dir}  ({n_ins} boxes inserted over all views)")
        moved = None   # step 10: the corrected tracks' points, split into (unchanged, moved by the correction)
        corrected = [t for t in tracks if t.id in corrected_ids]
        if corrected and source_tracks is not None:
            moved = _split_moved(corrected, {t.id: t for t in source_tracks}, session.views, data)
        if s["visualise_backproj_video"]:
            points = project_tracks([t for t in tracks if t.id not in corrected_ids], session.views, data)
            if moved is None:
                green = project_tracks(corrected, session.views, data)
            else:
                same, green = moved
                for pts, sm in zip(points, same):   # unchanged by the correction: red, like any track
                    for k, p in sm.items():
                        pts[k] = np.vstack([pts[k], p]) if k in pts else p
            for v, d, ins, pts, gr in zip(session.views, data, inserted, points, green):
                save_backproj_video(d.ds.frames, d.preds, ins, pts, out_dir / "reprojection" / f"{safe_name(v.name)}.mp4",
                                    v.name, session.model, d.vis_conf, fps=float(vis.get("video_fps", 30)),
                                    corrected_points=gr,
                                    hand_points=_hand_points((hand_points or {}).get(v.name, {}), d))
            msgs.append(f"Back-projection videos: {out_dir / 'reprojection'}/<camera>.mp4")
        if s["visualise_backproj_graph"]:
            plot_dir = out_dir / "track_plots_backproj"
            layers = _correction_layers(corrected, source_tracks, moved, hand_points, session.views, data)
            for v, d, ins, lay in zip(session.views, data, inserted, layers):
                save_track_plots(d.ds.frames, d.preds, plot_dir, session.model, d.vis_conf,
                                 max_distance=vis.get("track_max_distance_px", 50),
                                 max_gap=vis.get("track_max_frame_gap", 5),
                                 background=vis.get("track_plot_background", True),
                                 extra={k: (b[:, :2] + b[:, 2:4]) / 2 for k, b in ins.items()},
                                 extra_label="factor graph, back-projected where the detector has no box",
                                 layers=lay)
            msgs.append(f"Back-projected track plots: {plot_dir}  ({n_ins} boxes inserted over all views)")
    if results:
        (out_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    return msgs
