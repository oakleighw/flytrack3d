"""3D tracks for hand-picked objects, guided by annotated start / end points.

1. Choose a run (as in main_tracks3d.py).
2. An annotation window opens on each camera's video in turn. Mark where each object
   of interest starts and / or ends - in any views, any mix. "continue" moves to the
   next video, "finish" on the last one runs the tracking. In a later video, "suggest
   starts" / "suggest ends" proposes the points from the earlier views (epipolar
   geometry, see tracking3d/annotator/suggest.py); check them, delete or re-place any wrong ones.
   Annotations are saved to <run>/3d/annotations.json and loaded again next time, so
   they can be refined or reused (--no-gui).
3. Each annotated object gets exactly one factor-graph track: the annotations fix its
   frame range, act as trusted reprojection factors and seed it, so it can bridge
   stretches where the detector lost it. See tracking3d/guided.py.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python pipeline/step8_tracking3d/main_tracks3d_guided.py                  # pick a run, annotate, track
    python pipeline/step8_tracking3d/main_tracks3d_guided.py --no-gui         # reuse the saved annotations
    python pipeline/step8_tracking3d/main_tracks3d_guided.py --source gt      # track through the ground-truth labels

Outputs go to a new folder per run, <run>/3d/<model or ground_truth>/factor_graph/guided_<datetime>/
(same files as main_tracks3d.py - tracks, params, evaluation - plus the annotations drawn in the plots).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from scene3d import locate_arena  # noqa: E402
from common.config import load_config  # noqa: E402
from tracking3d.annotator import run_annotation_gui, view_input_from_source  # noqa: E402
from tracking3d.evaluate import evaluate_tracks  # noqa: E402
from tracking3d.guided import anchor_residuals, build_guided_tracks  # noqa: E402
from tracking3d.multiview import ground_truth_tracks, open_session  # noqa: E402
from tracking3d.report import write_track_outputs  # noqa: E402
from tracking3d.triangulate import MatchSettings, camera_baseline_report, triangulate_session  # noqa: E402
from visualise.plots2d.reprojection import save_reprojection_plots  # noqa: E402
from visualise.plots3d.scene import plot_tracks_3d, use_interactive_backend  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "configs" / "tracking.json"   # needs "cameras" (calibration); "tracking3d" tunes the graph
RUN = None           # run folder name or path; None = choose in the terminal
MODEL = None         # model in that run; None = the only one, or choose in the terminal
SOURCE = "pred"      # "pred" = the run's predictions, "gt" = its ground-truth labels
ANNOTATIONS = None   # annotations file; None = <run>/3d/annotations.json
GUI = True           # open the annotator (pre-filled with saved annotations); False = use the file as is
GT_OVERLAY = True    # draw the run's ground truth (triangulated + linked, no factor graph) in green
ARENA = True         # draw the fake fruit and leaves, found by HSV in each view (config arena_plotting)
SHOW = True          # open an interactive (rotatable) 3D window after saving
# =====================================================================================


def annotate(session, path: Path, use_gui: bool) -> list[dict]:
    saved = json.loads(path.read_text())["annotations"] if path.is_file() else []
    if not use_gui:
        if not saved:
            raise SystemExit(f"No saved annotations at {path}. Run with the GUI first.")
        print(f"\nUsing {len(saved)} saved annotations from {path}")
        return saved

    views = []
    for v in session.views:
        detections = {f - v.camera.frame_offset: d for f, d in v.raw.items()}  # back to label frame numbers
        try:
            views.append(view_input_from_source(v.name, v.input_name, v.source, detections, v.camera))
        except FileNotFoundError as e:
            raise SystemExit(f"Cannot open the video for {v.name}: {e}") from None
    print(f"\nOpening the annotator ({len(views)} views"
          f"{f', {len(saved)} saved annotations loaded' if saved else ''}) ...")
    what = "ground-truth labels" if session.source == "gt" else f"{session.model} detections"
    result = run_annotation_gui(views, saved, title=f"{session.run.name}  |  {session.label}",
                                detections_label=what, suggest_gate_px=session.settings["max_epipolar_px"])
    if result is None:
        raise SystemExit("Annotation cancelled - nothing saved, nothing tracked.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run": session.run.name, "views": [v.name for v in session.views],
                                "annotations": result}, indent=2))
    print(f"Saved {len(result)} annotations to {path}")
    return result


def describe_annotations(annotations: list[dict], view_names: list[str]) -> list[str]:
    lines = []
    for obj in sorted({a["object"] for a in annotations}):
        parts = []
        for name in view_names:
            here = {a["kind"]: a["frame"] for a in annotations if a["object"] == obj and a["view"] == name}
            if here:
                parts.append(f"{name} " + "/".join(f"{k} {here[k]}" for k in ("start", "end") if k in here))
        lines.append(f"  #{obj}: " + ("; ".join(parts) or "-"))
    return lines


def main():
    parser = argparse.ArgumentParser(description="3D tracks for annotated objects (start / end guided factor graph).")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--source", choices=["pred", "gt"], default=SOURCE)
    parser.add_argument("--conf", type=float, default=None, help="Override tracking3d.conf_threshold.")
    parser.add_argument("--annotations", default=ANNOTATIONS, help="Annotations file (default <run>/3d/annotations.json).")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=GUI,
                        help="Open the annotator; --no-gui reuses the saved annotations.")
    parser.add_argument("--gt-overlay", action=argparse.BooleanOptionalAction, default=GT_OVERLAY,
                        help="Overlay ground-truth 3D tracks in green (ignored with --source gt).")
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=ARENA,
                        help="Draw the fake fruit and its leaves (found by HSV in every view).")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    args = parser.parse_args()

    config = load_config(args.config)
    # Step 1 - load detections, as in main_tracks3d.py.
    session = open_session(config, args.run, args.model, args.source, args.conf)
    views, s = session.views, session.settings
    cams = [v.camera for v in views]
    print("\nCamera rig:\n" + "\n".join(camera_baseline_report(cams)))

    ann_path = Path(args.annotations) if args.annotations else session.run.path / "3d" / "annotations.json"
    annotations = annotate(session, ann_path, args.gui)
    if not annotations:
        raise SystemExit("No annotations - nothing to track.")
    print("\nAnnotated objects:\n" + "\n".join(describe_annotations(annotations, [v.name for v in views])))

    # Step 2 - cross-view matching and triangulation, as in main_tracks3d.py.
    points = triangulate_session(cams, [v.dets for v in views],
                                 MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
    print(f"\nTriangulated {sum(len(p) for p in points.values())} points in {len(points)} frames")

    # Steps 3-5 - the unguided pipeline builds candidate seeds, then one factor graph per
    # annotated object (tracking3d/guided.py: build_guided_tracks).
    tracks, objects, warnings = build_guided_tracks(views, points, annotations, s)
    by_id = {o.id: o for o in objects}
    extra = ["Annotations"] + describe_annotations(annotations, [v.name for v in views]) + ["", "Guidance"]
    anchors_3d, per_object = [], {}
    for t in tracks:
        obj = by_id[t.id]
        res = anchor_residuals(t, obj, views)
        extra.append(f"  #{t.id}: {len(obj.anchors)} annotation(s), track meets them within "
                     f"{np.median(res):.1f} px (median), {max(res):.1f} px (max)")
        extra += [f"       {n}" for n in obj.notes]
        per_object[t.id] = {"annotations": [a.raw for a in obj.anchors], "anchor_residual_px": res,
                            "notes": obj.notes}
        for a in obj.anchors:
            anchors_3d.append((t.id, a.kind, t.xyz[a.frame - t.frames[0]]))
    if warnings:
        extra += ["", "Warnings"] + [f"  {w}" for w in warnings]
    if not tracks:
        raise SystemExit("\n".join(["No object could be tracked."] + warnings))

    # Step 6 - outputs and evaluation.
    out_dir = session.factor_graph_dir("guided")
    report = write_track_outputs(session, tracks, s, out_dir, f"{len(tracks)} annotated object(s), by object number",
                                 extra, {"annotation_file": str(ann_path), "objects": per_object},
                                 params={"mode": "guided", "annotation_file": str(ann_path),
                                         "evaluation": config.get("evaluation", {})},
                                 cam_cfg=config["cameras"])
    print("\n" + report)

    gt = ground_truth_tracks(session, config["cameras"]) if args.source == "pred" else None
    print("\n" + "\n\n".join(evaluate_tracks(session, config, tracks, gt, out_dir)))

    pts = np.array([[f, *p.xyz, len(p.views)] for f, ps in points.items() for p in ps]).reshape(-1, 5)
    raw_ann = {}
    for a in annotations:
        raw_ann.setdefault(a["view"], []).append((a["object"], a["kind"], a["x"], a["y"]))
    arena = locate_arena(session, config) if args.arena else None
    save_reprojection_plots(tracks, views, out_dir / "reprojection", annotations=raw_ann,
                            arena=arena)
    title = (f"{session.run.name}  |  {session.label}  |  {len(tracks)} guided tracks "
             f"(+ = annotated start, x = annotated end)")
    gt = gt if args.gt_overlay else None
    if gt:
        title += "  |  green = ground truth"
    show = args.show and use_interactive_backend()
    plot_tracks_3d(tracks, cams, out_dir / "tracks_3d.png", title, pts, gt, anchors_3d, close=not show,
                   arena=arena)
    print(f"\nSaved to {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
