"""Estimate 3D tracks (scene entry -> fruit) from a previous batch evaluation run.

Triangulated points (see main_points3d.py) seed tracks, which a gtsam factor
graph then refines against every 2D detection - including those seen by a single
camera - with a constant-velocity motion prior. See tracking3d/factor_graph.py for details
and configs/shared/tracking3d.example.json for the settings.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python pipeline/step8_tracking3d/main_tracks3d.py                     # pick a run from a list
    python pipeline/step8_tracking3d/main_tracks3d.py --n-tracks 3        # keep the 3 best-supported tracks
    python pipeline/step8_tracking3d/main_tracks3d.py --source gt         # build tracks from the ground-truth labels
    python pipeline/step8_tracking3d/main_tracks3d.py --no-show

Outputs go to a new folder per run, <run>/3d/<model or ground_truth>/factor_graph/tracks_<datetime>/:
    tracks_3d.csv            per-frame position, uncertainty and views of every track
    tracks.json / summary.txt  per-track entry point, length, speed
    params.json              the settings this run used
    tracks_3d.png            3D + orthographic views with the raw triangulated points
    reprojection/<cam>.png   tracks projected back into each camera (calibration check)
    reprojection/<cam>.mp4   video per camera: detector / GT / back-projected boxes (visualise_backproj_video)
    reprojection/labels/     back-projected boxes as YOLO labels, one folder per input (either visualise_backproj_*)
    evaluation_3d.txt        tracks vs the ground-truth 3D points: recall, RMSE, ... (needs GT labels)
    evaluation_2d.txt        2D P/R/F1 per IoU, box localisation and detector AP, with the tracks back-projected into the gaps (evaluate_2d)
    evaluation.json          both evaluations in full
    evaluation_2d.png / evaluation_3d.png  bar charts of the two evaluations
    track_plots_backproj/    2D track plots plus the back-projected boxes (visualise_backproj_graph)
See tracking3d/evaluate.py for how the evaluation works.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from scene3d import locate_arena  # noqa: E402
from common.config import load_config  # noqa: E402
from tracking3d.evaluate import evaluate_tracks  # noqa: E402
from tracking3d.factor_graph import build_tracks  # noqa: E402
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
N_TRACKS = None      # keep the n best tracks; None = tracking3d.n_tracks (null = every valid track)
GT_OVERLAY = True    # draw the run's ground truth (triangulated + linked, no factor graph) in green
ARENA = True         # draw the fake fruit and leaves, found by HSV in each view (config arena_plotting)
SHOW = True          # open an interactive (rotatable) 3D window after saving
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="3D tracks from multi-view detections with a factor graph.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--source", choices=["pred", "gt"], default=SOURCE)
    parser.add_argument("--n-tracks", type=int, default=N_TRACKS,
                        help="Keep the n best-supported tracks (overrides tracking3d.n_tracks).")
    parser.add_argument("--conf", type=float, default=None, help="Override tracking3d.conf_threshold.")
    parser.add_argument("--gt-overlay", action=argparse.BooleanOptionalAction, default=GT_OVERLAY,
                        help="Overlay ground-truth 3D tracks in green (ignored with --source gt).")
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=ARENA,
                        help="Draw the fake fruit and its leaves (found by HSV in every view).")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    args = parser.parse_args()

    config = load_config(args.config)
    # Step 1 - load every camera's detections: conf filter, frame offset, undistortion,
    # static-clutter removal (tracking3d/multiview.py: load_views, remove_static_detections).
    session = open_session(config, args.run, args.model, args.source, args.conf)
    views, s = session.views, session.settings
    if args.n_tracks is not None:
        s["n_tracks"] = args.n_tracks
    cams = [v.camera for v in views]
    print("\nCamera rig:\n" + "\n".join(camera_baseline_report(cams)))

    # Step 2 - per frame: match detections across views (epipolar Hungarian) and triangulate
    # groups seen by >= 2 views (tracking3d/triangulate.py: match_frame).
    points = triangulate_session(cams, [v.dets for v in views],
                                 MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
    n_pts = sum(len(p) for p in points.values())
    print(f"\nTriangulated {n_pts} points in {len(points)} frames")
    if not n_pts:
        raise SystemExit("No detections could be matched across views - check the calibration.")

    # Steps 3-5 - link points into tracklets, refine each with the factor graph, then merge,
    # filter and rank the tracks (tracking3d/factor_graph.py: build_tracks).
    tracks, tracklets = build_tracks(views, points, s)
    print(f"Linked {len(tracklets)} tracklets (>= {s['min_tracklet_frames']} frames) -> {len(tracks)} tracks")
    if not tracks:
        raise SystemExit("No tracks survived. Try lowering min_tracklet_frames / min_track_frames or "
                         "raising link_max_distance (world units per frame).")

    # Step 6 - outputs, then the 3D / 2D evaluation against ground truth (if the run has labels).
    out_dir = session.factor_graph_dir("tracks")
    report = write_track_outputs(session, tracks, s, out_dir, f"{len(tracks)} track(s), best supported first",
                                 params={"mode": "unguided", "evaluation": config.get("evaluation", {})},
                                 cam_cfg=config["cameras"])
    print("\n" + report)

    gt = ground_truth_tracks(session, config["cameras"]) if args.source == "pred" else None
    print("\n" + "\n\n".join(evaluate_tracks(session, config, tracks, gt, out_dir)))

    pts = np.array([[f, *p.xyz, len(p.views)] for f, ps in points.items() for p in ps])
    arena = locate_arena(session, config) if args.arena else None
    save_reprojection_plots(tracks, views, out_dir / "reprojection", arena=arena)
    title = f"{session.run.name}  |  {session.label}  |  {len(tracks)} tracks (triangle = entry, square = exit)"
    gt = gt if args.gt_overlay else None
    if gt:
        title += "  |  green = ground truth"
    show = args.show and use_interactive_backend()
    plot_tracks_3d(tracks, cams, out_dir / "tracks_3d.png", title, pts, gt, close=not show,
                   arena=arena)
    print(f"\nSaved to {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
