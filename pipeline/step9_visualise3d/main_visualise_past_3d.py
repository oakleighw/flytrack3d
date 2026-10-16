"""Open the interactive (rotatable) 3D plot of a past 3D result again.

main_points3d.py, main_tracks3d.py and main_tracks3d_guided.py show this plot once, when
they finish. This script shows it for any result they saved, without running the
triangulation report or the factor graph again:

1. Choose a run (only runs with saved 3D results are listed).
2. Choose what to view: its triangulated points, a past unguided run or a past guided run.
3. If there are several of that kind, choose which one (they are listed by date).

The points / tracks are read from the result's csv. The cameras, the ground-truth overlay,
the arena and the grey triangulated points behind the tracks are rebuilt from the run and
the current config, using the tracking3d settings and frame offsets saved with the result
(params.json). The calibration is saved there too (results from before it was added lack
it), but is not read back: if cameras.views has changed since, those parts are drawn with
the new calibration. Nothing is written to the result's folder
(the arena is fitted and cached in <run>/3d/ if the run has no cache yet, as in steps 7 and 8).

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python pipeline/step9_visualise3d/main_visualise_past_3d.py                    # choose everything from lists
    python pipeline/step9_visualise3d/main_visualise_past_3d.py --kind guided      # choose a run, then a guided result
    python pipeline/step9_visualise3d/main_visualise_past_3d.py --run run_20261008-101724_showoff_1 --kind tracks --result latest
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from scene3d import locate_arena  # noqa: E402
from common.config import load_config  # noqa: E402
from tracking3d.multiview import ground_truth_tracks, open_session  # noqa: E402
from tracking3d.past_results import KINDS, choose_past_run, choose_result  # noqa: E402
from tracking3d.triangulate import MatchSettings, triangulate_session  # noqa: E402
from visualise.plots3d.scene import plot_points_3d, plot_tracks_3d, use_interactive_backend  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "configs" / "tracking.json"   # needs "cameras" (calibration)
RUN = None           # run folder name or path; None = choose in the terminal
MODEL = None         # whose results: a model in that run or "ground_truth"; None = the only one, or choose
KIND = None          # "points", "tracks" (unguided) or "guided"; None = choose in the terminal
RESULT = None        # result folder, e.g. "guided_20261012-143417", or "latest"; None = choose in the terminal
GT_OVERLAY = True    # draw the run's ground truth (triangulated + linked, no factor graph) in green
ARENA = True         # draw the fake fruit and leaves, found by HSV in each view (config arena_plotting)
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Show the interactive 3D plot of a past 3D result.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--model", default=MODEL, help="A model in the run, or 'ground_truth'.")
    parser.add_argument("--kind", choices=list(KINDS), default=KIND)
    parser.add_argument("--result", default=RESULT, help="Result folder name, or 'latest'.")
    parser.add_argument("--gt-overlay", action=argparse.BooleanOptionalAction, default=GT_OVERLAY,
                        help="Overlay ground-truth 3D tracks in green (ignored for ground-truth results).")
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=ARENA,
                        help="Draw the fake fruit and its leaves (found by HSV in every view).")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir, results = choose_past_run(Path(config["output_dir"]), args.run)
    result = choose_result(results, args.kind, args.model, args.result)
    print(f"\nViewing {result.path}")

    # Rebuild the session as the result saw it: its tracking3d settings and frame offsets.
    settings, offsets = result.saved_settings()
    config["tracking3d"] = {**config.get("tracking3d", {}), **settings}
    for name, offset in offsets.items():
        cam = config.get("cameras", {}).get("views", {}).get(name)
        if cam is not None and int(cam.get("frame_offset", 0)) != int(offset):
            print(f"  {name}: using the result's frame_offset {offset} (config has {cam.get('frame_offset', 0)})")
            cam["frame_offset"] = offset
    source = "gt" if result.label == "ground_truth" else "pred"
    session = open_session(config, str(run_dir), None if source == "gt" else result.label, source)
    views, s = session.views, session.settings
    cams = [v.camera for v in views]

    gt = ground_truth_tracks(session, config["cameras"]) if args.gt_overlay and source == "pred" else None
    arena = locate_arena(session, config) if args.arena else None
    if not use_interactive_backend():
        raise SystemExit("No interactive matplotlib backend (TkAgg / QtAgg) is available to show the plot.")

    title = f"{session.run.name}  |  {session.label}  |  "
    if result.kind == "points":
        points = result.load_points()
        title += f"{len(points)} triangulated points (colour = frame, large = 3 views)"
        if gt:
            title += "  |  green = ground truth"
        plot_points_3d(points, cams, None, title, gt, close=False, arena=arena)
    else:
        tracks = result.load_tracks()
        if not tracks:
            raise SystemExit(f"{result.csv} holds no tracks.")
        # The grey points behind the tracks are not saved with them: triangulate them again.
        triangulated = triangulate_session(cams, [v.dets for v in views],
                                           MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
        pts = np.array([[f, *p.xyz, len(p.views)] for f, ps in triangulated.items() for p in ps]).reshape(-1, 5)
        anchors = None
        if result.kind == "guided":
            anchors = result.anchors(tracks, {v.name: v.camera.frame_offset for v in views})
            title += f"{len(tracks)} guided tracks (+ = annotated start, x = annotated end)"
        else:
            title += f"{len(tracks)} tracks (triangle = entry, square = exit)"
        if gt:
            title += "  |  green = ground truth"
        plot_tracks_3d(tracks, cams, None, title, pts, gt, anchors, close=False, arena=arena)

    print(f"\nShowing {result.name}  ({result.created:%Y-%m-%d %H:%M}). Close the window to finish.")
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
