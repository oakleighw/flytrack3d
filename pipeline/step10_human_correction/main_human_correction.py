"""Hand corrections of a past factor-graph result (guided or unguided), solved again.

1. Choose a run, then the result to correct: one of its guided (main_tracks3d_guided.py) or
   unguided (main_tracks3d.py) results.
2. A correction window opens on each camera's video in turn (tracking3d/annotator/correction_tool.py).
   It shows the YOLO detections and the result's tracks: yellow where the track is married to a
   detection or inferred by the motion prior, red where its confidence is low (3D sigma above
   correction_low_sigma), on the image and on the timeline. Click where the selected track's fly
   really is, in any view, at any frame; "continue" moves to the next video, "finish" on the
   last one starts the correction. Corrections are saved to <run>/3d/corrections.json (per
   source result) and loaded again next time, where they can be changed or deleted (--no-gui
   reuses them as they are).
3. Post-processing, with a progress bar: every track with corrections is solved again by the
   factor graph, with the corrections as trusted anchors next to the result's START / END
   annotations (tracking3d/correction.py). Tracks without corrections are kept as they are.
4. Outputs, evaluation, back-projection videos (the corrected tracks and the correction clicks in
   green) and the 3D plot: the source tracks in their colours, the corrected ones in dark green.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python pipeline/step10_human_correction/main_human_correction.py                    # choose, correct, solve
    python pipeline/step10_human_correction/main_human_correction.py --result latest    # the run's latest result
    python pipeline/step10_human_correction/main_human_correction.py --no-gui           # solve the saved corrections

Outputs go to a new folder, <run>/3d/<model>/factor_graph/corrected_<datetime>/: the same files as
step 8 (tracks_3d.csv holds every track, corrected or kept), plus source_tracks_3d.csv (the tracks
before correction), corrections.json (the corrections used) and corrected_observations.csv (the
detections and corrections the corrected tracks are fitted to).
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from scene3d import locate_arena  # noqa: E402
from common.config import load_config  # noqa: E402
from tracking3d.annotator import run_correction_gui, view_inputs_from_session  # noqa: E402
from tracking3d.correction import (KIND, correct_tracks, corrected_results, fixed_annotations,  # noqa: E402
                                   load_result_tracks, read_corrections, reassociate, source_key,
                                   track_views, write_corrections)
from tracking3d.evaluate import evaluate_tracks  # noqa: E402
from tracking3d.guided import anchor_residuals  # noqa: E402
from tracking3d.multiview import _ask, ground_truth_tracks, open_session, set_session_offsets  # noqa: E402
from tracking3d.past_results import SOURCE_CSV, choose_past_run, choose_result  # noqa: E402
from tracking3d.report import write_track_outputs  # noqa: E402
from tracking3d.triangulate import MatchSettings, triangulate_session  # noqa: E402
from visualise.plots2d.reprojection import save_reprojection_plots  # noqa: E402
from visualise.plots3d.scene import plot_tracks_3d, use_interactive_backend  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "configs" / "tracking.json"   # needs "cameras" (calibration); "tracking3d" tunes the graph
RUN = None           # run folder name or path; None = choose in the terminal
MODEL = None         # whose results: a model in that run or "ground_truth"; None = the only one, or choose
KIND_TO_FIX = None   # "guided" or "tracks" (unguided); None = choose in the terminal
RESULT = None        # result folder, e.g. "guided_20261012-143417", or "latest"; None = choose in the terminal
GUI = True           # open the correction window (pre-filled with saved corrections); False = use them as is
LOW_SIGMA = None     # low-confidence threshold (3D sigma, world units); None = tracking3d.correction_low_sigma
VIDEO = True         # back-projection videos (reprojection/<camera>.mp4), corrections in green; False = none
GT_OVERLAY = True    # draw the run's ground truth (triangulated + linked, no factor graph) in pale green
ARENA = True         # draw the fake fruit and leaves, found by HSV in each view (config arena_plotting)
SHOW = True          # open an interactive (rotatable) 3D window after saving
# =====================================================================================

SOURCE_KINDS = ("guided", "tracks")


def choose_source(config: dict, args):
    """(run folder, every result of the run, the guided / unguided result to correct)."""
    run_dir, results = choose_past_run(Path(config["output_dir"]), args.run)
    fixable = [r for r in results if r.kind in SOURCE_KINDS]
    if not fixable:
        raise SystemExit(f"{run_dir.name} has no guided or unguided factor-graph result to correct. "
                         f"Run main_tracks3d_guided.py or main_tracks3d.py on it first.")
    return run_dir, results, choose_result(fixable, args.kind, args.model, args.result)


def use_saved_calibration(config: dict, result) -> None:
    """The tracks are only valid in the calibration they were built with: use the one saved in the
    result's params.json (its views' entries and t_scale) over cameras.json's, when it has one."""
    saved = result._json("params.json").get("cameras")
    if not saved:
        print("  This result predates saving the calibration with it: using the current cameras.json. "
              "If the calibration has changed since, its tracks will not line up (checked below).")
        return
    cams = config["cameras"]
    changed = [n for n, v in saved.get("views", {}).items() if cams.get("views", {}).get(n) != v]
    if "t_scale" in saved and saved["t_scale"] != cams.get("t_scale"):
        changed.append("t_scale")
    for name, v in saved.get("views", {}).items():
        cams.setdefault("views", {})[name] = {**v, "frame_offset": cams["views"].get(name, v).get(
            "frame_offset", v.get("frame_offset", 0))}   # offsets: set from the result just after
    if "t_scale" in saved:
        cams["t_scale"] = saved["t_scale"]
    if changed:
        print(f"  Using the calibration saved with the result ({', '.join(changed)} differ from cameras.json)")


def check_alignment(source_tracks, views, gate: float) -> None:
    """Warn if the source tracks no longer sit on the detections they were fitted to (a changed
    calibration or frame offset): far fewer are found again than the result recorded."""
    for t in source_tracks:
        had = int(np.sum(t.views))
        now = reassociate(t, views, gate).score
        if had >= 20 and now < 0.5 * had:
            print(f"\n  WARNING: track #{t.id} meets only {now} of the {had} detections it was fitted to. The "
                  f"calibration or frame offsets have probably changed since this result was made, so its "
                  f"tracks are drawn in the wrong place. Re-run step 8 with the current calibration, then "
                  f"correct that result.")


def load_saved_corrections(run_dir: Path, key: str) -> list[dict]:
    """This source's saved corrections; if it has none but another source of the run has, offer those."""
    saved = read_corrections(run_dir)
    if saved.get(key, {}).get("corrections") is not None:
        found = saved[key]["corrections"]
        print(f"\nLoaded {len(found)} saved correction(s) for {key} (updated {saved[key].get('updated', '?')})")
        return found
    others = [(k, v) for k, v in saved.items() if v.get("corrections")]
    if not others:
        return []
    print(f"\nNo corrections saved for {key}. Other results of this run have some (track numbers are those "
          f"results' - guided results number tracks by annotated object, so theirs carry over):")
    print(f"  {1:>3}  start with none")
    for i, (k, v) in enumerate(others, 2):
        print(f"  {i:>3}  copy the {len(v['corrections'])} correction(s) of {k}")
    k = _ask("Choose", len(others) + 1, 0)
    return [dict(c) for c in others[k - 1][1]["corrections"]] if k else []


def overlays(tracks, views, gate: float, ids=None) -> dict:
    """{track id: {view name: TrackView}} for the correction window."""
    return {t.id: track_views(t, views, gate) for t in tracks if ids is None or t.id in ids}


def describe(corrections: list[dict], view_names: list[str]) -> list[str]:
    lines = []
    for obj in sorted({c["object"] for c in corrections}):
        parts = [f"{n} {sum(c['object'] == obj and c['view'] == n for c in corrections)}" for n in view_names]
        lines.append(f"  #{obj}: " + ", ".join(parts))
    return lines


def save_observations(path: Path, tracks, objects: dict, views) -> None:
    """corrected_observations.csv: what each corrected track is fitted to - its married YOLO detections
    and the hand corrections - in raw pixels, with the aligned frame and the view's own frame number."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "frame", "view", "view_frame", "x", "y", "kind"])
        for t in tracks:
            if t.id not in objects:
                continue
            rows = [(f, v, *views[v].raw[f][j, :2], "detection") for f, v, j in t.observations]
            rows += [(a.frame, a.view, a.raw["x"], a.raw["y"], KIND) for a in objects[t.id].anchors if a.kind == KIND]
            for f, v, x, y, kind in sorted(rows, key=lambda r: (r[0], r[1])):
                w.writerow([t.id, int(f), views[v].name, int(f) - views[v].camera.frame_offset,
                            round(float(x), 2), round(float(y), 2), kind])


def main():
    parser = argparse.ArgumentParser(description="Hand-correct a past factor-graph result and solve it again.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--model", default=MODEL, help="A model in the run, or 'ground_truth'.")
    parser.add_argument("--kind", choices=SOURCE_KINDS, default=KIND_TO_FIX, help="Kind of result to correct.")
    parser.add_argument("--result", default=RESULT, help="Result folder name, or 'latest'.")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=GUI,
                        help="Open the correction window; --no-gui solves the saved corrections.")
    parser.add_argument("--low-sigma", type=float, default=LOW_SIGMA,
                        help="Low-confidence threshold, 3D sigma in world units (default tracking3d.correction_low_sigma).")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=VIDEO,
                        help="Back-projection videos with the corrections in green (--no-video: none).")
    parser.add_argument("--video-dir", action="append", default=[],
                        help="A folder to look for the run's videos in (and its sub-folders) when they are no longer "
                             "at the path saved in metadata.json; repeatable, searched before the config's video_dirs.")
    parser.add_argument("--gt-overlay", action=argparse.BooleanOptionalAction, default=GT_OVERLAY)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=ARENA)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    args = parser.parse_args()

    config = load_config(args.config)
    dirs = config.get("video_dirs") or []
    config["video_dirs"] = args.video_dir + ([dirs] if isinstance(dirs, str) else list(dirs))
    # Step 1 - the result to correct, and the session as it saw it (its tracking3d settings and offsets).
    run_dir, results, result = choose_source(config, args)
    key = source_key(result)
    print(f"\nCorrecting {result.path}")
    settings, offsets = result.saved_settings()
    config["tracking3d"] = {**config.get("tracking3d", {}), **settings}
    use_saved_calibration(config, result)
    source = "gt" if result.label == "ground_truth" else "pred"
    session = open_session(config, str(run_dir), None if source == "gt" else result.label, source)
    changes = set_session_offsets(session, offsets, config["cameras"])   # the result's, over the run's
    if changes:
        print("Using the source result's frame offsets: " + "; ".join(changes))
    views, s = session.views, session.settings
    if args.low_sigma is not None:
        s["correction_low_sigma"] = args.low_sigma
    s["visualise_backproj_video"] = bool(args.video)
    names = [v.name for v in views]
    gate = float(s["association_gate_px"])

    source_tracks = load_result_tracks(result.csv)
    if not source_tracks:
        raise SystemExit(f"{result.csv} holds no tracks.")
    check_alignment(source_tracks, views, gate)
    fixed = fixed_annotations(result)
    print(f"\nSource: {len(source_tracks)} track(s) " + ", ".join(
        f"#{t.id} {t.frames[0]}-{t.frames[-1]}" for t in source_tracks)
        + (f"; START / END annotations for {len(fixed)}" if fixed else " (unguided: no START / END)"))

    # Step 2 - corrections, in the window (or the saved ones).
    corrections = load_saved_corrections(run_dir, key)
    if args.gui:
        earlier = corrected_results(results, key)
        corrected_view = {}
        if earlier:
            last = earlier[-1]
            print(f"Earlier correction of this result shown in green: {last.name}")
            corrected_view = overlays(load_result_tracks(last.csv), views, gate, last.corrected_ids())
        print(f"\nOpening the correction window ({len(views)} views, {len(corrections)} correction(s) loaded) ...")
        what = "ground-truth labels" if session.source == "gt" else f"{session.model} detections"
        out, low = run_correction_gui(
            view_inputs_from_session(session), overlays(source_tracks, views, gate), corrections,
            [a for anns in fixed.values() for a in anns], corrected_view,
            title=f"{session.run.name}  |  {result.name}", detections_label=what,
            suggest_gate_px=s["max_epipolar_px"], low_sigma=float(s["correction_low_sigma"]))
        if out is None:
            raise SystemExit("Correction cancelled - nothing saved, nothing solved.")
        corrections, s["correction_low_sigma"] = out, low
        path = write_corrections(run_dir, key, corrections, {v.name: v.camera.frame_offset for v in views},
                                 result.path)
        print(f"Saved {len(corrections)} correction(s) to {path}")
    if not corrections:
        raise SystemExit("No corrections - nothing to solve again." + ("" if args.gui else
                         f" (none saved for {key}: run with the GUI first)"))
    print("\nCorrections per track and view:\n" + "\n".join(describe(corrections, names)))

    # Step 3 - post-processing: the corrected tracks are solved again (tracking3d/correction.py).
    tracks, objects, warnings = correct_tracks(source_tracks, fixed, corrections, views, s)
    if not objects:
        raise SystemExit("\n".join(["No track could be corrected."] + warnings))
    extra = [f"Source: {result.path}", f"Corrections: {len(objects)} of "
             f"{len(tracks)} track(s) corrected, the others kept as they were", ""]
    extra += describe(corrections, names) + ["", "Guidance"]
    per_object, anchors_3d = {}, []
    for t in tracks:
        if t.id not in objects:
            continue
        obj = objects[t.id]
        res = anchor_residuals(t, obj, views)
        corr_res = [r for r, a in zip(res, obj.anchors) if a.kind == KIND]
        extra.append(f"  #{t.id}: {len(corr_res)} correction(s), track meets them within {np.median(corr_res):.1f} px "
                     f"(median), {max(corr_res):.1f} px (max); {len(obj.anchors) - len(corr_res)} START / END")
        extra += [f"       {n}" for n in obj.notes]
        per_object[t.id] = {"annotations": [a.raw for a in obj.anchors], "anchor_residual_px": res, "notes": obj.notes}
        anchors_3d += [(t.id, a.kind, t.xyz[a.frame - t.frames[0]]) for a in obj.anchors]
    if warnings:
        extra += ["", "Warnings"] + [f"  {w}" for w in warnings]

    # Step 4 - outputs and evaluation.
    out_dir = session.factor_graph_dir("corrected")
    shutil.copy(result.csv, out_dir / SOURCE_CSV)
    (out_dir / "corrections.json").write_text(json.dumps({"source": str(result.path), "source_key": key,
                                                          "corrections": corrections}, indent=2))
    save_observations(out_dir / "corrected_observations.csv", tracks, objects, views)
    report = write_track_outputs(
        session, tracks, s, out_dir, f"{len(tracks)} track(s), {len(objects)} hand-corrected", extra,
        {"source": str(result.path), "source_key": key, "corrected": sorted(objects), "objects": per_object},
        params={"mode": "corrected", "source": str(result.path), "source_key": key, "source_kind": result.kind,
                "corrections_file": str(run_dir / "3d" / "corrections.json"), "corrections": corrections,
                "evaluation": config.get("evaluation", {})},
        cam_cfg=config["cameras"])
    print("\n" + report)

    hand: dict[str, dict[int, list]] = {}
    for c in corrections:
        if c["object"] in objects and c["view"] in names:
            off = views[names.index(c["view"])].camera.frame_offset
            hand.setdefault(c["view"], {}).setdefault(int(c["frame"]) + off, []).append((c["x"], c["y"]))
    gt = ground_truth_tracks(session, config["cameras"]) if source == "pred" else None
    print("\n" + "\n\n".join(evaluate_tracks(session, config, tracks, gt, out_dir, set(objects), hand,
                                             source_tracks)))

    corrected = [t for t in tracks if t.id in objects]
    points = triangulate_session([v.camera for v in views], [v.dets for v in views],
                                 MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
    pts = np.array([[f, *p.xyz, len(p.views)] for f, ps in points.items() for p in ps]).reshape(-1, 5)
    raw_ann = {}
    for t in corrected:
        for a in objects[t.id].anchors:
            raw_ann.setdefault(views[a.view].name, []).append((t.id, a.kind, a.raw["x"], a.raw["y"]))
    arena = locate_arena(session, config) if args.arena else None
    drawn = result.load_tracks()   # the source tracks as saved, with their detection counts for the legend
    save_reprojection_plots(drawn, views, out_dir / "reprojection", annotations=raw_ann, arena=arena,
                            corrected=corrected)
    title = (f"{session.run.name}  |  {session.label}  |  {result.name} in colour, dark green = hand-corrected "
             f"(diamonds = corrections)")
    gt = gt if args.gt_overlay else None
    if gt:
        title += "  |  pale green band = ground truth"
    show = args.show and use_interactive_backend()
    plot_tracks_3d(drawn, [v.camera for v in views], out_dir / "tracks_3d.png", title, pts, gt, anchors_3d,
                   close=not show, arena=arena, corrected=corrected)
    print(f"\nSaved to {out_dir}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
