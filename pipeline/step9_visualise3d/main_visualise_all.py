"""Plot the 3D tracks of every run in a folder together, and compare folders (groups) of runs.

Each group is a folder of run folders (the ones steps 5 to 8 write); every run in it with a
saved factor-graph result is drawn in the group's colour. By default that is its latest
hand-corrected result (step 10, main_human_correction.py), else its latest guided result
(main_tracks3d_guided.py), else its latest unguided one: the config's "kinds" (or --kind) sets the
order, e.g. "kinds": ["guided", "tracks"] to leave the corrections out. Each track is the factor-graph
estimate (the line) fitted to the YOLO detections it was married to (the dots: frames that
have detections in at least one camera; where the line runs without dots, the factor graph
bridged a gap on its motion prior alone).

Runs are recorded with the rig in different places and with different reference cameras, so
each run is moved into one shared frame centred on its fruit before drawing (scene3d/align.py):
the origin is the fruit's base on the wall, x is right, y towards the wall, z up, in world
units (mm). The transform is rigid, built from the run's cached fruit fit (3d/fruit/fruit.json)
and the calibration it was fitted with, so no video is needed. The leaves and arena frame are
moved with it. A run without a fruit fit cannot be aligned and is skipped (listed in the
terminal).

Saved to <output_dir>/visualise_all/<groups>_<datetime>/ (or "save_dir" in the config):
    aligned_tracks.png   the 4-panel plot
    side_by_side.png     with several groups (and "side_by_side" on): one column per group, the
                         same views on the same axes, to compare the groups' tracks
    aligned_tracks.csv   group, run, result, track, frame, x, y, z, sigma, n_views (shared frame)
    plot.json            colours, title, style and the aligned arena: with the csv, all that
                         main_visualise_past_all.py needs to show the plot again
    alignment.json       per run: the result drawn, its transform and fruit size

Config: configs/visualise_all.json (template: configs/visualise_all.example.json).

From the IDE: fill in the config (or the SETTINGS block below) and press Run (or F5).
From the command line, flags override the config; --group replaces the config's groups:
    python pipeline/step9_visualise3d/main_visualise_all.py
    python pipeline/step9_visualise3d/main_visualise_all.py --group cladding path/to/cladding "#e0458f" --group control path/to/control "#2f6fd6"
    python pipeline/step9_visualise3d/main_visualise_all.py --group all path/to/runs --no-show
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from common.config import load_config  # noqa: E402
from scene3d.align import FRUIT_NORMAL, align_run, calibration_mismatch  # noqa: E402
from scene3d.arena import frame_settings  # noqa: E402
from scene3d.fruit import Fruit, fruit_settings  # noqa: E402
from scene3d.leaves import leaves_settings  # noqa: E402
from tracking3d.past_results import KINDS, SavedTrack, find_results  # noqa: E402
from visualise.plots3d.groups import (DEFAULT_COLOURS, Group, SharedArena, TrackStyle, plot_groups,  # noqa: E402
                                      plot_side_by_side, save_plot)
from visualise.plots3d.scene import use_interactive_backend  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "configs" / "visualise_all.json"   # "visualise_all", "arena_plotting"
SHOW = None          # open the interactive window; None = the config's
SAVE_DIR = None      # where to save; None = the config's (null there = <output_dir>/visualise_all/...)
# =====================================================================================

MAX_DEPTH = 3        # how far below a group's folder runs are looked for


def find_runs(folder: Path, depth: int = 0) -> list[Path]:
    """Run folders (holding a 3d/ folder) in folder or below it; a run's own sub-folders are not searched."""
    if (folder / "3d").is_dir():
        return [folder]
    if depth >= MAX_DEPTH:
        return []
    return [r for d in sorted(p for p in folder.iterdir() if p.is_dir()) for r in find_runs(d, depth + 1)]


def pick_result(run: Path, kinds: list[str], detections: str | None):
    """The latest result of the first kind in kinds that the run has, or (None, reason)."""
    results = [r for r in find_results(run) if r.kind != "points"]
    labels = sorted({r.label for r in results})
    if detections:
        results = [r for r in results if r.label == detections]
    elif len(labels) > 1:
        preds = [lab for lab in labels if lab != "ground_truth"]
        results = [r for r in results if r.label == (preds or labels)[0]]
    for kind in kinds:
        hits = [r for r in results if r.kind == kind]
        if hits:
            return hits[-1], None
    return None, f"no saved {' / '.join(kinds)} result" + (f" for {detections}" if detections else "")


def load_group(name: str, folder: Path, colour: str, s: dict, arena: dict, rows: list, report: dict) -> Group:
    group = Group(name, colour)
    runs = find_runs(folder)
    print(f"\n{name}: {len(runs)} run(s) in {folder}")
    for run in runs:
        result, why = pick_result(run, s["kinds"], s.get("detections"))
        alignment = align_run(run) if result else None
        if isinstance(alignment, str):
            why = alignment
        if why:
            print(f"  skipped  {run.name}: {why}")
            report["skipped"].append({"group": name, "run": str(run), "reason": why})
            continue
        bad = calibration_mismatch(alignment, result._json("params.json"))
        if bad:
            print(f"  warning  {run.name}: the result's calibration of {', '.join(bad)} differs from the one "
                  f"its fruit was fitted with; it may not line up")
        saved = np.loadtxt(result.csv, delimiter=",", skiprows=1, ndmin=2)   # track, frame, x, y, z, sigma, n_views
        kept = 0
        for t in result.load_tracks():
            if len(t.frames) < int(s["min_track_frames"]):
                continue
            xyz = alignment.apply(t.xyz)
            group.tracks.append((run.name, SavedTrack(t.id, t.frames, xyz, t.score, t.n_views)))
            sigma = saved[saved[:, 0] == t.id, 5]
            rows += [[name, run.name, result.name, t.id, int(f), *np.round(p, 4), sg, int(nv)]
                     for f, p, sg, nv in zip(t.frames, xyz, sigma, t.n_views)]
            kept += 1
        group.runs += 1
        arena["alignments"].append(alignment)
        report["runs"].append({"group": name, "result": str(result.path), "kind": result.kind,
                               "tracks": kept, **alignment.to_json()})
        print(f"  {result.kind:<7}  {run.name}: {kept} track(s)  ({result.name})")
    return group


def shared_arena(alignments: list, config: dict, what: dict) -> SharedArena | None:
    """One fruit (median size) and arena frame at the origin, and every run's leaves."""
    if not alignments or not any(what.values()):
        return None
    fruit_style, leaves_style = fruit_settings(config)["mesh"], leaves_settings(config)["mesh"]
    frame_mesh = frame_settings(config)["mesh"]
    radius = float(np.median([a.fruit.radius for a in alignments]))
    depth = float(np.median([a.fruit.depth for a in alignments]))
    fruit = Fruit([0.0, 0.0, 0.0], FRUIT_NORMAL.tolist(), radius, depth, radius, {}, {})
    leaves = [lv for a in alignments if (lv := a.leaves_aligned()) is not None] if what["leaves"] else []
    return SharedArena(fruit if what["fruit"] else None,
                       alignments[0].frame_aligned(frame_mesh) if what["frame"] else None,
                       leaves, fruit_style, leaves_style, frame_mesh)


def main():
    parser = argparse.ArgumentParser(description="Plot every run's 3D tracks in a shared fruit-centred frame, "
                                                 "one colour per group (folder) of runs.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--group", nargs="+", action="append", metavar=("NAME FOLDER", "COLOUR"),
                        help="A group: its name, its folder of runs and optionally a colour. Repeat to compare "
                             "groups. Replaces the config's groups.")
    parser.add_argument("--kind", nargs="+", choices=[k for k in KINDS if k != "points"],
                        help="Result kinds to draw, in order of preference (default: the config's).")
    parser.add_argument("--save-dir", default=SAVE_DIR, help="Folder to save the plot, csv and json in.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    args = parser.parse_args()

    config = load_config(args.config)
    s = config["visualise_all"]
    groups_cfg = s["groups"]
    if args.group:
        if any(len(g) not in (2, 3) for g in args.group):
            raise SystemExit("--group takes NAME FOLDER [COLOUR]")
        groups_cfg = [{"name": g[0], "folder": str(Path(g[1]).resolve()), "colour": g[2] if len(g) > 2 else None}
                      for g in args.group]
    if args.kind:
        s["kinds"] = args.kind
    show = s.get("show", True) if args.show is None else args.show
    if not groups_cfg:
        raise SystemExit("No groups: give them in the config's visualise_all.groups or with --group.")

    arena, rows = {"alignments": []}, []
    report = {"created": datetime.now().isoformat(timespec="seconds"), "frame": {
        "origin": "the fruit's base on the wall", "x": "right along the wall, as the cameras see it",
        "y": "towards (into) the wall: the cameras are at negative y", "z": "up the wall"},
        "runs": [], "skipped": []}
    groups = []
    for n, g in enumerate(groups_cfg):
        folder = Path(g["folder"])
        if not folder.is_dir():
            raise SystemExit(f"Group '{g['name']}': folder not found: {folder}")
        colour = g.get("colour") or DEFAULT_COLOURS[n % len(DEFAULT_COLOURS)]
        groups.append(load_group(g["name"], folder, colour, s, arena, rows, report))
    if not any(g.tracks for g in groups):
        raise SystemExit("\nNo run could be drawn (see the reasons above).")

    what = {k: bool(s["arena"].get(k, True)) for k in ("fruit", "leaves", "frame")}
    shared = shared_arena(arena["alignments"], config, what)
    names = " vs ".join(g.name for g in groups)
    title = (f"{names}  |  {sum(len(g.tracks) for g in groups)} tracks from {sum(g.runs for g in groups)} runs, "
             f"aligned on the fruit")
    style = TrackStyle(**s["track_style"])

    out_dir = Path(args.save_dir or s.get("save_dir") or
                   Path(config["output_dir"]) / "visualise_all" /
                   f"{'_vs_'.join(g.name for g in groups)}_{datetime.now():%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if show and not use_interactive_backend():   # switch before plotting: switching closes open figures
        print("No interactive matplotlib backend (TkAgg / QtAgg) is available: saving only.")
        show = False
    plot_groups(groups, shared, title, style, out_dir / "aligned_tracks.png", close=not show,
                dpi=int(s.get("dpi", 150)))
    drawn = [g for g in groups if g.tracks]
    if len(drawn) > 1 and s.get("side_by_side", True):
        plot_side_by_side(drawn, shared, title, style, out_dir / "side_by_side.png", dpi=int(s.get("dpi", 150)))
    save_plot(out_dir, groups, shared, title, style, rows)
    (out_dir / "alignment.json").write_text(json.dumps(report, indent=1))
    skipped = len(report["skipped"])
    print(f"\nSaved to {out_dir}" + (f"  ({skipped} run(s) skipped, listed in alignment.json)" if skipped else ""))

    if show:
        import matplotlib.pyplot as plt
        print("Close the window to finish.")
        plt.show()


if __name__ == "__main__":
    main()
