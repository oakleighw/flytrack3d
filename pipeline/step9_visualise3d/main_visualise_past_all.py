"""Open the interactive (rotatable) 3D plot of a past main_visualise_all.py output again.

main_visualise_all.py finds every run, aligns it on its fruit and shows the plot once. This
script shows any plot it saved, without finding or aligning the runs again (the run folders
need not even be there any more): everything is read from the output folder, its
aligned_tracks.csv (the tracks, already in the shared frame) and plot.json (colours, title,
style and the aligned arena).

1. Choose an output (they are listed by date, newest last) from <output_dir>/visualise_all/.
2. If it has several groups, choose what to show: all of them together, all of them side by side
   (one column each, on the same axes), or one.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python pipeline/step9_visualise3d/main_visualise_past_all.py                      # choose from lists
    python pipeline/step9_visualise3d/main_visualise_past_all.py --output latest
    python pipeline/step9_visualise3d/main_visualise_past_all.py --output latest --side-by-side
    python pipeline/step9_visualise3d/main_visualise_past_all.py --output cladding_vs_control_20261106-105925 --group control
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from common.config import load_config  # noqa: E402
from tracking3d.multiview import _ask  # noqa: E402
from visualise.plots3d.groups import PLOT_FILES, load_plot, plot_groups, plot_side_by_side  # noqa: E402
from visualise.plots3d.scene import use_interactive_backend  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "configs" / "visualise_all.json"   # only its output_dir is used
OUTPUT = None        # output folder name or path, or "latest"; None = choose in the terminal
GROUP = None         # a group's name to show alone; None = all of them, or choose in the terminal
SIDE_BY_SIDE = None  # True = every group in its own column; None = choose in the terminal
# =====================================================================================


def output_root(config_path: Path) -> Path:
    """<output_dir>/visualise_all; the default output/ if the config is missing."""
    try:
        return Path(load_config(config_path)["output_dir"]) / "visualise_all"
    except SystemExit:
        return ROOT / "output" / "visualise_all"


def past_outputs(root: Path) -> list[Path]:
    """Output folders that can be drawn again, oldest first."""
    dirs = [d for d in root.glob("*") if d.is_dir() and all((d / f).is_file() for f in PLOT_FILES)]
    return sorted(dirs, key=lambda d: (d / PLOT_FILES[1]).stat().st_mtime)


def choose_output(root: Path, output: str | None) -> Path:
    outputs = past_outputs(root)
    if output and output != "latest":
        p = Path(output)
        p = p if p.is_dir() else root / output
        if not all((p / f).is_file() for f in PLOT_FILES):
            raise SystemExit(f"No saved plot in {p} (needs {' and '.join(PLOT_FILES)}; "
                             f"outputs from before plot.json was saved cannot be shown: run main_visualise_all.py again).")
        return p
    if not outputs:
        raise SystemExit(f"No saved main_visualise_all.py output in {root}. Run main_visualise_all.py first.")
    if output == "latest" or len(outputs) == 1:
        return outputs[-1]
    width = max(len(d.name) for d in outputs) + 2
    print(f"\nPast main_visualise_all.py outputs in {root}, oldest first:")
    print(f"  {'#':>3}  {'output':<{width}}{'created':<18}groups")
    for i, d in enumerate(outputs, 1):
        plot = json.loads((d / PLOT_FILES[1]).read_text())
        groups = ", ".join(f"{g['name']} ({g['runs']} runs)" for g in plot["groups"])
        created = datetime.fromtimestamp((d / PLOT_FILES[1]).stat().st_mtime)
        print(f"  {i:>3}  {d.name:<{width}}{created:%Y-%m-%d %H:%M}  {groups}")
    return outputs[_ask("Choose an output", len(outputs), len(outputs) - 1)]


def choose_groups(groups: list, name: str | None, side_by_side: bool | None) -> tuple[list, bool]:
    """(groups to show, whether side by side)."""
    if name:
        hits = [g for g in groups if g.name == name]
        if not hits:
            raise SystemExit(f"No group '{name}'. Available: {[g.name for g in groups]}")
        return hits, False
    if len(groups) == 1 or side_by_side is not None:
        return groups, bool(side_by_side) and len(groups) > 1
    names = " vs ".join(g.name for g in groups)
    print("\nGroups to show:")
    print(f"  {1:>3}  all together: {names}")
    print(f"  {2:>3}  all side by side: {names} (one column each, same axes)")
    for i, g in enumerate(groups, 3):
        print(f"  {i:>3}  {g.name}  ({len(g.tracks)} tracks, {g.runs} runs)")
    k = _ask("Choose the groups", len(groups) + 2, 0)
    return (groups, k == 1) if k < 2 else ([groups[k - 2]], False)


def main():
    parser = argparse.ArgumentParser(description="Show the interactive 3D plot of a past main_visualise_all.py output.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output", default=OUTPUT, help="Output folder name or path, or 'latest'.")
    parser.add_argument("--group", default=GROUP, help="Show only this group.")
    parser.add_argument("--side-by-side", action=argparse.BooleanOptionalAction, default=SIDE_BY_SIDE,
                        help="Every group in its own column (--no-side-by-side: all together).")
    args = parser.parse_args()

    out_dir = choose_output(output_root(Path(args.config)), args.output)
    print(f"\nViewing {out_dir}")
    groups, arena, title, style = load_plot(out_dir)
    shown, side_by_side = choose_groups(groups, args.group, args.side_by_side)
    if len(shown) < len(groups):
        title = (f"{shown[0].name}  |  {len(shown[0].tracks)} tracks from {shown[0].runs} runs, aligned on the fruit"
                 f"  (from {' vs '.join(g.name for g in groups)})")

    if not use_interactive_backend():   # before plotting: switching closes open figures
        raise SystemExit("No interactive matplotlib backend (TkAgg / QtAgg) is available to show the plot.")
    (plot_side_by_side if side_by_side else plot_groups)(shown, arena, title, style, close=False)
    print("Close the window to finish.")
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
