"""The 3D plot of many runs at once (main_visualise_all.py): every run's tracks, moved into the
shared fruit-centred frame of scene3d/align.py, one colour per group of runs, over the arena.

Four panels: the rotatable 3D scene, and three orthographic views of it: front (as the cameras
see the wall), top (looking down, the wall at the top) and side (the wall on the right).
plot_groups draws every group in one set of panels; plot_side_by_side gives each group its own
column of the same panels, on the same axis limits, to compare them.

A plot is saved as two files next to its png, which are all main_visualise_past_all.py needs to
draw it again (save_plot / load_plot):
    aligned_tracks.csv  every track point, in the shared frame
    plot.json           the groups' colours, the title, the track style and the arena objects
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scene3d.arena import ArenaFrame, draw_frame_3d, draw_frame_panel  # noqa: E402
from scene3d.fruit import Fruit, draw_fruit_3d, draw_fruit_panel, fruit_points  # noqa: E402
from scene3d.leaves import Leaves, draw_leaves_3d, draw_leaves_panel  # noqa: E402
from tracking3d.past_results import SavedTrack  # noqa: E402

AXES = ("x: right (mm)", "y: towards the wall (mm)", "z: up (mm)")
# (title, horizontal axis, vertical axis); in the top view the wall (y = 0) is at the top, the cameras below.
PANELS = (("front  (x-z, as the cameras see it)", 0, 2), ("top  (x-y, looking down)", 0, 1),
          ("side  (y-z, the wall on the right)", 1, 2))
DEFAULT_COLOURS = ("#e0458f", "#2f6fd6", "#e8871e", "#1a9e8f", "#7b5ac8", "#8a6d3b")   # fixed order


@dataclass
class TrackStyle:
    lw: float = 1.0
    alpha: float = 0.7
    detections: bool = True        # dots on the frames that have detections (n_views >= 1)
    dot_size: float = 2.0
    endpoints: bool = True         # entry triangle, exit square


@dataclass
class Group:
    name: str
    colour: str
    tracks: list = field(default_factory=list)   # (run name, SavedTrack in the shared frame)
    runs: int = 0


@dataclass
class SharedArena:
    """The arena objects in the shared frame: one fruit and frame (the same in every run there),
    and every run's leaves (they differ a little: they are found per run)."""
    fruit: object | None = None
    frame: object | None = None
    leaves: list = field(default_factory=list)
    fruit_style: dict = field(default_factory=dict)
    leaves_style: dict = field(default_factory=dict)
    frame_style: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        f = self.frame
        return {"fruit": self.fruit.to_json() if self.fruit else None,
                "frame": {"back": f.back.tolist(), "front": f.front.tolist(), "up": f.up.tolist()} if f else None,
                "leaves": [lv.to_json() for lv in self.leaves],
                "fruit_style": self.fruit_style, "leaves_style": self.leaves_style, "frame_style": self.frame_style}

    @classmethod
    def from_json(cls, d: dict) -> "SharedArena":
        f = d.get("frame")
        return cls(Fruit.from_json(d["fruit"]) if d.get("fruit") else None,
                   ArenaFrame(*(np.asarray(f[k], float) for k in ("back", "front", "up"))) if f else None,
                   [Leaves.from_json(lv) for lv in d.get("leaves", [])],
                   d.get("fruit_style", {}), d.get("leaves_style", {}), d.get("frame_style", {}))

    def points(self) -> np.ndarray:
        parts = ([fruit_points(self.fruit, self.fruit_style)] if self.fruit else []) + \
                [np.asarray(lv.outline) for lv in self.leaves] + ([self.frame.points] if self.frame else [])
        return np.vstack(parts) if parts else np.zeros((0, 3))

    def _leaves_style(self) -> dict:
        # Overlapping translucent fills: where most runs' leaves agree the green is darkest.
        a = float(self.leaves_style.get("alpha", 0.45))
        return {**self.leaves_style, "alpha": max(a / max(len(self.leaves), 1) ** 0.5, 0.04)}

    def draw_3d(self, ax) -> None:
        if self.frame:
            draw_frame_3d(ax, self.frame, self.frame_style, "arena")
        s = self._leaves_style()
        for n, lv in enumerate(self.leaves):
            draw_leaves_3d(ax, lv, s, f"leaves ({len(self.leaves)} runs)" if n == 0 else None)
        if self.fruit:
            draw_fruit_3d(ax, self.fruit, self.fruit_style, "fruit")

    def draw_panel(self, ax, i: int, j: int) -> None:
        if self.frame:
            draw_frame_panel(ax, self.frame, i, j, self.frame_style)
        s = self._leaves_style()
        for lv in self.leaves:
            draw_leaves_panel(ax, lv, i, j, s)
        if self.fruit:
            draw_fruit_panel(ax, self.fruit, i, j, self.fruit_style)


def _equal_3d(ax, xyz: np.ndarray) -> None:
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    centre, half = (lo + hi) / 2, max((hi - lo).max() / 2, 1e-6) * 1.05
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - half, c + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(AXES[0], fontsize=8)
    ax.set_ylabel(AXES[1], fontsize=8)
    ax.set_zlabel(AXES[2], fontsize=8)
    ax.tick_params(labelsize=7)


def _draw_track(axes3d, panels, xyz, n_views, colour, st: TrackStyle) -> None:
    det = n_views >= 1 if n_views is not None else np.zeros(len(xyz), bool)
    for ax in axes3d:
        ax.plot(*xyz.T, color=colour, lw=st.lw, alpha=st.alpha)
        if st.detections and det.any():
            ax.scatter(*xyz[det].T, color=colour, s=st.dot_size, alpha=st.alpha, linewidths=0)
        if st.endpoints:
            ax.scatter(*xyz[0], color=colour, marker="^", s=30, edgecolor="black", linewidths=0.5)
            ax.scatter(*xyz[-1], color=colour, marker="s", s=20, edgecolor="black", linewidths=0.5)
    for ax, (_, i, j) in zip(panels, PANELS):
        ax.plot(xyz[:, i], xyz[:, j], color=colour, lw=st.lw, alpha=st.alpha, zorder=3)
        if st.detections and det.any():
            ax.scatter(xyz[det, i], xyz[det, j], color=colour, s=st.dot_size, alpha=st.alpha, linewidths=0, zorder=3)
        if st.endpoints:
            ax.plot(xyz[0, i], xyz[0, j], "^", color=colour, mec="black", mew=0.5, ms=5, zorder=4)
            ax.plot(xyz[-1, i], xyz[-1, j], "s", color=colour, mec="black", mew=0.5, ms=4, zorder=4)


def _extent(groups: list[Group], arena: SharedArena | None) -> np.ndarray:
    """(N, 3) every track point and the arena: what the axes must show."""
    tracks = [t.xyz for g in groups for _, t in g.tracks]
    if not tracks:
        raise ValueError("no tracks to plot")
    return np.vstack(tracks + ([arena.points()] if arena else []))


def _setup_3d(ax, data: np.ndarray, title: str) -> None:
    _equal_3d(ax, data)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=20, azim=-70)   # from the cameras' side, a little to the left and above


def _setup_panel(ax, name: str, i: int, j: int, limits: np.ndarray | None = None) -> None:
    """limits: (2, 2) fixed [x, y] ranges (side by side), or None to fit the data (autoscale)."""
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(AXES[i], fontsize=8)
    ax.set_ylabel(AXES[j], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    if limits is None:
        ax.set_aspect("equal", adjustable="datalim")
    else:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_aspect("equal", adjustable="box")


def _draw_scene(ax3d, panels, groups: list[Group], arena: SharedArena | None, st: TrackStyle) -> None:
    if arena:
        arena.draw_3d(ax3d)
        for ax, (_, i, j) in zip(panels, PANELS):
            arena.draw_panel(ax, i, j)
    for g in groups:
        for _, t in g.tracks:
            _draw_track([ax3d], panels, t.xyz, t.n_views, g.colour, st)


def _legend(fig, groups: list[Group], st: TrackStyle, ax3d) -> None:
    handles = [Line2D([], [], color=g.colour, lw=2.5, label=f"{g.name}: {len(g.tracks)} tracks, {g.runs} runs")
               for g in groups]
    if st.endpoints:
        handles += [Line2D([], [], ls="", marker="^", color="0.6", mec="black", label="entry"),
                    Line2D([], [], ls="", marker="s", color="0.6", mec="black", label="exit")]
    if st.detections:
        handles.append(Line2D([], [], ls="", marker="o", ms=3, color="0.4",
                              label="frames with YOLO detections (line only = bridged by the factor graph)"))
    arena_handles = ax3d.get_legend_handles_labels()[0]   # fruit, leaves, arena added by the draw functions
    fig.legend(handles=handles + arena_handles, loc="outside lower center", ncol=min(len(handles) + 3, 5),
               fontsize=9, frameon=False)


def _finish(fig, out_path: Path | None, close: bool, dpi: int):
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
    if close:
        plt.close(fig)
    return fig


def plot_groups(groups: list[Group], arena: SharedArena | None, title: str, style: TrackStyle | None = None,
                out_path: Path | None = None, close: bool = True, dpi: int = 150):
    """Every group's tracks over the shared arena. Saved to out_path if given."""
    st = style or TrackStyle()
    data = _extent(groups, arena)
    fig = plt.figure(figsize=(16, 13), layout="constrained")
    fig.suptitle(title, fontsize=12)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    panels = [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]
    _setup_3d(ax3d, data, "scene (fruit-centred: the fruit's base is the origin)")
    for ax, (name, i, j) in zip(panels, PANELS):
        _setup_panel(ax, name, i, j)
    _draw_scene(ax3d, panels, groups, arena, st)
    for ax in panels:
        ax.autoscale_view()
    _legend(fig, groups, st, ax3d)
    return _finish(fig, out_path, close, dpi)


def plot_side_by_side(groups: list[Group], arena: SharedArena | None, title: str, style: TrackStyle | None = None,
                      out_path: Path | None = None, close: bool = True, dpi: int = 150):
    """One column per group (its tracks alone over the shared arena), one row per view: the 3D
    scene, front, top and side. Every column has the same axis limits, so the groups compare
    directly. Saved to out_path if given."""
    st = style or TrackStyle()
    data = _extent(groups, arena)
    lo, hi = data.min(axis=0), data.max(axis=0)
    pad = 0.04 * (hi - lo).max()
    n = len(groups)
    fig = plt.figure(figsize=(7.5 * n, 26), layout="constrained")
    fig.suptitle(title, fontsize=13)
    ax3d = None
    for c, g in enumerate(groups):
        ax3d = fig.add_subplot(4, n, 1 + c, projection="3d")
        panels = [fig.add_subplot(4, n, 1 + c + r * n) for r in (1, 2, 3)]
        _setup_3d(ax3d, data, f"{g.name}: {len(g.tracks)} tracks, {g.runs} runs")
        for ax, (name, i, j) in zip(panels, PANELS):
            _setup_panel(ax, name, i, j, np.array([[lo[i] - pad, hi[i] + pad], [lo[j] - pad, hi[j] + pad]]))
        _draw_scene(ax3d, panels, [g], arena, st)
    _legend(fig, groups, st, ax3d)
    return _finish(fig, out_path, close, dpi)



PLOT_FILES = ("aligned_tracks.csv", "plot.json")
CSV_COLUMNS = ["group", "run", "result", "track", "frame", "x", "y", "z", "sigma", "n_views"]


def save_plot(out_dir: Path, groups: list[Group], arena: SharedArena | None, title: str, style: TrackStyle,
              rows: list[list]) -> None:
    """rows: the csv rows (CSV_COLUMNS) of every track drawn, in the shared frame."""
    with open(out_dir / PLOT_FILES[0], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)
    plot = {"title": title, "groups": [{"name": g.name, "colour": g.colour, "runs": g.runs} for g in groups],
            "track_style": asdict(style), "arena": arena.to_json() if arena else None}
    (out_dir / PLOT_FILES[1]).write_text(json.dumps(plot, indent=1))


def load_plot(out_dir: Path) -> tuple[list[Group], SharedArena | None, str, TrackStyle]:
    """What save_plot wrote: (groups with their tracks, arena, title, track style)."""
    plot = json.loads((out_dir / PLOT_FILES[1]).read_text())
    groups = {g["name"]: Group(g["name"], g["colour"], runs=int(g["runs"])) for g in plot["groups"]}
    by_track: dict[tuple, list] = {}
    with open(out_dir / PLOT_FILES[0], newline="") as fh:
        for r in csv.DictReader(fh):
            by_track.setdefault((r["group"], r["run"], int(r["track"])), []).append(r)
    for (group, run, tid), rs in by_track.items():
        xyz = np.array([[float(r[k]) for k in "xyz"] for r in rs])
        n_views = np.array([int(r["n_views"]) for r in rs])
        track = SavedTrack(tid, np.array([int(r["frame"]) for r in rs]), xyz, int(n_views.sum()), n_views)
        groups[group].tracks.append((run, track))
    arena = SharedArena.from_json(plot["arena"]) if plot.get("arena") else None
    return list(groups.values()), arena, plot["title"], TrackStyle(**plot["track_style"])
