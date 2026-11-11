"""3D plots: triangulated points, factor-graph tracks and their ground-truth, annotation and
arena overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common import ANCHOR_MARKERS, CORRECTED_COLOUR, GT_COLOUR, TRACK_COLOURS  # noqa: E402

POINTS_CMAP = "plasma"   # frame colouring for triangulated points; contains no green
# Ground truth is a wide translucent band; predictions are drawn thinner on top, so where they
# agree the prediction runs down the middle of the band and where they differ the band shows.
GT_LW, GT_ALPHA = 6.0, 0.35
PRED_POINT_SIZE = (1.5, 4.0)   # scatter size for 2-view, 3-view points
PRED_TRACK_LW = 1.2
CORRECTED_LW = 2.0
PANELS = (("top  (X-Y)", 0, 1), ("front  (X-Z)", 0, 2))   # orthographic panels
RIG_ARENA_ALPHA = 0.25   # zoomed out to the cameras, the fruit is as small as the tracks: keep them visible


def use_interactive_backend() -> bool:
    """Switch from Agg to a GUI backend for plt.show(). False if none is available."""
    for backend in ("TkAgg", "QtAgg"):
        try:
            plt.switch_backend(backend)
            return True
        except Exception:
            continue
    return False


def _equal_3d(ax, xyz: np.ndarray) -> None:
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    centre, half = (lo + hi) / 2, max((hi - lo).max() / 2, 1e-6) * 1.05
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _draw_cameras(ax, cameras, length: float) -> None:
    for c in cameras:
        p, d = c.centre, c.axis * length
        ax.scatter(*p, color="black", marker="s", s=30)
        ax.quiver(*p, *d, color="black", lw=1)
        ax.text(*p, f" {c.name}", fontsize=8)


def _scene_figure(title: str, data: np.ndarray, cameras):
    """2x2 figure: 3D scene, two orthographic views, 3D rig (cameras + data).
    Returns (fig, ax3d, [ortho axes], ax_rig)."""
    fig = plt.figure(figsize=(14, 12), layout="constrained")
    fig.suptitle(title, fontsize=12)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ortho = [fig.add_subplot(2, 2, i) for i in (2, 3)]
    rig = fig.add_subplot(2, 2, 4, projection="3d")
    _equal_3d(ax3d, data)
    rig_pts = np.vstack([data, np.array([c.centre for c in cameras])])
    _equal_3d(rig, rig_pts)
    _draw_cameras(rig, cameras, 0.15 * np.ptp(rig_pts, axis=0).max())
    ax3d.set_title("scene")
    rig.set_title("cameras + scene (calibration check)")
    # The world is the reference camera's frame, where Y points down (as image rows do). Flip a
    # panel's vertical axis where it runs against the cameras' mean image-up, so the arena's
    # floor is at the bottom (the cameras are mounted upright).
    up = np.mean([-c.R[1] for c in cameras], axis=0)
    for ax, (name, _, j) in zip(ortho, PANELS):
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        if up[j] < 0:
            ax.invert_yaxis()
    return fig, ax3d, ortho, rig


def _draw_gt_tracks(ax3d, ortho, rig, gt_tracks) -> None:
    """Ground-truth 3D tracks ((K, 4) frame, x, y, z) as green lines under everything else."""
    for n, t in enumerate(gt_tracks):
        label = f"ground truth ({len(gt_tracks)} tracks)" if n == 0 else None
        for ax in (ax3d, rig):
            ax.plot(*t[:, 1:4].T, color=GT_COLOUR, lw=GT_LW, alpha=GT_ALPHA, label=label if ax is ax3d else None)
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.plot(t[:, 1 + i], t[:, 1 + j], color=GT_COLOUR, lw=GT_LW, alpha=GT_ALPHA, zorder=1)


def _draw_arena(ax3d, ortho, rig, arena) -> None:
    """Static arena objects (scene3d.ArenaObjects: fruit, leaves), under the tracks."""
    arena.draw_3d(ax3d)
    arena.draw_3d(rig, labels=False, alpha=RIG_ARENA_ALPHA)
    for ax, (_, i, j) in zip(ortho, PANELS):
        arena.draw_panel(ax, i, j)


def plot_points_3d(points: np.ndarray, cameras, out_path: Path | None, title: str,
                   gt_tracks: list[np.ndarray] | None = None, close: bool = True,
                   arena=None):
    """points: (N, 5) frame, x, y, z, n_views. Scatter coloured by frame number; optional
    ground-truth tracks in green and arena objects (scene3d.ArenaObjects)."""
    gt_tracks = gt_tracks or []
    data = np.vstack([points[:, 1:4]] + [t[:, 1:4] for t in gt_tracks]
                     + ([arena.points()] if arena else []))
    fig, ax3d, ortho, rig = _scene_figure(title, data, cameras)
    _draw_gt_tracks(ax3d, ortho, rig, gt_tracks)
    if arena:
        _draw_arena(ax3d, ortho, rig, arena)
    kw = dict(c=points[:, 0], cmap=POINTS_CMAP, s=np.where(points[:, 4] >= 3, PRED_POINT_SIZE[1], PRED_POINT_SIZE[0]),
              alpha=0.9, zorder=3)
    sc = ax3d.scatter(points[:, 1], points[:, 2], points[:, 3], **kw)
    rig.scatter(points[:, 1], points[:, 2], points[:, 3], **kw)
    for ax, (_, i, j) in zip(ortho, PANELS):
        ax.scatter(points[:, 1 + i], points[:, 1 + j], **kw)
    fig.colorbar(sc, ax=[ax3d, *ortho, rig], shrink=0.5, label="frame")
    if gt_tracks or arena:
        ax3d.legend(loc="upper left", fontsize=7, framealpha=0.8)
    if out_path is not None:   # None: only build the figure (viewing a past result)
        fig.savefig(out_path, dpi=110)
    if close:
        plt.close(fig)
    return fig


def plot_tracks_3d(tracks, cameras, out_path: Path | None, title: str, points: np.ndarray | None = None,
                   gt_tracks: list[np.ndarray] | None = None,
                   anchors: list[tuple[int, str, np.ndarray]] | None = None, close: bool = True,
                   arena=None, corrected=None):
    """Factor-graph tracks as lines (entry = triangle, exit = square); raw triangulated points
    grey; optional ground-truth tracks in green; optional annotated points
    (track id, "start" / "end" / "correction", xyz) as large plus / cross / diamond markers; optional
    arena objects (scene3d.ArenaObjects: fruit, leaves), included in the plot extent; optional
    hand-corrected tracks (step 10) as solid dark green lines over the tracks they correct."""
    gt_tracks, corrected = gt_tracks or [], corrected or []
    data = np.vstack([t.xyz for t in tracks] + [t.xyz for t in corrected] + [t[:, 1:4] for t in gt_tracks])
    extent = np.vstack([data, arena.points()]) if arena else data
    fig, ax3d, ortho, rig = _scene_figure(title, extent, cameras)
    _draw_gt_tracks(ax3d, ortho, rig, gt_tracks)
    if arena:
        _draw_arena(ax3d, ortho, rig, arena)
    if points is not None and len(points):
        # Only points inside the tracks' (padded) bounding box, so clutter doesn't set the scale.
        pad = 0.1 * np.ptp(data, axis=0).max()
        inside = np.all((points[:, 1:4] >= data.min(0) - pad) & (points[:, 1:4] <= data.max(0) + pad), axis=1)
        points = points[inside]
        ax3d.scatter(points[:, 1], points[:, 2], points[:, 3], color="0.6", s=2, alpha=0.3)
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.scatter(points[:, 1 + i], points[:, 1 + j], color="0.6", s=2, alpha=0.3)
    for t in tracks:
        col = TRACK_COLOURS[(t.id - 1) % len(TRACK_COLOURS)]
        label = f"track {t.id}: frames {t.frames[0]}-{t.frames[-1]} ({t.score} dets)"
        for ax in (ax3d, rig):
            ax.plot(*t.xyz.T, color=col, lw=PRED_TRACK_LW, label=label if ax is ax3d else None)
            ax.scatter(*t.xyz[0], color=col, marker="^", s=60, edgecolor="black")
            ax.scatter(*t.xyz[-1], color=col, marker="s", s=40, edgecolor="black")
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.plot(t.xyz[:, i], t.xyz[:, j], color=col, lw=PRED_TRACK_LW, zorder=3)
            ax.plot(t.xyz[0, i], t.xyz[0, j], "^", color=col, mec="black", ms=8)
            ax.plot(t.xyz[-1, i], t.xyz[-1, j], "s", color=col, mec="black", ms=6)
    for n, t in enumerate(corrected):
        label = f"corrected track {t.id}: frames {t.frames[0]}-{t.frames[-1]}"
        for ax in (ax3d, rig):
            ax.plot(*t.xyz.T, color=CORRECTED_COLOUR, lw=CORRECTED_LW, label=label if ax is ax3d else None, zorder=4)
            ax.scatter(*t.xyz[0], color=CORRECTED_COLOUR, marker="^", s=60, edgecolor="black")
            ax.scatter(*t.xyz[-1], color=CORRECTED_COLOUR, marker="s", s=40, edgecolor="black")
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.plot(t.xyz[:, i], t.xyz[:, j], color=CORRECTED_COLOUR, lw=CORRECTED_LW, zorder=4)
            ax.plot(t.xyz[0, i], t.xyz[0, j], "^", color=CORRECTED_COLOUR, mec="black", ms=8)
            ax.plot(t.xyz[-1, i], t.xyz[-1, j], "s", color=CORRECTED_COLOUR, mec="black", ms=6)
    if anchors:
        _draw_anchors(ax3d, ortho, rig, anchors)
    if tracks or corrected or arena:
        ax3d.legend(loc="upper left", fontsize=7, framealpha=0.8)
    if out_path is not None:   # None: only build the figure (viewing a past result)
        fig.savefig(out_path, dpi=110)
    if close:
        plt.close(fig)
    return fig


ANCHOR_COLOUR = "yellow"
ANCHOR_COLOURS = {"correction": "lime"}   # anything else: ANCHOR_COLOUR
ANCHOR_SIZE, ANCHOR_RIG_SIZE = 170 / 9, 120 / 9   # scatter sizes are areas: a third of the old width


def _draw_anchors(ax3d, ortho, rig, anchors) -> None:
    labelled = set()
    for _, kind, xyz in anchors:
        label = (f"annotated {kind}" if kind != "correction" else "hand correction") if kind not in labelled else None
        labelled.add(kind)
        kw = dict(color=ANCHOR_COLOURS.get(kind, ANCHOR_COLOUR), marker=ANCHOR_MARKERS[kind], edgecolor="black", linewidth=0.5, zorder=6)
        ax3d.scatter(*xyz, s=ANCHOR_SIZE, label=label, **kw)
        rig.scatter(*xyz, s=ANCHOR_RIG_SIZE, **kw)
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.scatter(xyz[i], xyz[j], s=ANCHOR_SIZE, **kw)
