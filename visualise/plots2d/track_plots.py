"""Frame-sized 2D track plots: ground-truth and predicted tracks drawn as lines.

Frames are grouped into sequences by the folder part of their key
("<sequence>/frame_000161"), so each sequence gets its own plot. The tracks come from
common.linking (labels and detections carry no track IDs).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common.datasets import Frame, frame_number, group_sequences  # noqa: E402
from common.linking import link_tracks  # noqa: E402

from ..common import EXTRA_COLOUR, GT_COLOUR, PRED_COLOUR, first_image, safe_name  # noqa: E402


def _centres(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def save_track_plots(frames: list[Frame], preds: dict[str, np.ndarray], out_dir: Path, model_name: str,
                     conf: float, max_distance: float = 50, max_gap: int = 5,
                     background: bool = True, dpi: int = 100, extra: dict[str, np.ndarray] | None = None,
                     extra_label: str = "", layers: list[tuple[dict[str, np.ndarray], dict, str]] = ()) -> list[Path]:
    """One frame-sized PNG per sequence with GT and predicted tracks drawn as lines.
    extra: {frame key: (N, 2) xy} points drawn as orange dots on top (legend: extra_label).
    layers: more point sets, in drawing order: ({frame key: (N, 2) xy}, matplotlib plot style, legend
    label), e.g. step 10's source track, corrected track and correction clicks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for seq, seq_frames in group_sequences(frames).items():
        w, h = seq_frames[0].width, seq_frames[0].height
        gt_pts, pr_pts = [], []
        for i, f in enumerate(seq_frames):
            fno = frame_number(f, i)
            gt_pts.append((fno, _centres(f.gt[:, 1:5])))
            p = preds.get(f.key, np.zeros((0, 6)))
            pr_pts.append((fno, _centres(p[p[:, 4] >= conf, :4])))
        gt_tracks = link_tracks(gt_pts, max_distance, max_gap)
        pr_tracks = link_tracks(pr_pts, max_distance, max_gap)

        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        img = first_image(seq_frames[0]) if background else None
        if img is not None:
            ax.imshow(img, extent=(0, w, h, 0), alpha=0.45)
        else:
            ax.set_facecolor("white")
        for tracks, colour, style, z in ((gt_tracks, GT_COLOUR, "-", 2), (pr_tracks, PRED_COLOUR, "--", 3)):
            for t in tracks:
                if len(t) > 1:
                    ax.plot(t[:, 1], t[:, 2], style, color=colour, lw=1.6, alpha=0.9, zorder=z)
                else:
                    ax.plot(t[:, 1], t[:, 2], "o", color=colour, ms=3, alpha=0.9, zorder=z)
        ex = [extra[f.key] for f in seq_frames if extra and f.key in extra]
        ex = np.vstack(ex) if ex else np.zeros((0, 2))
        if len(ex):
            ax.plot(ex[:, 0], ex[:, 1], "o", color=EXTRA_COLOUR, ms=3, alpha=0.9, zorder=4)
        drawn = []
        for z, (points, style, label) in enumerate(layers, 5):
            pts = [points[f.key] for f in seq_frames if f.key in points]
            pts = np.vstack(pts) if pts else np.zeros((0, 2))
            if len(pts):
                ax.plot(pts[:, 0], pts[:, 1], ls="", zorder=z, **style)
                drawn.append((style, label, len(pts)))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # image coordinates: origin top-left
        ax.set_xticks([])
        ax.set_yticks([])

        n_gt = sum(len(f.gt) for f in seq_frames)
        n_pr = sum(len(t) for t in pr_tracks)
        handles = [
            Line2D([], [], color=GT_COLOUR, ls="-", lw=2.5,
                   label=f"Ground truth ({len(gt_tracks)} tracks, {n_gt} boxes)"),
            Line2D([], [], color=PRED_COLOUR, ls="--", lw=2.5,
                   label=f"{model_name} (conf>={conf:g}; {len(pr_tracks)} tracks, {n_pr} boxes)"),
        ]
        if len(ex):
            handles.append(Line2D([], [], color=EXTRA_COLOUR, ls="", marker="o", ms=6,
                                  label=f"{extra_label} ({len(ex)} points)"))
        for style, label, n in drawn:
            handles.append(Line2D([], [], ls="", **{**style, "ms": max(6, style.get("ms", 6))},
                                  label=f"{label} ({n} points)"))
        ax.legend(handles=handles, loc="upper left", fontsize=12, framealpha=0.85,
                  title=f"{seq}  |  frames {len(seq_frames)}", title_fontsize=11)

        path = out_dir / f"{safe_name(seq)}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
    return saved
