"""Output files shared by main_tracks3d.py and main_tracks3d_guided.py (the summary text itself is
laid out by visualise/metrics/tracking.py)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from visualise.metrics.tracking import format_summary

from .factor_graph import Track3D


def camera_config(session, cam_cfg: dict) -> dict:
    """config["cameras"] as written in cameras.json, cut down to the views this session used."""
    return {**{k: v for k, v in cam_cfg.items() if k not in ("_readme", "views")},
            "views": {v.name: cam_cfg["views"][v.name] for v in session.views}}


def write_track_outputs(session, tracks: list[Track3D], s: dict, out_dir: Path, heading: str,
                        extra_lines: list[str] = (), extra_json: dict | None = None,
                        params: dict | None = None, cam_cfg: dict | None = None) -> str:
    """summary.txt, tracks.json, tracks_3d.csv and params.json (the settings this run used, its
    views' entries from cam_cfg = config["cameras"], plus `params`). Returns the summary text."""
    summaries = [t.summary() for t in tracks]
    report = format_summary(session, tracks, summaries, s, heading)
    if extra_lines:
        report += "\n\n" + "\n".join(extra_lines)
    (out_dir / "summary.txt").write_text(report + "\n")
    (out_dir / "tracks.json").write_text(json.dumps({"run": session.run.name, "detections": session.label,
                                                     "settings": s, "tracks": summaries, **(extra_json or {})},
                                                    indent=2, default=str))
    (out_dir / "params.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"), "run": session.run.name,
        "detections": session.label, "views": {v.name: v.input_name for v in session.views},
        "frame_offsets": {v.name: v.camera.frame_offset for v in session.views},
        **({"cameras": camera_config(session, cam_cfg)} if cam_cfg else {}),
        **(params or {}), "tracking3d": s}, indent=2, default=str))
    with open(out_dir / "tracks_3d.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "frame", "x", "y", "z", "sigma", "n_views"])
        for t in tracks:
            for f, p, sg, nv in zip(t.frames, t.xyz, t.sigma, t.views):
                w.writerow([t.id, int(f), *p.round(4), round(float(sg), 4), int(nv)])
    return report
