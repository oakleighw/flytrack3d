"""Find and load the 3D results saved by main_points3d.py, main_tracks3d.py and
main_tracks3d_guided.py, so they can be plotted again without recomputing them
(main_visualise_past_3d.py).

Under <run>/3d/<model or ground_truth>/:
    points/points.csv                             kind "points"
    factor_graph/tracks_<datetime>/tracks_3d.csv  kind "tracks"  (unguided)
    factor_graph/guided_<datetime>/tracks_3d.csv  kind "guided"
plus the folders older versions wrote (triangulated/, tracks/, tracks_guided/, factor_graph/tracks.csv).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .multiview import _ask, run_folders

KINDS = {"points": "triangulated points  (main_points3d.py)",
         "tracks": "unguided tracks      (main_tracks3d.py)",
         "guided": "guided tracks        (main_tracks3d_guided.py)"}
# kind -> globs of its csv, relative to <run>/3d/<label>/; the current layout first, then older ones.
PATTERNS = {"points": ("points/points.csv", "triangulated/points.csv"),
            "tracks": ("factor_graph/tracks_*/tracks_3d.csv", "factor_graph/tracks.csv", "tracks/tracks.csv"),
            "guided": ("factor_graph/guided_*/tracks_3d.csv", "tracks_guided/tracks.csv")}


@dataclass
class SavedTrack:
    """What plot_tracks_3d needs of a Track3D, read back from tracks_3d.csv."""
    id: int
    frames: np.ndarray   # (T,) aligned frame numbers
    xyz: np.ndarray      # (T, 3)
    score: int           # number of detections on the track
    n_views: np.ndarray | None = None   # (T,) cameras whose detections the track was fitted to, per frame


@dataclass
class PastResult:
    kind: str     # "points", "tracks" or "guided"
    label: str    # model name, or "ground_truth"
    csv: Path

    @property
    def path(self) -> Path:
        return self.csv.parent

    @property
    def name(self) -> str:
        """Folder relative to <run>/3d/<label>/, e.g. factor_graph/guided_20261012-143417."""
        return self.path.relative_to(self.csv.parents[self._depth]).as_posix()

    @property
    def _depth(self) -> int:
        return 2 if self.path.parent.name == "factor_graph" else 1

    @property
    def created(self) -> datetime:
        return datetime.fromtimestamp(self.csv.stat().st_mtime)

    def _json(self, name: str) -> dict:
        p = self.path / name
        return json.loads(p.read_text()) if p.is_file() else {}

    def describe(self) -> str:
        if self.kind == "points":
            with open(self.csv) as fh:
                return f"{max(sum(1 for _ in fh) - 1, 0)} points"
        tracks = self._json("tracks.json").get("tracks")
        return f"{len(tracks)} track(s)" if tracks is not None else ""

    def saved_settings(self) -> tuple[dict, dict]:
        """(tracking3d settings, frame offset per camera) that the result was made with;
        each empty if it was not saved (points, older results)."""
        params = self._json("params.json")
        return (params.get("tracking3d") or self._json("tracks.json").get("settings") or {},
                params.get("frame_offsets") or {})

    def load_points(self) -> np.ndarray:
        """(N, 5) frame, x, y, z, n_views."""
        return np.loadtxt(self.csv, delimiter=",", skiprows=1, usecols=range(5), ndmin=2)

    def load_tracks(self) -> list[SavedTrack]:
        rows = np.loadtxt(self.csv, delimiter=",", skiprows=1, ndmin=2)   # track, frame, x, y, z, sigma, n_views
        tracks = []
        for tid in np.unique(rows[:, 0]):
            r = rows[rows[:, 0] == tid]
            tracks.append(SavedTrack(int(tid), r[:, 1].astype(int), r[:, 2:5], int(r[:, 6].sum()),
                                     r[:, 6].astype(int)))
        return tracks

    def anchors(self, tracks: list[SavedTrack], frame_offsets: dict[str, int]) -> list[tuple[int, str, np.ndarray]]:
        """Annotated start / end points of a guided result as (track id, kind, xyz on the track)."""
        by_id = {t.id: t for t in tracks}
        out = []
        for tid, obj in (self._json("tracks.json").get("objects") or {}).items():
            t = by_id.get(int(tid))
            for a in obj.get("annotations", []) if t else []:
                k = int(a["frame"]) + int(frame_offsets.get(a["view"], 0)) - int(t.frames[0])
                if 0 <= k < len(t.xyz):
                    out.append((t.id, a["kind"], t.xyz[k]))
        return out


def find_results(run_dir: Path) -> list[PastResult]:
    """Every saved 3D result of a run, oldest first."""
    results = []
    for label_dir in sorted(p for p in (run_dir / "3d").glob("*") if p.is_dir()):
        for kind, patterns in PATTERNS.items():
            results += [PastResult(kind, label_dir.name, csv) for pat in patterns for csv in label_dir.glob(pat)]
    return sorted(results, key=lambda r: r.created)


def choose_past_run(output_root: Path, run: str | None = None) -> tuple[Path, list[PastResult]]:
    """Pick a run by name/path, or list the runs that have saved 3D results and ask in the terminal."""
    if run:
        p = Path(run)
        p = p if p.is_dir() else output_root / run
        if not p.is_dir():
            raise SystemExit(f"Run not found: {run}")
        results = find_results(p)
        if not results:
            raise SystemExit(f"{p.name} has no saved 3D results. Run main_points3d.py or main_tracks3d*.py on it first.")
        return p, results

    runs = [(d, find_results(d)) for d in run_folders(output_root)]
    runs = [(d, r) for d, r in runs if r]
    if not runs:
        raise SystemExit(f"No run in {output_root} has saved 3D results. "
                         f"Run main_points3d.py or main_tracks3d*.py first.")
    width = max(len(d.name) for d, _ in runs) + 2
    print(f"\nRuns with saved 3D results in {output_root}")
    print(f"  {'#':>3}  {'run':<{width}}{'points':>7}{'unguided':>10}{'guided':>8}  latest")
    for i, (d, results) in enumerate(runs, 1):
        n = {k: sum(r.kind == k for r in results) for k in KINDS}
        print(f"  {i:>3}  {d.name:<{width}}{n['points']:>7}{n['tracks']:>10}{n['guided']:>8}  "
              f"{results[-1].created:%Y-%m-%d %H:%M}")
    latest = max(range(len(runs)), key=lambda i: runs[i][1][-1].created)
    return runs[_ask("Choose a run", len(runs), latest)]


def choose_result(results: list[PastResult], kind: str | None = None, label: str | None = None,
                  name: str | None = None) -> PastResult:
    """Narrow a run's results down to one: by detections (model / ground truth) if it has several,
    then by kind, then by the past result itself. Anything not given is asked in the terminal."""
    labels = sorted({r.label for r in results})
    if label:
        if label not in labels:
            raise SystemExit(f"No saved 3D results for '{label}'. Available: {labels}")
    elif len(labels) > 1:
        print("\nDetections the 3D results were made from:")
        for i, lab in enumerate(labels, 1):
            print(f"  {i:>3}  {lab}")
        label = labels[_ask("Choose the detections", len(labels), 0)]
    else:
        label = labels[0]
    results = [r for r in results if r.label == label]

    kinds = [k for k in KINDS if any(r.kind == k for r in results)]
    if kind:
        if kind not in kinds:
            raise SystemExit(f"No saved '{kind}' result for {label}. Available: {kinds}")
    else:
        print(f"\nWhat to view ({label}):")
        for i, k in enumerate(kinds, 1):
            print(f"  {i:>3}  {KINDS[k]}   {sum(r.kind == k for r in results)} saved")
        kind = kinds[_ask("Choose what to view", len(kinds), len(kinds) - 1)]
    results = [r for r in results if r.kind == kind]

    if name and name != "latest":
        hits = [r for r in results if name in (r.name, r.path.name)]
        if not hits:
            raise SystemExit(f"No saved result '{name}'. Available: {[r.name for r in results]}")
        return hits[0]
    if name == "latest" or len(results) == 1:
        return results[-1]
    width = max(len(r.name) for r in results) + 2
    print(f"\nSaved {KINDS[kind].split('  ')[0]}, oldest first:")
    for i, r in enumerate(results, 1):
        print(f"  {i:>3}  {r.name:<{width}}{r.created:%Y-%m-%d %H:%M}   {r.describe()}")
    return results[_ask("Choose a result", len(results), len(results) - 1)]
