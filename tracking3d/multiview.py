"""Treat a previous batch evaluation run as one multi-camera recording session.

A batch run (main_evaluate2d.py with a "batch" dataset) holds one folder per input
video. Each camera in config["cameras"]["views"] is paired with the input whose
name contains the camera's name (or its "match" string), e.g. camera "344-above"
<-> input "260723-123554_344-above_session0_...". Frame N of every view is assumed
to be the same instant, after adding each camera's "frame_offset".

Detections come from the run's saved predictions (one YOLO .txt per frame) or,
with source="gt", from the ground-truth labels the run was evaluated against.
Box centres are used as the 2D observation of each object.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.datasets import IMAGE_EXTS, SPLIT_LISTS, Frame, resolve_label_dir
from common.yolo_format import read_yolo_labels, read_yolo_rows
from detection.predictions import PRED_DIR
from common.linking import link_tracks

from .camera import Camera, camera_from_config
from .triangulate import MatchSettings, triangulate_session

FRAME_RE = re.compile(r"(\d+)$")


@dataclass
class RunInfo:
    path: Path
    inputs: list[str]                       # input folders that hold predictions
    models: list[str]                       # models present for every input
    metadata: dict = field(default_factory=dict)
    matches: dict[str, str] = field(default_factory=dict)   # camera -> input
    problems: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def usable(self) -> bool:
        return len(self.matches) >= 2 and bool(self.models) and not self.problems


@dataclass
class View:
    camera: Camera
    input_name: str
    size: tuple[int, int]
    raw: dict[int, np.ndarray]              # frame -> (N, 3) x, y, conf in raw pixels
    dets: dict[int, np.ndarray]             # frame -> (N, 3) x, y, conf undistorted
    background: Frame | None = None         # first frame, for plots
    source: dict = field(default_factory=dict)  # the input's dataset config (video / images, start_index)

    @property
    def name(self) -> str:
        return self.camera.name


# ----------------------------------------------------------------------------- runs

def camera_match_string(name: str, cfg: dict) -> str:
    return cfg.get("match") or name


def match_cameras(inputs: list[str], cam_cfg: dict) -> tuple[dict[str, str], list[str]]:
    matches, problems = {}, []
    for name, cfg in cam_cfg.get("views", {}).items():
        s = camera_match_string(name, cfg)
        hits = [i for i in inputs if s in i]
        if len(hits) == 1:
            matches[name] = hits[0]
        elif len(hits) > 1:
            problems.append(f"camera '{name}' matches several inputs: {hits}")
    return matches, problems


def scan_run(run_dir: Path, cam_cfg: dict) -> RunInfo:
    inputs, model_sets = [], []
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        models = {m.name for m in d.iterdir() if (m / PRED_DIR).is_dir()}
        if models:
            inputs.append(d.name)
            model_sets.append(models)
    models = sorted(set.intersection(*model_sets)) if model_sets else []
    meta_path = run_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    matches, problems = match_cameras(inputs, cam_cfg)
    return RunInfo(run_dir, inputs, models, metadata, matches, problems)


def list_runs(output_root: Path, cam_cfg: dict) -> list[RunInfo]:
    if not output_root.is_dir():
        return []
    return [scan_run(d, cam_cfg) for d in sorted(output_root.glob("run_*")) if d.is_dir()]


def _ask(prompt: str, n: int, default: int) -> int:
    if not sys.stdin.isatty():
        return default
    while True:
        ans = input(f"{prompt} [1-{n}, Enter = {default + 1}]: ").strip()
        if not ans:
            return default
        if ans.isdigit() and 1 <= int(ans) <= n:
            return int(ans) - 1
        print("  Please enter a number from the list.")


def choose_run(output_root: Path, cam_cfg: dict, run: str | None = None) -> RunInfo:
    """Pick a run by name/path, or list the runs and ask in the terminal."""
    if run:
        p = Path(run)
        p = p if p.is_dir() else output_root / run
        if not p.is_dir():
            raise SystemExit(f"Run not found: {run}")
        return scan_run(p, cam_cfg)

    runs = list_runs(output_root, cam_cfg)
    if not runs:
        raise SystemExit(f"No runs in {output_root}. Run main_evaluate2d.py with a batch config, or main_inference2d.py, first.")
    n_cams = len(cam_cfg.get("views", {}))
    width = max(len(r.name) for r in runs) + 2
    print(f"\nEvaluation runs in {output_root}  (each run = one recording session, views = input folders)")
    print(f"  {'#':>3}  {'run':<{width}}{'views':>6}{'cameras':>9}  models")
    for i, r in enumerate(runs, 1):
        flag = "" if r.usable else "   (not usable: " + (
            "; ".join(r.problems) or ("no models" if not r.models else "< 2 cameras matched")) + ")"
        print(f"  {i:>3}  {r.name:<{width}}{len(r.inputs):>6}{f'{len(r.matches)}/{n_cams}':>9}  "
              f"{', '.join(r.models) or '-'}{flag}")
    usable = [i for i, r in enumerate(runs) if r.usable]
    if not usable:
        raise SystemExit("No run has >= 2 inputs matching the cameras in config['cameras']['views'].")
    idx = _ask("Choose a run", len(runs), usable[-1])
    chosen = runs[idx]
    if not chosen.usable:
        raise SystemExit(f"{chosen.name} cannot be used: {chosen.problems or 'fewer than 2 cameras matched'}")
    return chosen


def choose_model(run: RunInfo, model: str | None = None) -> str:
    if model:
        if model not in run.models:
            raise SystemExit(f"Model '{model}' not in {run.name}. Available: {run.models}")
        return model
    if len(run.models) == 1:
        return run.models[0]
    print("\nModels in this run:")
    for i, m in enumerate(run.models, 1):
        print(f"  {i:>3}  {m}")
    return run.models[_ask("Choose a model", len(run.models), 0)]


def describe_views(run: RunInfo) -> None:
    print(f"\nSession {run.name}: {len(run.matches)} views")
    for cam, inp in run.matches.items():
        print(f"  {cam:<14} <- {inp}")
    for inp in sorted(set(run.inputs) - set(run.matches.values())):
        print(f"  {'(unused)':<14}    {inp}")


# ----------------------------------------------------------------------------- loading

def _frame_no(stem: str) -> int | None:
    m = FRAME_RE.search(stem)
    return int(m.group(1)) if m else None


def _read_centres(path: Path, w: int, h: int, with_conf: bool) -> np.ndarray:
    a = read_yolo_rows(path)
    conf = a[:, 5] if with_conf else np.ones(len(a))
    return np.stack([a[:, 1] * w, a[:, 2] * h, conf], axis=1)


def _input_meta(run: RunInfo, input_name: str) -> dict:
    return run.metadata.get("inputs", {}).get(input_name, {}).get("dataset", {})


def _background(source: dict, size: tuple[int, int]) -> Frame | None:
    empty = np.zeros((0, 5))
    if source.get("video") and Path(source["video"]).is_file():
        return Frame("background", size[0], size[1], empty, video_path=source["video"], video_index=0)
    if source.get("images") and Path(source["images"]).is_dir():
        imgs = sorted(p for p in Path(source["images"]).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if imgs:
            return Frame("background", size[0], size[1], empty, image_path=str(imgs[0]))
    return None


def load_views(run: RunInfo, model: str, cam_cfg: dict, source: str = "pred",
               conf_threshold: float | None = None) -> list[View]:
    """One View per matched camera, in config order, with detections keyed by aligned frame number."""
    views = []
    for cam_name, inp in run.matches.items():
        meta = _input_meta(run, inp)
        cfg = cam_cfg["views"][cam_name]
        sizes = meta.get("image_sizes") or ([cfg["image_size"]] if cfg.get("image_size") else [])
        if not sizes:
            raise SystemExit(f"Unknown image size for {inp}: no metadata.json in the run and no "
                             f"image_size for camera '{cam_name}'.")
        size = tuple(sizes[0])
        try:
            camera = camera_from_config(cam_name, cfg, size, float(cam_cfg.get("t_scale", 1.0)))
        except ValueError as e:
            raise SystemExit(f"Camera calibration problem: {e}") from None

        if source == "gt":
            label_src = meta.get("source", {}).get("labels")
            if not label_src:
                raise SystemExit(f"metadata.json of {run.name} has no labels path for {inp}")
            files = [p for p in resolve_label_dir(label_src).glob("*.txt") if p.name not in SPLIT_LISTS]
        else:
            files = [p for p in (run.path / inp / model / PRED_DIR).rglob("*.txt")]

        raw, dets = {}, {}
        for p in tqdm(files, desc=f"Loading {cam_name}", unit="file", dynamic_ncols=True, leave=False):
            fno = _frame_no(p.stem)
            if fno is None:
                continue
            if source == "gt":
                b = read_yolo_labels(p, *size)
                c = np.stack([(b[:, 1] + b[:, 3]) / 2, (b[:, 2] + b[:, 4]) / 2, np.ones(len(b))], axis=1)
            else:
                c = _read_centres(p, *size, with_conf=True)
            if conf_threshold is not None:
                c = c[c[:, 2] >= conf_threshold]
            if len(c):
                f = fno + camera.frame_offset   # align: frame f is the same instant in every view
                raw[f] = c
                dets[f] = np.hstack([camera.undistort(c[:, :2]), c[:, 2:3]])   # to ideal pinhole pixels
        src = meta.get("source", {})
        views.append(View(camera, inp, size, raw, dets, _background(src, size), src))
    return views


def remove_static_detections(view: View, radius: float, min_fraction: float) -> int:
    """Drop detections at a spot the detector fires on in >= min_fraction of the view's frames
    (within radius px) - background clutter (specks, cage, fruit features), not a moving animal.
    Left in, matching spots in several views triangulate to a fixed 3D point every frame and
    tracks lock onto it. Returns the number of detections removed."""
    frames = sorted(view.dets)
    if not frames:
        return 0
    pts = np.vstack([view.dets[f][:, :2] for f in frames])
    occupied = np.zeros(len(pts), int)   # frames with a detection within radius of each detection
    for f in tqdm(frames, desc=f"Clutter filter {view.name}", unit="frame", dynamic_ncols=True, leave=False):
        occupied += (np.linalg.norm(pts[:, None] - view.dets[f][None, :, :2], axis=2) <= radius).any(axis=1)
    static = occupied >= min_fraction * (frames[-1] - frames[0] + 1)
    splits = np.cumsum([len(view.dets[f]) for f in frames])[:-1]
    for f, drop in zip(frames, np.split(static, splits)):
        if drop.all():
            del view.dets[f], view.raw[f]
        elif drop.any():
            view.dets[f], view.raw[f] = view.dets[f][~drop], view.raw[f][~drop]
    return int(static.sum())


def views_summary(views: list[View]) -> list[str]:
    lines = []
    for v in views:
        n = sum(len(d) for d in v.dets.values())
        frames = sorted(v.dets)
        span = f"frames {frames[0]}-{frames[-1]}" if frames else "no detections"
        lines.append(f"  {v.name:<14} {v.size[0]}x{v.size[1]}  {n:>6} detections in {len(frames):>5} frames ({span})")
    return lines


def tracking_settings(config: dict) -> dict:
    """config["tracking3d"] with defaults filled in."""
    defaults = {
        "conf_threshold": 0.25,
        "max_epipolar_px": 15.0,
        "max_reprojection_px": 10.0,
        "link_max_distance": 20.0,
        "link_max_frame_gap": 5,
        "min_tracklet_frames": 5,
        "n_tracks": None,
        "min_track_frames": 10,
        "min_track_extent": 5.0,
        "pixel_sigma": 2.0,
        "huber_k": 2.0,
        "accel_sigma": 3.0,
        "association_gate_px": 25.0,
        "association_iterations": 3,
        "extend_frames": 15,
        "merge_distance": 10.0,
        "static_radius_px": 6.0,
        "static_min_fraction": 0.3,
        "eval_match_distance": 20.0,
        "eval_match_px": 10.0,
        "evaluate_2d": True,
        "backproj_score": 0.5,
        "visualise_backproj_graph": True,
        "visualise_backproj_video": True,
        "anchor_pixel_sigma": 3.0,
        "anchor_gate_px": 40.0,
        "anchor_frame_window": 30,
        "anchor_max_px_per_frame": 40.0,
        "guided_iterations": 5,
    }
    user = {k: v for k, v in config.get("tracking3d", {}).items() if not k.startswith("_")}
    if "visualise_backproj" in user:   # old name of visualise_backproj_graph
        user.setdefault("visualise_backproj_graph", user.pop("visualise_backproj"))
    return {**defaults, **user}


# ----------------------------------------------------------------------------- session

@dataclass
class Session:
    run: RunInfo
    model: str
    source: str                             # "pred" or "gt"
    views: list[View]
    settings: dict

    @property
    def label(self) -> str:
        return "ground_truth" if self.source == "gt" else self.model

    def out_dir(self, kind: str) -> Path:
        d = self.run.path / "3d" / self.label / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def factor_graph_dir(self, mode: str) -> Path:
        """<run>/3d/<label>/factor_graph/<mode>_<datetime>/: a new folder for every factor-graph run,
        so runs with different settings (see its params.json) can be compared."""
        d = self.run.path / "3d" / self.label / "factor_graph" / f"{mode}_{datetime.now():%Y%m%d-%H%M%S}"
        d.mkdir(parents=True, exist_ok=True)
        return d


def open_session(config: dict, run: str | None, model: str | None, source: str,
                 conf_threshold: float | None = None) -> Session:
    """Pick the run (interactive unless given), the model, and load every camera's detections."""
    cam_cfg = config.get("cameras") or {}
    if not cam_cfg.get("views"):
        raise SystemExit("config has no 'cameras' -> 'views'. See configs/shared/cameras.example.json.")
    settings = tracking_settings(config)
    if conf_threshold is not None:
        settings["conf_threshold"] = conf_threshold

    info = choose_run(Path(config["output_dir"]), cam_cfg, run)
    describe_views(info)
    if len(info.matches) < 2:
        raise SystemExit("Need at least 2 matched camera views to triangulate.")
    chosen = choose_model(info, model) if source == "pred" else (model or "-")
    views = load_views(info, chosen, cam_cfg, source,
                       settings["conf_threshold"] if source == "pred" else None)
    print(f"\nDetections ({'ground truth' if source == 'gt' else chosen}"
          f"{'' if source == 'gt' or settings['conf_threshold'] is None else f', conf >= {settings['conf_threshold']}'}):")
    print("\n".join(views_summary(views)))
    # Clutter mask: drop detections the detector fires on at the same spot in most frames.
    if source == "pred" and settings["static_min_fraction"]:
        removed = [remove_static_detections(v, settings["static_radius_px"], settings["static_min_fraction"])
                   for v in views]
        print(f"\nStatic clutter removed (same spot within {settings['static_radius_px']} px in >= "
              f"{settings['static_min_fraction']:.0%} of frames): "
              + ", ".join(f"{v.name} {n}" for v, n in zip(views, removed)))
    return Session(info, chosen, source, views, settings)


def ground_truth_tracks(session: Session, cam_cfg: dict) -> list[np.ndarray] | None:
    """3D ground-truth tracks for plot overlays: the run's GT labels matched across views
    (Hungarian on epipolar distance) and triangulated, then linked frame to frame (Hungarian
    on 3D distance). No factor graph - labels are already correct, so this is only needed to
    give them identities over time. Frames labelled in a single view are lost (small gaps).
    Returns (K, 4) frame, x, y, z arrays, or None if the run's labels can't be loaded."""
    s = session.settings
    try:
        views = load_views(session.run, session.model, cam_cfg, "gt")
    except (SystemExit, OSError, ValueError) as e:
        print(f"  Ground-truth overlay skipped: {e}")
        return None
    points = triangulate_session([v.camera for v in views], [v.dets for v in views],
                                 MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
    seq = [(f, np.array([p.xyz for p in pts])) for f, pts in sorted(points.items())]
    tracks = [t for t in link_tracks(seq, s["link_max_distance"], int(s["link_max_frame_gap"])) if len(t) >= 2]
    print(f"Ground-truth overlay: {len(tracks)} tracks from {sum(len(p) for p in points.values())} "
          f"triangulated labels")
    return tracks
