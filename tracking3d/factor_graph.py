"""3D track estimation with a factor graph (gtsam).

Pipeline:
    1. Triangulated points (triangulate.py, >= 2 views per point) are linked frame to
       frame into 3D tracklets (Hungarian on 3D distance, common.linking.link_tracks).
    2. Each tracklet seeds one track: a 3D position variable X_t for every frame of a
       contiguous range, with
         - a reprojection factor for every 2D detection associated with the track
           (Huber-robust, pixel_sigma). Single-view detections count too, so the
           track continues where per-frame triangulation had to drop the point;
         - a constant-velocity factor X_{t-1} - 2 X_t + X_{t+1} ~ N(0, accel_sigma),
           which carries the track through missed detections;
         - a very weak prior on every X_t, only so the system is never singular.
    3. Data association is iterated: project the current track into every view, take
       the nearest detection within association_gate_px, re-optimise. Each iteration
       first extends the track by extend_frames at both ends (constant-velocity
       extrapolation), then trims ends that found no detections - this is how a track
       grows back to where the object entered the scene.
       Ends are trimmed back to the run of observed frames with no gap longer than
       link_max_frame_gap, so stray false positives near the ends are not absorbed.
    4. Tracks that agree where they overlap (fragments of the same object) are merged
       and re-optimised over their union. Tracks that barely move (extent below
       min_track_extent - something the detector fires on in every frame, e.g. a speck
       on the cage) are dropped; the rest are ranked by number of associated detections and the best n_tracks kept.

accel_sigma, link_max_distance and merge_distance are in world units (per frame^2 /
per frame); see configs/shared/tracking3d.example.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X
from tqdm import tqdm

from common.linking import link_tracks

from .multiview import View
from .camera import Camera
from .triangulate import Point3D

WEAK_PRIOR_SIGMA = 1e4      # world units; only keeps the system well-posed
MIN_MULTIVIEW_FRAMES = 3    # a track must be seen by >= 2 views in at least this many frames


@dataclass
class Track3D:
    id: int
    frames: np.ndarray                  # (T,) aligned frame numbers, contiguous
    xyz: np.ndarray                     # (T, 3)
    sigma: np.ndarray                   # (T,) 1-sigma position uncertainty (RMS over axes), nan if unknown
    views: np.ndarray                   # (T,) number of views with a detection on this track
    observations: list[tuple[int, int, int]] = field(default_factory=list)  # (frame, view, det index)
    reproj_px: np.ndarray = field(default_factory=lambda: np.zeros(0))       # per observation

    @property
    def extent(self) -> float:
        """Robust size of the region the track covers (5th-95th percentile box diagonal)."""
        return float(np.linalg.norm(np.percentile(self.xyz, 95, axis=0) - np.percentile(self.xyz, 5, axis=0)))

    @property
    def score(self) -> int:
        return len(self.observations)

    def summary(self) -> dict:
        steps = np.linalg.norm(np.diff(self.xyz, axis=0), axis=1)
        return {
            "id": self.id,
            "start_frame": int(self.frames[0]),
            "end_frame": int(self.frames[-1]),
            "frames": int(len(self.frames)),
            "detections": self.score,
            "frames_by_views": {int(n): int((self.views == n).sum()) for n in np.unique(self.views)},
            "entry_xyz": self.xyz[0].round(3).tolist(),
            "entry_sigma": round(float(self.sigma[0]), 3) if np.isfinite(self.sigma[0]) else None,
            "first_multiview_frame": int(self.frames[np.argmax(self.views >= 2)]),
            "exit_xyz": self.xyz[-1].round(3).tolist(),
            "path_length": round(float(steps.sum()), 3),
            "extent": round(self.extent, 3),
            "mean_speed_per_frame": round(float(steps.mean()), 3) if len(steps) else 0.0,
            "reprojection_px_median": round(float(np.median(self.reproj_px)), 3) if len(self.reproj_px) else None,
            "sigma_median": round(float(np.nanmedian(self.sigma)), 3) if np.isfinite(self.sigma).any() else None,
        }


# ----------------------------------------------------------------------------- factors

def gtsam_camera(cam: Camera) -> gtsam.PinholeCameraCal3_S2:
    """Undistorted pinhole camera (detections are undistorted on load)."""
    K = cam.K
    cal = gtsam.Cal3_S2(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2])
    pose = gtsam.Pose3(gtsam.Rot3(cam.R.T), gtsam.Point3(*cam.centre))
    return gtsam.PinholeCameraCal3_S2(pose, cal)


def _accel_error(this: gtsam.CustomFactor, values: gtsam.Values, H: list[np.ndarray] | None) -> np.ndarray:
    k0, k1, k2 = this.keys()
    if H is not None:
        eye = np.eye(3)
        H[0], H[1], H[2] = eye, -2 * eye, eye
    return values.atPoint3(k0) - 2 * values.atPoint3(k1) + values.atPoint3(k2)


def _solve(frames: np.ndarray, init: np.ndarray, obs: list[tuple[int, int, int]], views: list[View],
           gcams: list, s: dict, with_sigma: bool = False,
           anchors: list[tuple[int, int, np.ndarray]] = ()) -> tuple[np.ndarray, np.ndarray]:
    """anchors: (k, view, undistorted xy) hand annotations - plain Gaussian reprojection factors
    with sigma anchor_pixel_sigma (not robust: they are trusted, unlike detections)."""
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    pix = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(s["huber_k"]),
                                         gtsam.noiseModel.Isotropic.Sigma(2, s["pixel_sigma"]))
    acc = gtsam.noiseModel.Isotropic.Sigma(3, s["accel_sigma"])
    weak = gtsam.noiseModel.Isotropic.Sigma(3, WEAK_PRIOR_SIGMA)
    # Variables: one 3D position per frame (detected or not), each with a very weak prior.
    for k, p in enumerate(init):
        values.insert(X(k), gtsam.Point3(*p))
        graph.add(gtsam.PriorFactorPoint3(X(k), gtsam.Point3(*p), weak))
    # Reprojection factors: one per associated 2D detection (Huber-robust).
    for k, v, j in obs:
        xy = views[v].dets[int(frames[k])][j, :2]
        graph.add(gtsam.TriangulationFactorCal3_S2(gcams[v], gtsam.Point2(*xy), pix, X(k)))
    if anchors:
        anchor_noise = gtsam.noiseModel.Isotropic.Sigma(2, s["anchor_pixel_sigma"])
        for k, v, xy in anchors:
            graph.add(gtsam.TriangulationFactorCal3_S2(gcams[v], gtsam.Point2(*xy), anchor_noise, X(k)))
    # Constant-velocity factors over every 3 consecutive frames - these bridge missed detections.
    for k in range(1, len(frames) - 1):
        graph.add(gtsam.CustomFactor(acc, [X(k - 1), X(k), X(k + 1)], _accel_error))

    # Solve all factors jointly (Levenberg-Marquardt).
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
    xyz = np.array([result.atPoint3(X(k)) for k in range(len(frames))])

    sigma = np.full(len(frames), np.nan)
    if with_sigma:
        sigma = _position_sigma(graph, result, len(frames))
    return xyz, sigma


def _position_sigma(graph: gtsam.NonlinearFactorGraph, result: gtsam.Values, n: int) -> np.ndarray:
    """1-sigma (RMS over axes) of every X(k). gtsam.Marginals uses a Cholesky factorisation,
    which fails when long undetected gaps leave the system only weakly determined, so the
    dense information matrix (X(0)..X(n-1) in order) is inverted instead."""
    try:
        # Factorise once, then read every frame's covariance from it (building Marginals per
        # frame gave the same numbers but re-factorised the whole graph n times: O(n^2)).
        marginals = gtsam.Marginals(graph, result)
        return np.array([np.sqrt(np.trace(marginals.marginalCovariance(X(k))) / 3) for k in range(n)])
    except RuntimeError:
        pass
    try:
        ordering = gtsam.Ordering()
        for k in range(n):
            ordering.push_back(X(k))
        info = graph.linearize(result).hessian(ordering)[0]
        cov_diag = np.diag(np.linalg.pinv(info, hermitian=True)).reshape(n, 3)
        return np.sqrt(np.clip(cov_diag, 0, None).mean(axis=1))
    except (RuntimeError, np.linalg.LinAlgError):
        return np.full(n, np.nan)


# ----------------------------------------------------------------------------- association

def _associate(frames: np.ndarray, xyz: np.ndarray, views: list[View], gate: float,
               claimed: set[tuple[int, int, int]] | None = None) -> list[tuple[int, int, int]]:
    """Nearest detection within `gate` px of the track's projection, per frame and view.
    Detections in `claimed` ((frame, view, det index), used by another track) are skipped."""
    obs = []
    for v, view in enumerate(views):
        proj, depth = view.camera.project(xyz), view.camera.depth(xyz)
        for k, f in enumerate(frames):
            d = view.dets.get(int(f))
            if d is None or depth[k] <= 0:
                continue
            dist = np.linalg.norm(d[:, :2] - proj[k], axis=1)
            if claimed:
                dist[[j for j in range(len(d)) if (int(f), v, j) in claimed]] = np.inf
            j = int(dist.argmin())
            if dist[j] <= gate:
                obs.append((k, v, j))
    return obs


def _extend(frames: np.ndarray, xyz: np.ndarray, n: int, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad both ends by up to n frames with constant-velocity extrapolation."""
    if n <= 0 or len(frames) < 2:
        return frames, xyz
    m = min(5, len(frames) - 1)
    v0, v1 = (xyz[m] - xyz[0]) / m, (xyz[-1] - xyz[-1 - m]) / m
    pre = np.arange(max(lo, frames[0] - n), frames[0])
    post = np.arange(frames[-1] + 1, min(hi, frames[-1] + n) + 1)
    return (np.concatenate([pre, frames, post]),
            np.vstack([xyz[0] + (pre - frames[0])[:, None] * v0, xyz,
                       xyz[-1] + (post - frames[-1])[:, None] * v1]))


def _main_run(ks: list[int], max_gap: int) -> tuple[int, int]:
    """First/last index of the run of observed frames (gaps <= max_gap) holding the most
    observations. Drops stray detections picked up by the extension at either end."""
    counts = np.bincount(ks)
    observed = np.flatnonzero(counts)
    breaks = np.flatnonzero(np.diff(observed) > max_gap)
    runs = np.split(observed, breaks + 1)
    best = max(runs, key=lambda r: counts[r].sum())
    return int(best[0]), int(best[-1])


def optimise_track(seed: np.ndarray, views: list[View], gcams: list, s: dict,
                   frame_range: tuple[int, int], report=None) -> Track3D | None:
    """seed: (K, 4) frame, x, y, z from a tracklet. Returns the optimised track or None.
    report: optional callable(str), told what is happening (for the progress bar)."""
    frames = np.arange(int(seed[0, 0]), int(seed[-1, 0]) + 1)
    xyz = np.stack([np.interp(frames, seed[:, 0], seed[:, i]) for i in (1, 2, 3)], axis=1)
    obs: list = []
    iterations = max(1, int(s["association_iterations"]))
    for it in range(iterations):
        # Extend both ends at the current velocity.
        frames, xyz = _extend(frames, xyz, int(s["extend_frames"]), *frame_range)
        # Associate: nearest detection in every view (incl. ones triangulation discarded).
        obs = _associate(frames, xyz, views, s["association_gate_px"])
        if not obs:
            return None
        # Trim the ends back to where detections were actually found.
        k0, k1 = _main_run([o[0] for o in obs], int(s["link_max_frame_gap"]))
        frames, xyz = frames[k0:k1 + 1], xyz[k0:k1 + 1]
        obs = [(k - k0, v, j) for k, v, j in obs if k0 <= k <= k1]
        if report:
            last = it == iterations - 1
            report(f"round {it + 1}/{iterations}: solving {len(frames)} frames, {len(obs)} dets"
                   + (" + sigma" if last else ""))
        # Solve; per-frame 3D uncertainty only on the last round.
        xyz, sigma = _solve(frames, xyz, obs, views, gcams, s, with_sigma=(it == iterations - 1))

    n_views = np.bincount([k for k, _, _ in obs], minlength=len(frames))
    if (n_views >= 2).sum() < MIN_MULTIVIEW_FRAMES:
        return None
    reproj = np.array([np.linalg.norm(views[v].camera.project(xyz[k])[0] - views[v].dets[int(frames[k])][j, :2])
                       for k, v, j in obs])
    return Track3D(id=-1, frames=frames, xyz=xyz, sigma=sigma, views=n_views,
                   observations=[(int(frames[k]), v, j) for k, v, j in obs], reproj_px=reproj)


# ----------------------------------------------------------------------------- tracks

def _same_object(a: Track3D, b: Track3D, dist: float, min_common: int = 5) -> bool:
    """True if the tracks share >= min_common frames and are within `dist` in >= 80% of them."""
    common, ia, ib = np.intersect1d(a.frames, b.frames, return_indices=True)
    if len(common) < min(min_common, len(a.frames), len(b.frames)):
        return False
    return (np.linalg.norm(a.xyz[ia] - b.xyz[ib], axis=1) <= dist).mean() >= 0.8


def _merged_seed(a: Track3D, b: Track3D) -> np.ndarray:
    """(K, 4) frame, x, y, z over both tracks' frames, averaged where they overlap."""
    frames = np.union1d(a.frames, b.frames)
    acc, n = np.zeros((len(frames), 3)), np.zeros(len(frames))
    for t in (a, b):
        idx = np.searchsorted(frames, t.frames)
        acc[idx] += t.xyz
        n[idx] += 1
    return np.hstack([frames[:, None], acc / n[:, None]])


def link_points(points: dict[int, list[Point3D]], s: dict) -> list[np.ndarray]:
    """Frame-to-frame 3D tracklets, (K, 4) frame, x, y, z, longest first."""
    seq = [(f, np.array([p.xyz for p in pts])) for f, pts in sorted(points.items())]
    tracklets = link_tracks(seq, s["link_max_distance"], int(s["link_max_frame_gap"]))
    tracklets = [t for t in tracklets if len(t) >= int(s["min_tracklet_frames"])]
    return sorted(tracklets, key=len, reverse=True)


def build_tracks(views: list[View], points: dict[int, list[Point3D]], s: dict) -> tuple[list[Track3D], list[np.ndarray]]:
    """Returns (tracks ranked best first, the tracklets that seeded them)."""
    # Step 3: link triangulated points over time into seed tracklets.
    tracklets = link_points(points, s)
    gcams = [gtsam_camera(v.camera) for v in views]
    all_frames = [f for v in views for f in v.dets]
    frame_range = (min(all_frames), max(all_frames)) if all_frames else (0, 0)

    tracks: list[Track3D] = []
    # Step 4: refine each tracklet with the factor graph (step 5's skip / merge happen inline).
    n_skipped = n_merges = 0
    bar = tqdm(tracklets, desc="Factor graph", unit="tracklet", dynamic_ncols=True)

    def status(what: str, msg: str):
        # Live detail after the bar: totals so far, the current step, then what the solver is doing.
        bar.set_postfix_str(f"{len(tracks)} tracks, {n_skipped} skipped, {n_merges} merges | {what} | {msg}")

    for seed in bar:
        if any(_covered(seed, t, s["merge_distance"]) for t in tracks):
            n_skipped += 1
            continue  # a fragment of a track that is already estimated
        what = f"seed {len(seed)}f @ {int(seed[0, 0])}-{int(seed[-1, 0])}"
        tr = optimise_track(seed, views, gcams, s, frame_range, report=lambda m: status(what, m))
        if tr is None or len(tr.frames) < int(s["min_track_frames"]):
            continue
        # Fragments of one object (e.g. a tracklet broken by missed detections) overlap once
        # extended: merge them and re-optimise over the union, until nothing else overlaps.
        while (j := next((i for i, t in enumerate(tracks) if _same_object(tr, t, s["merge_distance"])),
                         None)) is not None:
            other = tracks.pop(j)
            n_merges += 1
            what = f"merging {len(tr.frames)}f + {len(other.frames)}f"
            merged = optimise_track(_merged_seed(tr, other), views, gcams, s, frame_range,
                                    report=lambda m: status(what, m))
            tr = merged if merged is not None and merged.score >= max(tr.score, other.score) \
                else max(tr, other, key=lambda t: t.score)
        tracks.append(tr)

    # Step 5: drop stationary tracks, rank by support, keep the best n_tracks.
    tracks = [t for t in tracks if t.extent >= float(s["min_track_extent"])]  # stationary false positives
    tracks.sort(key=lambda t: t.score, reverse=True)
    if s.get("n_tracks"):
        tracks = tracks[:int(s["n_tracks"])]
    for i, t in enumerate(tracks, 1):
        t.id = i
    return tracks, tracklets


def _covered(seed: np.ndarray, track: Track3D, dist: float) -> bool:
    common, ia, ib = np.intersect1d(seed[:, 0].astype(int), track.frames, return_indices=True)
    if len(common) < 0.8 * len(seed):
        return False
    return (np.linalg.norm(seed[ia, 1:] - track.xyz[ib], axis=1) <= dist).mean() > 0.8
