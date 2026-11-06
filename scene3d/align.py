"""Put the 3D results of many runs into one shared frame, centred on the fruit, so they can be
drawn together (main_visualise_all.py).

Each run's world is the frame of its calibration's reference camera, and which camera that is
(and where the rig stood) changes between recordings. So the raw coordinates of two runs are
not comparable, and averaging them would not fix that: they differ by a rotation as well as a
shift. What every run does share is the arena. The fruit is fitted in every run (its cached
<run>/3d/fruit/fruit.json), and with the cameras' image-up direction it gives the wall frame
of scene3d/geometry.py. Each run is moved into that frame by a rigid transform (rotation and
shift, no scaling, so distances stay in world units):

    origin  the fruit's base on the wall
    x       right along the wall, as the cameras see it
    y       into the wall (the cameras are at negative y, the wall is y = 0)
    z       up the wall (the cameras' mean image-up, laid into the wall)

which is right-handed, so nothing is mirrored. The leaves and the arena frame are moved with
the same transform. The fruit, leaves and calibration are all read from the caches; no video
is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking3d.camera import Camera, camera_from_config

from .arena.mesh.reconstruct import ArenaFrame, build_frame
from .fruit import Fruit
from .geometry import wall_of
from .leaves import Leaves

# The fruit's normal (out of the wall, towards the cameras) in the shared frame.
FRUIT_NORMAL = np.array([0.0, -1.0, 0.0])


@dataclass
class RunAlignment:
    run: Path
    origin: np.ndarray          # (3,) the fruit's base, in the run's world
    rotation: np.ndarray        # (3, 3) rows: the shared x, y, z axes, in the run's world
    cameras: list[Camera]       # the calibration the fruit was fitted with
    fruit: Fruit                # in the run's world
    leaves: Leaves | None       # in the run's world

    def apply(self, X: np.ndarray) -> np.ndarray:
        """(N, 3) points of the run's world -> (N, 3) shared frame."""
        return (np.asarray(X, float).reshape(-1, 3) - self.origin) @ self.rotation.T

    def to_json(self) -> dict:
        return {"run": str(self.run), "origin": self.origin.tolist(), "rotation": self.rotation.tolist(),
                "fruit_radius": self.fruit.radius, "fruit_depth": self.fruit.depth,
                "reference_camera": next((c.name for c in self.cameras if np.allclose(c.t, 0) and
                                          np.allclose(c.R, np.eye(3))), None)}

    # The arena objects, moved into the shared frame.
    def leaves_aligned(self) -> Leaves | None:
        if self.leaves is None:
            return None
        return Leaves(self.apply(self.leaves.outline).tolist(), self.leaves.area, self.leaves.iou)

    def frame_aligned(self, mesh: dict) -> ArenaFrame:
        f = build_frame(self.fruit, self.cameras, mesh)
        return ArenaFrame(self.apply(f.back), self.apply(f.front), self.rotation @ f.up)


def _cached(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _cache_cameras(key: dict) -> list[Camera]:
    """The calibration saved in an arena cache's key (scene3d/settings.py cache_key)."""
    return [Camera(name, np.asarray(c["K"], float), np.asarray(c["dist"], float), np.asarray(c["R"], float),
                   np.asarray(c["t"], float)) for name, c in key["cameras"].items()]


def align_run(run_dir: Path) -> RunAlignment | str:
    """The run's transform into the shared frame, or the reason it has none (a string)."""
    run_dir = Path(run_dir)
    saved = _cached(run_dir / "3d" / "fruit" / "fruit.json")
    if saved is None:
        return "no fruit cache (3d/fruit/fruit.json): run a step 7 / 8 script with the arena on it first"
    if not saved.get("result"):
        return "the fruit was not found in this run (see 3d/fruit/hsv/)"
    fruit = Fruit.from_json(saved["result"])
    cams = _cache_cameras(saved["key"])
    wall = wall_of(fruit, cams)
    rotation = np.array([wall.right, -wall.normal, wall.up])

    leaves = None
    saved_leaves = _cached(run_dir / "3d" / "leaves" / "leaves.json")
    if saved_leaves and saved_leaves.get("result"):
        fit_on = (saved_leaves.get("key") or {}).get("fruit") or {}
        if np.allclose(fit_on.get("centre", np.nan), fruit.centre) and np.allclose(fit_on.get("normal", np.nan),
                                                                                   fruit.normal):
            leaves = Leaves.from_json(saved_leaves["result"])
    return RunAlignment(run_dir, np.asarray(fruit.centre, float), rotation, cams, fruit, leaves)


def calibration_mismatch(alignment: RunAlignment, params: dict, tol_mm: float = 1.0) -> list[str]:
    """Cameras whose calibration in a result's params.json differs from the one the fruit was
    fitted with (then the result's points are in another world and will not line up). Empty if
    they agree, or the result did not save its calibration (older results)."""
    cfg = (params or {}).get("cameras") or {}
    views, scale = cfg.get("views") or {}, float(cfg.get("t_scale", 1.0))
    out = []
    for cam in alignment.cameras:
        if cam.name not in views:
            continue
        other = camera_from_config(cam.name, views[cam.name], None, scale)
        if np.abs(other.R - cam.R).max() > 1e-4 or np.abs(other.t - cam.t).max() > tol_mm:
            out.append(cam.name)
    return out
