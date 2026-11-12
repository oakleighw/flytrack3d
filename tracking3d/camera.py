"""The camera model and the multi-view geometry built on it.

Camera conventions follow OpenCV (see configs/shared/cameras.example.json):
    matrix       3x3 intrinsics K
    distortions  k1 k2 p1 p2 [k3 [k4 k5 k6]] for "pinhole", k1 k2 k3 k4 for "fisheye"
    rotation     world -> camera, as a Rodrigues vector (3 values) or a 3x3 matrix
    translation  world -> camera:  x_cam = R @ X_world + t  (what calibrateCamera,
                 solvePnP and stereoCalibrate return)
The OpenCV short names K, dist, R / rvec and t are accepted too.
World units are the units of translation times cameras.t_scale.

Detections are undistorted once to ideal pinhole pixels (same K). Everything
downstream (matching, triangulation, the factor graph) then uses a plain
pinhole camera, which supports any OpenCV distortion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ----------------------------------------------------------------------------- cameras

@dataclass
class Camera:
    name: str
    K: np.ndarray                     # (3, 3)
    dist: np.ndarray                  # (k,)
    R: np.ndarray                     # (3, 3) world -> camera
    t: np.ndarray                     # (3,)
    model: str = "pinhole"            # "pinhole" or "fisheye"
    image_size: tuple[int, int] | None = None
    frame_offset: int = 0             # added to this view's frame numbers to align views

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def axis(self) -> np.ndarray:
        """Viewing direction (camera +z) in world coordinates."""
        return self.R[2]

    def undistort(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) raw pixels -> (N, 2) ideal pinhole pixels."""
        if not len(xy):
            return np.zeros((0, 2))
        pts = np.asarray(xy, np.float64).reshape(-1, 1, 2)
        if self.model == "fisheye":
            out = cv2.fisheye.undistortPoints(pts, self.K, self.dist, P=self.K)
        else:
            out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def project(self, X: np.ndarray, distort: bool = False) -> np.ndarray:
        """(N, 3) world points -> (N, 2) pixels (undistorted unless distort=True)."""
        X = np.ascontiguousarray(X, np.float64).reshape(-1, 3)   # cv2 rejects strided views (csv columns)
        if not len(X):
            return np.zeros((0, 2))
        if distort:
            rvec = cv2.Rodrigues(self.R)[0]
            if self.model == "fisheye":
                out = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), rvec, self.t, self.K, self.dist)[0]
            else:
                out = cv2.projectPoints(X, rvec, self.t, self.K, self.dist)[0]
            return out.reshape(-1, 2)
        x = X @ self.P[:, :3].T + self.P[:, 3]
        return x[:, :2] / x[:, 2:3]

    def depth(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X).reshape(-1, 3) @ self.R.T + self.t)[:, 2]


def _matrix(v, shape: tuple[int, ...], what: str) -> np.ndarray:
    if v is None:
        raise ValueError(f"{what} is not filled in")
    a = np.asarray(v, dtype=np.float64)
    if a.size != np.prod(shape):
        raise ValueError(f"{what} must have {np.prod(shape)} values, got {a.size}")
    return a.reshape(shape)


def _get(cfg: dict, *keys: str):
    """First of `keys` present in cfg (calibration fields accept OpenCV short names too)."""
    return next((cfg[k] for k in keys if cfg.get(k) is not None), None)


def camera_from_config(name: str, cfg: dict, image_size: tuple[int, int] | None = None,
                       t_scale: float = 1.0) -> Camera:
    """Build a Camera from one entry of config["cameras"]["views"]. t is multiplied by t_scale
    (cameras.t_scale, e.g. 1000 for a calibration in metres and a world in mm). If the
    calibration image_size differs from the video's, K is rescaled (same aspect crop assumed)."""
    where = f"cameras.views.{name}"
    K = _matrix(_get(cfg, "matrix", "K"), (3, 3), f"{where}.matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{where}.matrix has no focal length - fill in the calibration")
    dist = np.asarray(_get(cfg, "distortions", "dist") or [], dtype=np.float64).ravel()
    rot = _get(cfg, "rotation", "R", "rvec")
    if rot is not None and np.size(rot) == 9:
        R = _matrix(rot, (3, 3), f"{where}.rotation")
    else:
        R = cv2.Rodrigues(_matrix(rot, (3,), f"{where}.rotation"))[0]
    if abs(np.linalg.det(R) - 1) > 1e-3:
        raise ValueError(f"{where}.rotation is not a rotation (det={np.linalg.det(R):.4f})")
    t = _matrix(_get(cfg, "translation", "t"), (3,), f"{where}.translation") * t_scale
    model = cfg.get("model", "pinhole")
    if model not in ("pinhole", "fisheye"):
        raise ValueError(f"{where}.model must be 'pinhole' or 'fisheye', got '{model}'")
    if model == "fisheye" and dist.size != 4:
        raise ValueError(f"{where}: fisheye 'distortions' needs 4 values (k1 k2 k3 k4)")

    calib_size = tuple(cfg["image_size"]) if cfg.get("image_size") else None
    if calib_size and image_size and tuple(image_size) != calib_size:
        sx, sy = image_size[0] / calib_size[0], image_size[1] / calib_size[1]
        print(f"  NOTE: {name} calibrated at {calib_size[0]}x{calib_size[1]} but video is "
              f"{image_size[0]}x{image_size[1]}; scaling K by ({sx:.4f}, {sy:.4f})")
        K = np.diag([sx, sy, 1.0]) @ K
    return Camera(name=name, K=K, dist=dist, R=R, t=t, model=model,
                  image_size=tuple(image_size or calib_size or ()) or None,
                  frame_offset=int(cfg.get("frame_offset", 0)))


# ----------------------------------------------------------------------------- geometry

def fundamental(a: Camera, b: Camera) -> np.ndarray:
    """F with x_b^T F x_a = 0 for undistorted pixels."""
    R = b.R @ a.R.T
    t = b.t - R @ a.t
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(b.K).T @ tx @ R @ np.linalg.inv(a.K)


def epipolar_distance(F: np.ndarray, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """(Na, Nb) mean point-to-epipolar-line distance in pixels."""
    ha = np.hstack([xa, np.ones((len(xa), 1))])
    hb = np.hstack([xb, np.ones((len(xb), 1))])
    lb = ha @ F.T                       # lines in b for each point in a  (Na, 3)
    la = hb @ F                         # lines in a for each point in b  (Nb, 3)
    alg = np.abs(ha @ F.T @ hb.T)       # |x_b^T F x_a|                   (Na, Nb)
    da = alg / np.linalg.norm(la[:, :2], axis=1)[None, :]
    db = alg / np.linalg.norm(lb[:, :2], axis=1)[:, None]
    return (da + db) / 2


def triangulate_dlt(cams: list[Camera], xy: list[np.ndarray]) -> np.ndarray:
    """Linear triangulation of one point from >= 2 views (undistorted pixels).
    Solved in normalised image coordinates for conditioning."""
    rows = []
    for cam, p in zip(cams, xy):
        n = np.linalg.solve(cam.K, [p[0], p[1], 1.0])
        Rt = np.hstack([cam.R, cam.t[:, None]])
        rows += [n[0] * Rt[2] - n[2] * Rt[0], n[1] * Rt[2] - n[2] * Rt[1]]
    X = np.linalg.svd(np.array(rows))[2][-1]
    return X[:3] / X[3]


def reprojection_errors(cams: list[Camera], xy: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(c.project(X)[0] - p) for c, p in zip(cams, xy)])
