"""Detector wrappers.

All detectors share one interface so the evaluation pipeline doesn't care what
model is underneath:

    det = build_detector(model_cfg)
    det.cache_params()        -> dict describing everything that affects predictions
    det.ensure_loaded()       -> load weights once (only called if inference is needed)
    det.predict(rgb_image)    -> np.ndarray (M, 6): x1, y1, x2, y2, score, class_id
    det.class_names           -> {class_id: name}

To add a new model family, subclass `Detector`, implement `load`/`predict`
(and extend `cache_params` with any settings that change the output), then
register it in `DETECTOR_TYPES`.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class Detector(ABC):
    # Config keys that don't change predictions and so are excluded from the cache key.
    # (model_path is replaced by the weights' sha256, so moving the file keeps the cache valid.)
    NON_CACHE_KEYS = {"name", "enabled", "device", "notes", "model_path"}

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.name: str = cfg["name"]
        self.model_path: str | None = cfg.get("model_path")
        self.confidence_threshold: float = float(cfg.get("confidence_threshold", 0.001))
        self.class_names: dict[int, str] = {}
        self._loaded = False
        self._weights_sha256: str | None = None

    def cache_params(self) -> dict:
        params = {k: v for k, v in self.cfg.items() if k not in self.NON_CACHE_KEYS}
        params["confidence_threshold"] = self.confidence_threshold
        if self.model_path:
            # Hash the weights, not just the path, so retrained weights never reuse old predictions.
            if self._weights_sha256 is None:
                self._weights_sha256 = file_sha256(self.model_path)
            params["model_sha256"] = self._weights_sha256
        return params

    def ensure_loaded(self) -> None:
        """Load the model on first use; batch runs reuse it for every input."""
        if not self._loaded:
            self.load()
            self._loaded = True

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def predict(self, image_rgb: np.ndarray) -> np.ndarray: ...


class SahiDetector(Detector):
    """Sliced inference through SAHI. Works with any SAHI `model_type`
    (ultralytics, huggingface, torchvision, mmdet, ...)."""

    # batch_size only changes speed (tiles per GPU forward pass), not the predictions.
    NON_CACHE_KEYS = Detector.NON_CACHE_KEYS | {"batch_size"}

    DEFAULTS = {
        "model_type": "ultralytics",
        "batch_size": 12,
        "slice_height": 640,
        "slice_width": 640,
        "overlap_height_ratio": 0.2,
        "overlap_width_ratio": 0.2,
        "perform_standard_pred": True,
        # NMS/IOU, not SAHI's GREEDYNMM/IOS: greedy merging chains the many low-score boxes of a
        # low confidence_threshold into oversized boxes. Passed with force_postprocess_type, so
        # what is configured is what runs (SAHI otherwise switches to NMS/IOU below 0.1 itself).
        "postprocess_type": "NMS",
        "postprocess_match_metric": "IOU",
        "postprocess_match_threshold": 0.5,
        "postprocess_class_agnostic": False,
        "image_size": None,
    }

    def __init__(self, cfg: dict):
        super().__init__({**self.DEFAULTS, **cfg})
        self.model = None

    def load(self) -> None:
        from sahi import AutoDetectionModel

        c = self.cfg
        self.model = AutoDetectionModel.from_pretrained(
            model_type=c["model_type"],
            model_path=self.model_path,
            config_path=c.get("config_path"),
            confidence_threshold=self.confidence_threshold,
            device=c.get("device", "cuda:0"),
            image_size=c["image_size"],
        )
        self.class_names = {int(k): v for k, v in (self.model.category_mapping or {}).items()}

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        from sahi.predict import get_sliced_prediction

        c = self.cfg
        result = get_sliced_prediction(
            image_rgb,  # SAHI expects RGB
            self.model,
            slice_height=c["slice_height"],
            slice_width=c["slice_width"],
            overlap_height_ratio=c["overlap_height_ratio"],
            overlap_width_ratio=c["overlap_width_ratio"],
            perform_standard_pred=c["perform_standard_pred"],
            postprocess_type=c["postprocess_type"],
            postprocess_match_metric=c["postprocess_match_metric"],
            postprocess_match_threshold=c["postprocess_match_threshold"],
            postprocess_class_agnostic=c["postprocess_class_agnostic"],
            force_postprocess_type=True,
            batch_size=c["batch_size"],
            verbose=0,
        )
        return np.array(
            [[*p.bbox.to_xyxy(), p.score.value, p.category.id] for p in result.object_prediction_list],
            dtype=np.float64,
        ).reshape(-1, 6)


class MMDetSahiDetector(SahiDetector):
    """MMDetection models (DINO, RT-DETR, ...) with the same SAHI slicing and merging as
    `SahiDetector`. mmdet runs in a worker process on another Python env, since it can't be
    installed in this one (see mmdet_remote.py).

    Config keys on top of SahiDetector's:
        config_path     the config mmdet saved in the training work_dir
        model_path      the checkpoint (epoch_N.pth)
        mmdet_repo      source checkout put first on sys.path, as the training script did
                        (the rtdetr-mmdet fork for RT-DETR); null = the env's installed mmdet
        python          the interpreter of the env that has mmdet (the openmmlab conda env)
    image_size and model_type don't apply: each image is resized by the config's test pipeline.
    """

    # python only picks the interpreter; config_path is replaced by the config's sha256 below.
    NON_CACHE_KEYS = SahiDetector.NON_CACHE_KEYS | {"python", "config_path", "model_type", "image_size"}

    def __init__(self, cfg: dict):
        from .mmdet_remote import DEFAULT_PYTHON

        super().__init__({"python": DEFAULT_PYTHON, "mmdet_repo": None, **cfg})
        if not self.cfg.get("config_path"):
            raise ValueError(f"Model '{self.name}': type 'mmdet_sahi' needs 'config_path'.")

    def cache_params(self) -> dict:
        params = super().cache_params()
        params["config_sha256"] = file_sha256(self.cfg["config_path"])
        return params

    def load(self) -> None:
        from .mmdet_remote import RemoteMMDetModel

        c = self.cfg
        self.model = RemoteMMDetModel(
            python=c["python"], repo=c["mmdet_repo"],
            model_path=self.model_path, config_path=c["config_path"],
            confidence_threshold=self.confidence_threshold, device=c.get("device", "cuda:0"),
        )
        self.class_names = {int(k): v for k, v in self.model.category_mapping.items()}


class UltralyticsDetector(Detector):
    """Plain full-frame Ultralytics inference (no slicing) - a useful baseline."""

    DEFAULTS = {"image_size": 640, "iou": 0.7, "max_det": 300}

    def __init__(self, cfg: dict):
        super().__init__({**self.DEFAULTS, **cfg})
        self.model = None

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)
        self.class_names = {int(k): v for k, v in self.model.names.items()}

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        c = self.cfg
        r = self.model.predict(
            image_rgb[:, :, ::-1],  # Ultralytics expects BGR numpy arrays
            conf=self.confidence_threshold, iou=c["iou"], imgsz=c["image_size"],
            max_det=c["max_det"], device=c.get("device", "cuda:0"), verbose=False,
        )[0]
        b = r.boxes
        return np.concatenate(
            [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], axis=1
        ).astype(np.float64).reshape(-1, 6)


DETECTOR_TYPES: dict[str, type[Detector]] = {
    "sahi": SahiDetector,
    "mmdet_sahi": MMDetSahiDetector,
    "ultralytics": UltralyticsDetector,
}


def build_detector(cfg: dict) -> Detector:
    kind = cfg.get("type", "sahi")
    if kind not in DETECTOR_TYPES:
        raise ValueError(f"Unknown detector type '{kind}'. Options: {list(DETECTOR_TYPES)}")
    return DETECTOR_TYPES[kind](cfg)
