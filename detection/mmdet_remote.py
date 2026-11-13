"""MMDetection models (DINO, RT-DETR, ...) inside the evaluation pipeline.

mmcv / mmdet cannot be installed in this repo's env, so the model runs in a worker process
(`mmdet_worker.py`) started with another env's Python (the openmmlab conda env). Here that worker
is wrapped as a SAHI `DetectionModel`, so the slicing, the full-frame pass and the NMS merge are
SAHI's, exactly as for the YOLO models (`SahiDetector`); only the forward pass is remote.
"""

from __future__ import annotations

import atexit
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

WORKER = Path(__file__).with_name("mmdet_worker.py")
DEFAULT_PYTHON = str(Path.home() / "miniconda3/envs/openmmlab/bin/python")


class MMDetWorker:
    """A running mmdet_worker.py process; `predict` sends a batch of RGB images, gets boxes back."""

    def __init__(self, python: str, config: str, checkpoint: str, repo: str | None,
                 device: str, score_thr: float, cuda_visible_devices: str | None = None):
        """device is as written in the config ("cuda:1"), relative to cuda_visible_devices, which
        is CUDA_VISIBLE_DEVICES as the user launched the script (None = unset, all GPUs visible)."""
        cmd = [python, "-I", str(WORKER), "--config", config, "--checkpoint", checkpoint,
               "--device", device, "--score-thr", str(score_thr)]
        if repo:
            cmd += ["--repo", repo]
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)  # possibly overwritten by SAHI's select_device
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        # stderr is inherited, so mmdet's load messages and any traceback show in this terminal.
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=sys.stderr, env=env)
        atexit.register(self.close)
        msg = self._recv()
        self.classes: list[str] = msg["classes"]

    def _read(self, n: int) -> bytes:
        buf = self.proc.stdout.read(n)
        if len(buf) < n:
            code = self.proc.poll()
            raise RuntimeError(f"mmdet worker exited (code {code}); see its output above")
        return buf

    def _recv(self) -> dict:
        msg = json.loads(self._read(struct.unpack("<I", self._read(4))[0]))
        if "error" in msg:
            raise RuntimeError(f"mmdet worker failed:\n{msg['error']}")
        return msg

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """RGB uint8 images -> per image an (N, 6) array: x1, y1, x2, y2, score, class_id."""
        images = [np.ascontiguousarray(im, dtype=np.uint8) for im in images]
        head = json.dumps({"op": "predict", "shapes": [list(im.shape) for im in images]}).encode()
        try:
            self.proc.stdin.write(struct.pack("<I", len(head)) + head)
            for im in images:
                self.proc.stdin.write(im.data)
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError(f"mmdet worker exited (code {self.proc.poll()}); see its output above")
        msg = self._recv()
        counts = msg["counts"]
        flat = np.frombuffer(self._read(sum(counts) * 6 * 4), "<f4").reshape(-1, 6).astype(np.float64)
        return np.split(flat, np.cumsum(counts)[:-1])

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()  # EOF ends the worker's loop
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class RemoteMMDetModel(DetectionModel):
    """SAHI detection model whose forward pass runs in an MMDetWorker."""

    def __init__(self, *, python: str, repo: str | None, worker_device: str,
                 cuda_visible_devices: str | None, **kwargs):
        # The worker gets the configured device string, not SAHI's self.device: SAHI's
        # select_device rewrites CUDA_VISIBLE_DEVICES, so its device index means another GPU.
        self.python, self.repo = python, repo
        self.worker_device, self.cuda_visible_devices = worker_device, cuda_visible_devices
        self.worker: MMDetWorker | None = None
        super().__init__(**kwargs)

    def load_model(self) -> None:
        self.worker = MMDetWorker(self.python, self.config_path, self.model_path, self.repo,
                                  self.worker_device, self.confidence_threshold,
                                  self.cuda_visible_devices)
        self.model = self.worker
        self.category_mapping = {str(i): n for i, n in enumerate(self.worker.classes)}

    def perform_inference(self, image: np.ndarray) -> None:
        self.perform_batch_inference([image])

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        self._original_predictions = self.worker.predict(images)
        self._original_shapes = [im.shape for im in images]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = [[0, 0]],
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        # Same box handling as SAHI's UltralyticsDetectionModel, so both families are post-processed alike.
        shifts = fix_shift_amount_list(shift_amount_list)
        full_shapes = fix_full_shape_list(full_shape_list)
        per_image = []
        for i, boxes in enumerate(self._original_predictions):
            shift = [int(x) for x in shifts[i]]
            full_shape = None if full_shapes is None else [int(x) for x in full_shapes[i]]
            image_shape = self._original_shapes[i]
            preds = []
            for x1, y1, x2, y2, score, cls in boxes:
                if score < self.confidence_threshold:
                    continue
                bbox = [max(0.0, float(v)) for v in (x1, y1, x2, y2)]
                if full_shape is not None:
                    bbox = [min(full_shape[1], bbox[0]), min(full_shape[0], bbox[1]),
                            min(full_shape[1], bbox[2]), min(full_shape[0], bbox[3])]
                if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                    continue
                cid = int(cls)
                preds.append(ObjectPrediction(
                    bbox=bbox, category_id=cid, score=float(score),
                    category_name=self.category_mapping.get(str(cid), str(cid)),
                    shift_amount=shift,
                    full_shape=list(image_shape[:2]) if full_shape is None else full_shape,
                ))
            per_image.append(preds)
        self._object_prediction_list_per_image = per_image
