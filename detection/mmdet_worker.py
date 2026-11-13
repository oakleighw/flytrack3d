"""MMDetection inference worker, run in a separate Python that has mmdet installed.

mmcv / mmdet cannot be installed in the evaluation env (Python 3.14), so `detection/mmdet_remote.py`
starts this script with the openmmlab env's interpreter and talks to it over stdin/stdout.
It imports nothing from this repository and must stay Python 3.10-compatible.

    <openmmlab python> -I mmdet_worker.py --config cfg.py --checkpoint epoch_5.pth \
        [--repo path/to/mmdetection] --device cuda:0 --score-thr 0.001

--repo puts a source checkout of mmdet first on sys.path (as run_mm.py did for training), e.g. the
rtdetr-mmdet fork for RT-DETR, so its custom modules are registered.

Protocol (all little-endian; a message is a uint32 length + a UTF-8 JSON header, then raw bytes):
    -> {"op": "predict", "shapes": [[h, w, 3], ...]}   then the uint8 RGB images back to back
    <- {"counts": [n0, n1, ...]}                       then float32 (sum(n), 6): x1 y1 x2 y2 score class
    <- {"error": "<traceback>"}                        if the batch failed (the worker keeps running)
The first message the worker sends is {"ready": true, "classes": [...]}. EOF on stdin ends it.
stdout carries only the protocol: anything mmdet prints goes to stderr.
"""

import argparse
import json
import os
import struct
import sys
import traceback

# Keep the protocol stream clean: fd 1 becomes a private pipe handle, and the normal stdout
# (fd 1 / sys.stdout) is pointed at stderr before mmdet or torch get the chance to print.
_PROTO_OUT = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr
_PROTO_IN = sys.stdin.buffer


def send(header, payload=b""):
    blob = json.dumps(header).encode()
    _PROTO_OUT.write(struct.pack("<I", len(blob)) + blob + payload)
    _PROTO_OUT.flush()


def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = _PROTO_IN.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv():
    head = read_exact(4)
    if head is None:
        return None
    return json.loads(read_exact(struct.unpack("<I", head)[0]))


def build_pipeline(cfg):
    """The config's test pipeline, fed numpy arrays instead of files and without GT loading."""
    from mmcv.transforms import Compose

    ds = cfg.test_dataloader.dataset
    while "pipeline" not in ds and "dataset" in ds:  # wrapped datasets (RepeatDataset, ...)
        ds = ds.dataset
    steps = [dict(s) for s in ds.pipeline if "LoadAnnotations" not in s["type"]]
    steps[0] = dict(type="mmdet.LoadImageFromNDArray")
    return Compose(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.001)
    args = ap.parse_args()

    if args.repo:
        sys.path.insert(0, os.path.abspath(args.repo))

    import numpy as np
    import torch
    from mmdet.apis import init_detector

    print(f"[mmdet_worker] mmdet from {sys.modules['mmdet'].__file__}", file=sys.stderr, flush=True)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    pipeline = build_pipeline(model.cfg)
    classes = list(model.dataset_meta.get("classes", ())) if model.dataset_meta else []
    send({"ready": True, "classes": classes})

    while True:
        msg = recv()
        if msg is None:
            return
        try:
            shapes = [tuple(s) for s in msg["shapes"]]
            raw = read_exact(sum(int(np.prod(s)) for s in shapes))
            if raw is None:
                return
            inputs, samples, offset = [], [], 0
            for i, shape in enumerate(shapes):
                n = int(np.prod(shape))
                rgb = np.frombuffer(raw, np.uint8, n, offset).reshape(shape)
                offset += n
                # mmdet's data preprocessors expect BGR, as cv2/mmcv load images
                data = pipeline(dict(img=np.ascontiguousarray(rgb[:, :, ::-1]), img_id=i))
                inputs.append(data["inputs"])
                samples.append(data["data_samples"])
            with torch.no_grad():
                results = model.test_step(dict(inputs=inputs, data_samples=samples))
            out, counts = [], []
            for r in results:
                inst = r.pred_instances
                keep = inst.scores >= args.score_thr
                arr = torch.cat([inst.bboxes[keep], inst.scores[keep, None],
                                 inst.labels[keep, None].to(inst.scores.dtype)], dim=1)
                out.append(arr.float().cpu().numpy())
                counts.append(len(arr))
            payload = np.concatenate(out).astype("<f4").tobytes() if out else b""
            send({"counts": counts}, payload)
        except Exception:
            send({"error": traceback.format_exc()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        send({"error": traceback.format_exc()})
        raise
