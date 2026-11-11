"""Annotated videos: boxes drawn on every frame of a sequence.

Frames are grouped into sequences by the folder part of their key
("<sequence>/frame_000161"), so each sequence gets its own video.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common.datasets import Frame, group_sequences, iter_frame_images

from ..common import GT_COLOUR, PRED_COLOUR, safe_name, to_bgr

# Back-projection videos (BGR): detector boxes, ground truth, boxes the factor graph inserted.
VIDEO_DET_BGR, VIDEO_BACKPROJ_BGR = (255, 120, 0), (0, 0, 255)
VIDEO_GT_BGR = to_bgr(GT_COLOUR)
VIDEO_CORRECTED_BGR = (0, 255, 0)   # step 10: hand-corrected tracks and the hand-placed correction points


@lru_cache(maxsize=1)
def _ffmpeg_h264() -> str | None:
    """Path of an ffmpeg binary with libx264, if there is one."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        encoders = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except OSError:
        return None
    return exe if "libx264" in encoders else None


class VideoWriter:
    """MP4 writer for BGR frames: H.264 through ffmpeg when it is installed (far smaller files at
    full resolution; crf 18 = near lossless, 28 = small, small objects still visible), else
    OpenCV's mp4v."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, crf: int = 26):
        self.path, self.proc, self.cv = path, None, None
        exe = _ffmpeg_h264()
        if exe:
            w, h = size
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{fps:g}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                 "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(path)], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, bgr: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())
        else:
            self.cv.write(bgr)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait():
                raise IOError(f"ffmpeg failed writing {self.path}")
        else:
            self.cv.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _draw_boxes(img: np.ndarray, boxes: np.ndarray, colour_rgb, with_score: bool) -> None:
    colour = to_bgr(colour_rgb)
    for b in boxes:
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        if with_score:
            cv2.putText(img, f"{b[4]:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                        cv2.LINE_AA)


def save_videos(frames: list[Frame], preds: dict[str, np.ndarray], out_dir: Path, model_name: str,
                conf: float, fps: float = 30.0) -> list[Path]:
    """One annotated MP4 per sequence: GT boxes in green, predictions in magenta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for seq, seq_frames in group_sequences(frames).items():
        w, h = seq_frames[0].width, seq_frames[0].height
        path = out_dir / f"{safe_name(seq)}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f, rgb in tqdm(iter_frame_images(seq_frames), total=len(seq_frames),
                           desc=f"Video [{seq[:40]}]", unit="frame", leave=False):
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            p = preds.get(f.key, np.zeros((0, 6)))
            _draw_boxes(img, f.gt[:, 1:5], GT_COLOUR, with_score=False)
            _draw_boxes(img, p[p[:, 4] >= conf], PRED_COLOUR, with_score=True)
            cv2.rectangle(img, (0, 0), (520, 70), (0, 0, 0), -1)
            cv2.putText(img, f"GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        to_bgr(GT_COLOUR), 2, cv2.LINE_AA)
            cv2.putText(img, f"{model_name} (conf>={conf:g})", (60, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        to_bgr(PRED_COLOUR), 2, cv2.LINE_AA)
            cv2.putText(img, f.key[-45:], (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            writer.write(img)
        writer.release()
        saved.append(path)
    return saved


def _legend(img: np.ndarray, entries: list[tuple[tuple, bool, bool, str]], title: str) -> None:
    """Top-left legend on a translucent dark panel. entries: (colour, box sample, dot sample, text)."""
    font, scale, row = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 26
    width = 60 + max(cv2.getTextSize(t, font, scale, 1)[0][0] for t in [title] + [e[3] for e in entries])
    x2, y2 = min(width, img.shape[1]), min(row * (len(entries) + 1) + 12, img.shape[0])
    img[:y2, :x2] = (img[:y2, :x2] * 0.35).astype(img.dtype)
    cv2.putText(img, title, (10, row), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (bgr, box, dot, text) in enumerate(entries, 2):
        y = row * i
        if box:
            cv2.rectangle(img, (12, y - 14), (40, y + 2), bgr, 1)
        if dot:
            cv2.circle(img, (26, y - 6), 2, bgr, -1, cv2.LINE_AA)
        cv2.putText(img, text, (50, y), font, scale, bgr, 1, cv2.LINE_AA)


def save_backproj_video(frames: list[Frame], preds: dict[str, np.ndarray], inserted: dict[str, np.ndarray],
                        track_points: dict[str, np.ndarray], path: Path, name: str, det_label: str, conf: float,
                        fps: float = 30.0, crf: int = 26, corrected_points: dict[str, np.ndarray] | None = None,
                        hand_points: dict[str, np.ndarray] | None = None) -> Path:
    """One video of a camera's frames: detector boxes (conf >= conf) in thin blue, ground truth in
    green, the factor-graph tracks reprojected as a red dot in every frame (on top of the detector
    box where there is one, to show how well the track follows it), and the boxes inserted where the
    detector has none in thin red. Boxes are (M, >=4) x1, y1, x2, y2 and track_points (N, 2) x, y,
    in raw pixels, keyed by frame key.
    Step 10 (hand corrections): corrected_points, the hand-corrected tracks reprojected where the
    correction moved them (elsewhere they are in track_points, red), as green dots; hand_points (N, 3) x, y, exact: the hand-placed correction points as green rings, thick on
    their own frame (exact = 1) and thin for a few frames either side, so they can be seen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [(VIDEO_DET_BGR, True, False, f"{det_label} (conf >= {conf:g})")]
    if any(len(f.gt) for f in frames):
        entries.append((VIDEO_GT_BGR, True, False, "ground truth"))
    entries += [(VIDEO_BACKPROJ_BGR, False, True, "factor-graph track, reprojected (every frame)"),
                (VIDEO_BACKPROJ_BGR, True, False, "factor graph, box inserted where the detector has none")]
    corrected_points, hand_points = corrected_points or {}, hand_points or {}
    if corrected_points:
        entries.append((VIDEO_CORRECTED_BGR, False, True, "hand-corrected track, where the correction moved it"))
    if hand_points:
        entries.append((VIDEO_CORRECTED_BGR, False, False, "o  hand-placed correction point"))
    empty = np.zeros((0, 6))
    with VideoWriter(path, (frames[0].width, frames[0].height), fps, crf) as writer:
        for f, rgb in tqdm(iter_frame_images(frames), total=len(frames), desc=f"Video [{name}]", unit="frame",
                           leave=False):
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            p = preds.get(f.key, empty)
            ins = inserted.get(f.key, empty)
            for boxes, bgr in ((p[p[:, 4] >= conf], VIDEO_DET_BGR), (f.gt[:, 1:5], VIDEO_GT_BGR),
                               (ins, VIDEO_BACKPROJ_BGR)):
                for x1, y1, x2, y2 in np.round(boxes[:, :4]).astype(int):
                    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 1)
            for x, y in np.round(track_points.get(f.key, np.zeros((0, 2)))).astype(int):
                cv2.circle(img, (x, y), 2, VIDEO_BACKPROJ_BGR, -1, cv2.LINE_AA)
            for x, y in np.round(corrected_points.get(f.key, np.zeros((0, 2)))).astype(int):
                cv2.circle(img, (x, y), 2, VIDEO_CORRECTED_BGR, -1, cv2.LINE_AA)
            for x, y, exact in hand_points.get(f.key, np.zeros((0, 3))):
                cv2.circle(img, (int(round(x)), int(round(y))), 12, VIDEO_CORRECTED_BGR, 2 if exact else 1, cv2.LINE_AA)
            _legend(img, entries, f"{name}  |  {f.key.rsplit('/', 1)[-1]}")
            writer.write(img)
    return path
