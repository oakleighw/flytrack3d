"""Start / end annotation GUI for guided 3D tracking (main_tracks3d_guided.py).

The views of one recording session are annotated one after another in a single
window. Object numbers are shared by every view: object #1 in the top camera is the
same fly as object #1 in the bottom camera. Each object can have one START and one
END point per view; any of them may be left out (e.g. a start in one view only).

    image      circles            the detections being tracked (model predictions, or
                                  ground-truth labels with --source gt)
               left click         place a point for the selected object (snaps to a
                                  detection if one is close)
               left drag          draw a box (its centre is used)
               right click        delete the nearest annotation on this frame
               wheel / middle     zoom / pan
    timeline   click or drag      seek; ticks mark annotations (colour = object)
    keys       a / d   A / D      previous / next frame, -10 / +10
               s / e              START / END mode
               n, 1-9, [ / ]      new object, select object, previous / next object
               h                  show / hide detections
               l                  show / hide the selected object's epipolar lines
               S / E              suggest STARTs / ENDs in this view from the other views
                                  (epipolar geometry, see tracking3d/annotator/suggest.py); marked "?",
                                  delete or replace them like any annotation
               g                  go to the instant of the selected object's START / END
                                  (current mode) in another view
               , / .   < / >      this view's frame offset -1 / +1, -10 / +10
               o                  type this view's frame offset (or click the offset box)
               O                  apply the offset the panel suggests (from a START / END
                                  annotated in two views)
               W                  save the current offsets to cameras.json as the defaults
               Enter              continue to the next view (finish on the last)
               Backspace          back to the previous view
               Esc                cancel everything

Views are synchronised with each camera's frame_offset (frame + frame_offset = the same
instant in every view): switching view opens it at the instant you were looking at, the
panel shows the matching frame in every other view, and when the same object's START (or
END) is annotated in two views at instants that differ, it shows the offset that would
line them up - annotate one sharp event (e.g. a take-off) in every view to find an offset.
Offsets can be changed while annotating; the panel, epipolar lines and suggestions follow
at once. The GUI works on copies of the cameras: the offsets it ends with are returned next
to the annotations, and the caller decides where to keep them.

Annotations are returned (with the final frame offsets, camera -> offset) as plain dicts:
    {"object": 1, "view": "344-above", "kind": "start", "frame": 214,
     "x": 1032.5, "y": 455.0, "box": [x, y, w, h] or None}
with x, y in raw (distorted) pixels and frame = the label frame number (frame_000214).
Suggested points also have "suggested": true (dropped once placed by hand).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from common.datasets import IMAGE_EXTS

from .suggest import aligned, epipolar_polyline, suggest

# BGR
BG = (32, 32, 32)
PANEL = (44, 44, 44)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
ACCENT = (200, 140, 40)
START_C = (90, 200, 90)
END_C = (80, 80, 230)
# Same order as the 3D plots' track colours (matplotlib tab10 without green), as BGR.
OBJECT_COLOURS = [(180, 119, 31), (14, 127, 255), (40, 39, 214), (189, 103, 148), (75, 86, 140),
                  (194, 119, 227), (127, 127, 127), (34, 189, 188), (207, 190, 23)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

PANEL_W = 400
SCROLL_W = 26                 # width of the up / down column beside an overflowing table
TIMELINE_H = 54
MIN_H = 830
MAX_VIEW_W, MAX_VIEW_H = 1280, 820
SNAP_PX = 14          # display pixels within which a click snaps to a detection
DRAG_MIN_PX = 4       # display pixels: shorter drags count as clicks

KEY_LEFT = {81, 65361, 2424832, 16777234}
KEY_RIGHT = {83, 65363, 2555904, 16777236}


LABEL_SCALE = 0.4             # text beside the marks on the image (kept small: the flies are tiny)
SMALL_SCALE = 0.35            # secondary text on the image (other frames' marks, epipolar lines)


def draw_cross(img: np.ndarray, p: tuple[int, int], colour, tilted: bool = False, size: int = 11, gap: int = 4) -> None:
    """A 1 px cross (x if tilted) with an open centre, so the pixel it marks stays visible."""
    x, y = p
    d = (1, 1) if tilted else (1, 0)
    for sx, sy in (d, (-d[0], -d[1]), (-d[1], d[0]), (d[1], -d[0])):   # four arms, 90 degrees apart
        cv2.line(img, (x + sx * gap, y + sy * gap), (x + sx * size, y + sy * size), colour, 1, cv2.LINE_AA)


def object_colour(obj: int) -> tuple[int, int, int]:
    return OBJECT_COLOURS[(obj - 1) % len(OBJECT_COLOURS)]


@dataclass
class ViewInput:
    """One camera view to annotate."""
    name: str                                  # camera name (stored in annotations)
    label: str                                 # input folder name (display only)
    video: str | None = None                   # a video file ...
    images: list[str] | None = None            # ... or the frames as image files
    frame_numbers: list[int] = field(default_factory=list)       # label frame number per position
    detections: dict[int, np.ndarray] = field(default_factory=dict)  # frame number -> (N, >=2) x, y
    camera: object | None = None               # tracking3d Camera; needed for suggestions


class FrameSource:
    """Random access to a video or image folder, reading sequentially when it can."""

    def __init__(self, view: ViewInput):
        self.view = view
        self.cap = cv2.VideoCapture(view.video) if view.video else None
        self.pos = -1
        self.cache: tuple[int, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.view.frame_numbers)

    def read(self, i: int) -> np.ndarray:
        if self.cache and self.cache[0] == i:
            return self.cache[1]
        if self.cap is not None:
            if i != self.pos + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = self.cap.read()
            self.pos = i if ok else -1
            if not ok:
                img = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(img, f"cannot read frame {i}", (20, 240), FONT, 0.8, (0, 0, 255), 2)
        else:
            img = cv2.imread(self.view.images[i], cv2.IMREAD_COLOR)
        self.cache = (i, img)
        return img

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def view_input_from_source(name: str, label: str, source: dict, detections: dict[int, np.ndarray],
                           camera=None) -> ViewInput:
    """Build a ViewInput from a batch input's dataset config (video or folder of frames)."""
    if source.get("video") and Path(source["video"]).is_file():
        cap = cv2.VideoCapture(source["video"])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        start = int(source.get("start_index", 1))   # same default as datasets._load_video
        return ViewInput(name, label, video=source["video"], frame_numbers=[i + start for i in range(n)],
                         detections=detections, camera=camera)
    if source.get("images") and Path(source["images"]).is_dir():
        imgs = sorted(p for p in Path(source["images"]).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        nums = [int(m.group(1)) if (m := re.search(r"(\d+)$", p.stem)) else i for i, p in enumerate(imgs)]
        return ViewInput(name, label, images=[str(p) for p in imgs], frame_numbers=nums, detections=detections,
                         camera=camera)
    raise FileNotFoundError(f"No readable video or image folder for view '{name}' ({source})")


def view_inputs_from_session(session) -> list[ViewInput]:
    """One ViewInput per view of a tracking3d Session (its video, its detections back in label
    frame numbers, its camera). Exits with a message if a view's video cannot be opened."""
    views = []
    for v in session.views:
        detections = {f - v.camera.frame_offset: d for f, d in v.raw.items()}  # back to label frame numbers
        try:
            views.append(view_input_from_source(v.name, v.input_name, v.source, detections, v.camera))
        except FileNotFoundError as e:
            raise SystemExit(f"Cannot open the video for {v.name}: {e}") from None
    return views


class AnnotationGUI:
    window = "Annotate start / end points"

    def __init__(self, views: list[ViewInput], annotations: list[dict] | None = None, title: str = "",
                 detections_label: str = "detections", suggest_gate_px: float = 15.0,
                 default_offsets: dict[str, int] | None = None,
                 save_defaults: Callable[[dict[str, int]], str] | None = None):
        # Copies of the cameras, so offset changes stay here until the caller takes them.
        self.views = [replace(v, camera=copy.copy(v.camera)) if v.camera is not None else v for v in views]
        self.default_offsets = dict(default_offsets or self.offsets)   # cameras.json's, shown for reference
        self.save_defaults = save_defaults   # offsets -> status message; writes them as the defaults
        self.suggest_gate_px = suggest_gate_px     # epipolar / reprojection gate for suggestions
        self.title = title
        self.detections_label = detections_label   # what the circles are, shown in the help text
        self.ann: list[dict] = [dict(a) for a in (annotations or [])]
        self.vi = 0
        self.positions = [0] * len(views)          # current frame position per view
        self.visited = [False] * len(views)
        self.mode = "start"
        ids = sorted({a["object"] for a in self.ann})
        self.obj = ids[0] if ids else 1
        self.show_dets = True
        self.show_epipolar = True
        self.status = ""
        self.done: bool | None = None               # True = finished, False = cancelled
        # interaction
        self.zoom, self.base_scale, self.offset = 1.0, 1.0, [0.0, 0.0]
        self.drag_start = self.drag_now = None
        self.pan = None
        self.seeking = False
        self.frame_input: str | None = None
        self.offset_input: str | None = None
        self.buttons: dict[str, tuple[int, int, int, int]] = {}
        self.rows: list[tuple[tuple[int, int, int, int], dict]] = []
        self.table_scroll = 0
        self.source: FrameSource | None = None
        self.img: np.ndarray | None = None

    # ------------------------------------------------------------------ main loop

    def run(self) -> list[dict] | None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._open_view(0, sync=False)
        try:
            while self.done is None:
                cv2.imshow(self.window, self._render())
                key = cv2.waitKeyEx(15)
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    self.done = False
                    break
                if key != -1:
                    self._on_key(key)
        finally:
            if self.source:
                self.source.close()
            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
        return sorted(self.ann, key=lambda a: (a["object"], a["view"], a["kind"])) if self.done else None

    # ------------------------------------------------------------------ views / geometry

    @property
    def view(self) -> ViewInput:
        return self.views[self.vi]

    @property
    def pos(self) -> int:
        return self.positions[self.vi]

    @property
    def frame_no(self) -> int:
        return self.view.frame_numbers[self.pos]

    @staticmethod
    def _offset(view: ViewInput) -> int:
        return view.camera.frame_offset if view.camera is not None else 0

    @property
    def offsets(self) -> dict[str, int]:
        """Current frame offset of every view with a camera."""
        return {v.name: v.camera.frame_offset for v in self.views if v.camera is not None}

    def _set_offset(self, value: int) -> None:
        """Change this view's frame offset. The view stays on its frame; the other views' matching
        frames, epipolar lines and suggestions follow (they read the offset when drawn / made)."""
        cam = self.view.camera
        if cam is None:
            self.status = "This view has no camera - no frame offset"
            return
        old, cam.frame_offset = cam.frame_offset, int(value)
        self.status = f"{self.view.name} offset {old:+d} -> {cam.frame_offset:+d}"
        if any(a.get("suggested") for a in self.ann):
            self.status += " (re-run S / E suggestions)"

    def _apply_hint(self) -> None:
        hints = self._offset_hints()
        if not hints:
            self.status = f"No offset to suggest: annotate #{self.obj} {self.mode.upper()} here and in another view"
            return
        self._set_offset(hints[0][1])

    def _save_defaults(self) -> None:
        if self.save_defaults is None:
            self.status = "Saving default offsets is not available here"
            return
        try:
            self.status = self.save_defaults(self.offsets)
            self.default_offsets.update(self.offsets)
        except (OSError, ValueError) as e:
            self.status = f"Not saved: {e}"
            print(f"\nCould not save the frame offsets as defaults: {e}")

    def _instant(self) -> int:
        """The current frame on the shared timeline (frame + this view's frame_offset)."""
        return self.frame_no + self._offset(self.view)

    def _frame_at(self, view: ViewInput, instant: int) -> int:
        """view's label frame number at a shared-timeline instant."""
        return instant - self._offset(view)

    def _seek_instant(self, instant: int) -> str:
        """Seek the current view to an instant; returns a note if it is outside this video."""
        frame = self._frame_at(self.view, instant)
        nums = np.asarray(self.view.frame_numbers)
        self._seek(int(np.abs(nums - frame).argmin()))
        if frame < nums.min() or frame > nums.max():
            return f" ({frame} not in video)"
        return ""

    def _open_view(self, vi: int, sync: bool = True) -> None:
        instant = self._instant() if sync and self.source else None
        if self.source:
            self.source.close()
        self.vi = vi
        self.visited[vi] = True
        self.source = FrameSource(self.view)
        if not len(self.source):
            raise RuntimeError(f"View {self.view.name} has no frames")
        img = self.source.read(self.pos)
        h, w = img.shape[:2]
        self.base_scale = min(MAX_VIEW_W / w, MAX_VIEW_H / h, 1.0)
        self.vp = (int(w * self.base_scale), int(h * self.base_scale))
        self.zoom, self.offset = 1.0, [0.0, 0.0]
        self.canvas_size = (self.vp[0] + PANEL_W, max(self.vp[1] + TIMELINE_H, MIN_H))
        cv2.resizeWindow(self.window, *self.canvas_size)
        self.table_scroll = 0
        self.status = f"View {vi + 1}/{len(self.views)}: {self.view.name}"
        if instant is not None:
            note = self._seek_instant(instant)
            self.status += f", synced to {self.frame_no}{note}"

    def _seek(self, pos: int) -> None:
        self.positions[self.vi] = int(np.clip(pos, 0, len(self.source) - 1))

    @property
    def scale(self) -> float:
        return self.base_scale * self.zoom

    def _clamp_offset(self) -> None:
        h, w = self.img.shape[:2]
        vis_w, vis_h = self.vp[0] / self.scale, self.vp[1] / self.scale
        self.offset[0] = float(np.clip(self.offset[0], 0, max(0.0, w - vis_w)))
        self.offset[1] = float(np.clip(self.offset[1], 0, max(0.0, h - vis_h)))

    def _to_disp(self, x: float, y: float) -> tuple[int, int]:
        return int(round((x - self.offset[0]) * self.scale)), int(round((y - self.offset[1]) * self.scale))

    def _to_img(self, x: int, y: int) -> tuple[float, float]:
        return x / self.scale + self.offset[0], y / self.scale + self.offset[1]

    def _in_viewport(self, x: int, y: int) -> bool:
        return 0 <= x < self.vp[0] and 0 <= y < self.vp[1]

    def _in_timeline(self, x: int, y: int) -> bool:
        return 0 <= x < self.vp[0] and self.vp[1] <= y < self.vp[1] + TIMELINE_H

    # ------------------------------------------------------------------ annotations

    def _view_ann(self) -> list[dict]:
        return [a for a in self.ann if a["view"] == self.view.name]

    def _objects(self) -> list[int]:
        return sorted({a["object"] for a in self.ann} | {self.obj})

    def _place(self, x: float, y: float, box: list[float] | None) -> None:
        if box is None and self.show_dets:
            d = self.view.detections.get(self.frame_no)
            if d is not None and len(d):
                dist = np.hypot(d[:, 0] - x, d[:, 1] - y)
                j = int(dist.argmin())
                if dist[j] * self.scale <= SNAP_PX:
                    x, y = float(d[j, 0]), float(d[j, 1])
        old = next((a for a in self.ann if a["object"] == self.obj and a["view"] == self.view.name
                    and a["kind"] == self.mode), None)
        if old:
            self.ann.remove(old)
        self.ann.append({"object": self.obj, "view": self.view.name, "kind": self.mode, "frame": self.frame_no,
                         "x": round(x, 2), "y": round(y, 2), "box": box})
        self.status = (f"#{self.obj} {self.mode.upper()} set at frame {self.frame_no}"
                       + (f" (replaced {'suggestion at ' if old.get('suggested') else ''}frame {old['frame']})"
                          if old else ""))

    def _suggest(self, kind: str) -> None:
        if self.view.camera is None or sum(v.camera is not None for v in self.views) < 2:
            self.status = "Suggestions need camera calibrations"
            return
        # Refresh: earlier suggestions of this kind in this view are replaced.
        self.ann = [a for a in self.ann if not (a.get("suggested") and a["view"] == self.view.name
                                                and a["kind"] == kind)]
        new, notes = suggest(self.views, self.vi, self.ann, kind, self.suggest_gate_px)
        self.ann += new
        print(f"\nSuggested {kind.upper()}s for {self.view.name}:" + "".join(f"\n  {n}" for n in notes or ["none"]))
        skipped = len(notes) - len(new)
        self.status = (f"{len(new)} {kind.upper()} suggestion(s)" + (f", {skipped} skipped" if skipped else "")
                       + (" - details in terminal" if notes else " - nothing annotated in other views"))
        if new:
            self.obj, self.mode = new[0]["object"], kind
            self._seek(list(self.view.frame_numbers).index(new[0]["frame"]))

    def _goto_other(self) -> None:
        """Seek to the instant of the selected object's `mode` point in another view
        (hand-placed points first, then the nearest view after this one)."""
        n = len(self.views)
        order = {self.views[(self.vi + k) % n].name: k for k in range(1, n)}
        cands = [a for a in self.ann if a["object"] == self.obj and a["kind"] == self.mode and a["view"] in order]
        if not cands:
            self.status = f"#{self.obj} has no {self.mode.upper()} in another view"
            return
        a = min(cands, key=lambda a: (bool(a.get("suggested")), order[a["view"]]))
        other = next(v for v in self.views if v.name == a["view"])
        note = self._seek_instant(a["frame"] + self._offset(other))
        self.status = f"At #{self.obj} {self.mode.upper()} of {a['view']} (its frame {a['frame']}){note}"

    def _offset_hints(self) -> list[tuple[str, int]]:
        """For the selected object's `mode` point here and in each other view: the offset for
        this view that would put both at the same instant, when they are not already.
        (text, offset) pairs."""
        here = next((a for a in self._view_ann() if a["object"] == self.obj and a["kind"] == self.mode
                     and not a.get("suggested")), None)
        if here is None:
            return []
        off = self._offset(self.view)
        hints = []
        for v in self.views:
            a = next((a for a in self.ann if a["object"] == self.obj and a["kind"] == self.mode
                      and a["view"] == v.name and not a.get("suggested")), None)
            if v is self.view or a is None:
                continue
            want = a["frame"] + self._offset(v) - here["frame"]
            if want != off:
                hints.append((f"#{self.obj} {self.mode.upper()} vs {v.name[:12]}: {want - off:+d} fr"
                              f" -> offset {want:+d} would match", want))
        return hints

    def _delete_near(self, x: float, y: float) -> None:
        here = [a for a in self._view_ann() if a["frame"] == self.frame_no]
        if not here:
            self.status = "Nothing to delete on this frame"
            return
        a = min(here, key=lambda a: np.hypot(a["x"] - x, a["y"] - y))
        self.ann.remove(a)
        self.status = f"Deleted #{a['object']} {a['kind'].upper()} (frame {a['frame']})"

    # ------------------------------------------------------------------ input

    def _on_key(self, key: int) -> None:
        if self.offset_input is not None:
            ch = key & 0xFF
            if ch in (13, 10):
                if self.offset_input not in ("", "-", "+"):
                    self._set_offset(int(self.offset_input))
                self.offset_input = None
            elif ch in (8, 127):
                self.offset_input = self.offset_input[:-1]
            elif ch == 27:
                self.offset_input = None
            elif ch in (ord("-"), ord("+")) and not self.offset_input:
                self.offset_input += chr(ch)
            elif ord("0") <= ch <= ord("9"):
                self.offset_input += chr(ch)
            return
        if self.frame_input is not None:
            ch = key & 0xFF
            if ch in (13, 10):
                if self.frame_input:
                    target = int(self.frame_input)
                    nums = np.asarray(self.view.frame_numbers)
                    self._seek(int(np.abs(nums - target).argmin()))
                self.frame_input = None
            elif ch in (8, 127):
                self.frame_input = self.frame_input[:-1]
            elif ch == 27:
                self.frame_input = None
            elif ord("0") <= ch <= ord("9"):
                self.frame_input += chr(ch)
            return
        if key in KEY_LEFT:
            return self._seek(self.pos - 1)
        if key in KEY_RIGHT:
            return self._seek(self.pos + 1)
        ch = key & 0xFF
        actions = {
            ord("a"): lambda: self._seek(self.pos - 1), ord("d"): lambda: self._seek(self.pos + 1),
            ord("A"): lambda: self._seek(self.pos - 10), ord("D"): lambda: self._seek(self.pos + 10),
            ord("s"): lambda: self._set_mode("start"), ord("e"): lambda: self._set_mode("end"),
            ord("n"): self._new_object, ord("["): lambda: self._step_object(-1),
            ord("]"): lambda: self._step_object(1), ord("h"): self._toggle_dets, ord("l"): self._toggle_epipolar,
            ord("S"): lambda: self._suggest("start"), ord("E"): lambda: self._suggest("end"),
            ord("g"): self._goto_other,
            ord(","): lambda: self._nudge_offset(-1), ord("."): lambda: self._nudge_offset(1),
            ord("<"): lambda: self._nudge_offset(-10), ord(">"): lambda: self._nudge_offset(10),
            ord("o"): self._start_offset_input, ord("O"): self._apply_hint, ord("W"): self._save_defaults,
            13: self._continue, 10: self._continue, 8: self._back, 27: self._cancel,
        }
        if ord("1") <= ch <= ord("9"):
            self.obj = ch - ord("0")
            self.status = f"Object #{self.obj} selected"
        elif ch in actions:
            actions[ch]()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.status = f"Mode: {mode.upper()}"

    def _new_object(self) -> None:
        self.obj = max({a["object"] for a in self.ann} | {0}) + 1   # first unused number
        self.mode = "start"
        self.status = f"New object #{self.obj} (mode START)"

    def _has_any(self, obj: int) -> bool:
        return any(a["object"] == obj for a in self.ann)

    def _step_object(self, step: int) -> None:
        objs = self._objects()
        self.obj = objs[(objs.index(self.obj) + step) % len(objs)]
        self.status = f"Object #{self.obj} selected"

    def _toggle_dets(self) -> None:
        self.show_dets = not self.show_dets

    def _toggle_epipolar(self) -> None:
        self.show_epipolar = not self.show_epipolar
        self.status = f"Epipolar lines {'shown' if self.show_epipolar else 'hidden'}"

    def _continue(self) -> None:
        if self.vi + 1 < len(self.views):
            self._open_view(self.vi + 1)
        else:
            self.done = True

    def _back(self) -> None:
        if self.vi > 0:
            self._open_view(self.vi - 1)

    def _cancel(self) -> None:
        self.done = False

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            if self._in_viewport(x, y):
                ix, iy = self._to_img(x, y)
                self.zoom = float(np.clip(self.zoom * (1.2 if delta > 0 else 1 / 1.2), 1.0, 12.0))
                self.offset = [ix - x / self.scale, iy - y / self.scale]
                self._clamp_offset()
            elif x >= self.vp[0]:
                self._scroll_table(-1 if delta > 0 else 1)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (x1, y1, x2, y2) in self.buttons.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return self._button(name)
            for (x1, y1, x2, y2), a in self.rows:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.obj, self.mode = a["object"], a["kind"]
                    nums = list(self.view.frame_numbers)
                    if a["frame"] in nums:
                        self._seek(nums.index(a["frame"]))
                    return
            if self._in_timeline(x, y):
                self.seeking = True
                self._seek_to_x(x)
            elif self._in_viewport(x, y):
                self.drag_start = self.drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.seeking:
                self._seek_to_x(x)
            elif self.drag_start:
                self.drag_now = (int(np.clip(x, 0, self.vp[0] - 1)), int(np.clip(y, 0, self.vp[1] - 1)))
            elif self.pan:
                (px, py), (ox, oy) = self.pan
                self.offset = [ox - (x - px) / self.scale, oy - (y - py) / self.scale]
                self._clamp_offset()
        elif event == cv2.EVENT_LBUTTONUP:
            if self.seeking:
                self.seeking = False
            elif self.drag_start:
                (x0, y0), (x1, y1) = self.drag_start, self.drag_now
                self.drag_start = self.drag_now = None
                if max(abs(x1 - x0), abs(y1 - y0)) < DRAG_MIN_PX:
                    self._place(*self._to_img(x0, y0), None)
                else:
                    (ax, ay), (bx, by) = self._to_img(x0, y0), self._to_img(x1, y1)
                    box = [round(min(ax, bx), 1), round(min(ay, by), 1), round(abs(bx - ax), 1), round(abs(by - ay), 1)]
                    self._place(box[0] + box[2] / 2, box[1] + box[3] / 2, box)
        elif event == cv2.EVENT_RBUTTONDOWN:
            for (x1, y1, x2, y2), a in self.rows:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.ann.remove(a)
                    self.status = f"Deleted #{a['object']} {a['kind'].upper()} (frame {a['frame']})"
                    return
            if self._in_viewport(x, y):
                self._delete_near(*self._to_img(x, y))
        elif event == cv2.EVENT_MBUTTONDOWN and self._in_viewport(x, y):
            self.pan = ((x, y), tuple(self.offset))
        elif event == cv2.EVENT_MBUTTONUP:
            self.pan = None

    def _seek_to_x(self, x: int) -> None:
        n = len(self.source)
        self._seek(round(np.clip(x / max(1, self.vp[0] - 1), 0, 1) * (n - 1)))

    def _button(self, name: str) -> None:
        if name.startswith("obj:"):
            self.obj = int(name[4:])
            self.status = f"Object #{self.obj} selected"
            return
        {"start": lambda: self._set_mode("start"), "end": lambda: self._set_mode("end"),
         "new": self._new_object, "prev_obj": lambda: self._step_object(-1),
         "next_obj": lambda: self._step_object(1), "m10": lambda: self._seek(self.pos - 10),
         "m1": lambda: self._seek(self.pos - 1), "p1": lambda: self._seek(self.pos + 1),
         "p10": lambda: self._seek(self.pos + 10), "goto": self._start_input, "dets": self._toggle_dets, "epipolar": self._toggle_epipolar,
         "clear_view": self._clear_view, "goto_other": self._goto_other, "sugg_start": lambda: self._suggest("start"),
         "sugg_end": lambda: self._suggest("end"), "back": self._back, "continue": self._continue,
         "cancel": self._cancel, "off_m10": lambda: self._nudge_offset(-10),
         "off_m1": lambda: self._nudge_offset(-1), "off_p1": lambda: self._nudge_offset(1),
         "off_p10": lambda: self._nudge_offset(10), "off_edit": self._start_offset_input,
         "off_hint": self._apply_hint, "off_save": self._save_defaults,
         "tbl_up": lambda: self._scroll_table(-1), "tbl_dn": lambda: self._scroll_table(1)}[name]()

    def _start_input(self) -> None:
        self.offset_input = None
        self.frame_input = ""

    def _start_offset_input(self) -> None:
        if self.view.camera is None:
            self.status = "This view has no camera - no frame offset"
            return
        self.frame_input = None
        self.offset_input = ""
        self.status = "Type the offset (- for negative), Enter to apply, Esc to cancel"

    def _nudge_offset(self, step: int) -> None:
        if self.view.camera is not None:
            self._set_offset(self.view.camera.frame_offset + step)
        else:
            self._set_offset(0)

    def _clear_view(self) -> None:
        n = len(self._view_ann())
        self.ann = [a for a in self.ann if a["view"] != self.view.name]
        self.status = f"Cleared {n} annotation(s) in {self.view.name}"

    # ------------------------------------------------------------------ drawing

    def _render(self) -> np.ndarray:
        W, H = self.canvas_size
        canvas = np.full((H, W, 3), BG, np.uint8)
        self.img = self.source.read(self.pos)
        self._clamp_offset()
        self._draw_viewport(canvas)
        self._draw_timeline(canvas)
        self._draw_panel(canvas)
        return canvas

    def _draw_image(self, canvas: np.ndarray) -> np.ndarray:
        """The (zoomed / panned) frame into the viewport, with the detections if shown.
        Returns the viewport region of the canvas, to draw on in display pixels."""
        h, w = self.img.shape[:2]
        x0, y0 = int(self.offset[0]), int(self.offset[1])
        x1 = min(w, int(np.ceil(self.offset[0] + self.vp[0] / self.scale)))
        y1 = min(h, int(np.ceil(self.offset[1] + self.vp[1] / self.scale)))
        crop = self.img[y0:y1, x0:x1]
        interp = cv2.INTER_NEAREST if self.scale > 2 else cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        view = cv2.resize(crop, self.vp, interpolation=interp)
        # resize stretches the integer crop to the viewport: match _to_disp to it
        self.offset = [float(x0), float(y0)]
        canvas[:self.vp[1], :self.vp[0]] = view

        vp = canvas[:self.vp[1], :self.vp[0]]
        if self.show_dets:
            d = self.view.detections.get(self.frame_no)
            for x, y in (d[:, :2] if d is not None else []):
                p = self._to_disp(x, y)
                cv2.circle(vp, p, 9, (210, 210, 210), 1, cv2.LINE_AA)   # no centre dot: it hides the fly
        return vp

    def _draw_viewport(self, canvas: np.ndarray) -> None:
        vp = self._draw_image(canvas)
        self._draw_epipolar(vp)
        for a in self._view_ann():
            col = object_colour(a["object"])
            p = self._to_disp(a["x"], a["y"])
            q = "?" if a.get("suggested") else ""
            if a["frame"] != self.frame_no:
                if a["object"] == self.obj:  # ghost of the selected object's other annotations
                    cv2.circle(vp, p, 5, col, 1, cv2.LINE_AA)
                    cv2.putText(vp, f"{a['kind'][0].upper()}{a['frame']}{q}", (p[0] + 7, p[1] + 4), FONT,
                                SMALL_SCALE, col, 1, cv2.LINE_AA)
                continue
            if a.get("box"):
                bx, by, bw, bh = a["box"]
                cv2.rectangle(vp, self._to_disp(bx, by), self._to_disp(bx + bw, by + bh), col, 1, cv2.LINE_AA)
            kind_c = START_C if a["kind"] == "start" else END_C
            draw_cross(vp, p, kind_c, tilted=a["kind"] != "start")
            cv2.circle(vp, p, 14, col, 1, cv2.LINE_AA)
            self._label(vp, f"#{a['object']} {a['kind'].upper()}{q}", (p[0] + 17, p[1] - 10), col, LABEL_SCALE)
        if self.drag_start and self.drag_now:
            cv2.rectangle(vp, self.drag_start, self.drag_now, object_colour(self.obj), 1, cv2.LINE_AA)
        self._label(vp, f"{self.view.name}   frame {self.frame_no}   ({self.pos + 1}/{len(self.source)})"
                        f"{'   zoom x%.1f' % self.zoom if self.zoom > 1 else ''}", (10, 22), TEXT, 0.5)

    def _draw_epipolar(self, vp: np.ndarray) -> None:
        """Epipolar lines of the selected object's points in the other views, on their frames."""
        cam = self.view.camera
        if cam is None or not self.show_epipolar:
            return
        h, w = self.img.shape[:2]
        for a in self.ann:
            if a["object"] != self.obj or a["view"] == self.view.name or a.get("suggested"):
                continue
            other = next((v for v in self.views if v.name == a["view"]), None)
            if other is None or other.camera is None or aligned(a["frame"], other.camera) != aligned(self.frame_no, cam):
                continue
            pts = epipolar_polyline(other.camera, (a["x"], a["y"]), cam, (w, h))
            disp = np.array([self._to_disp(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)], np.int32)
            if len(disp) >= 2:
                cv2.polylines(vp, [disp], False, object_colour(a["object"]), 1, cv2.LINE_AA)
                self._label(vp, f"{a['view']} {a['kind']}", tuple(disp[len(disp) // 2] + [6, -6]),
                            object_colour(a["object"]), SMALL_SCALE)

    @staticmethod
    def _label(img, text, org, colour, scale=LABEL_SCALE) -> None:
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), (20, 20, 20), -1)
        cv2.putText(img, text, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)

    def _draw_timeline(self, canvas: np.ndarray) -> None:
        top, w, n = self.vp[1], self.vp[0], len(self.source)
        cv2.rectangle(canvas, (0, top), (w, top + TIMELINE_H), PANEL, -1)
        mid = top + TIMELINE_H // 2
        cv2.line(canvas, (8, mid), (w - 8, mid), (90, 90, 90), 3)
        to_x = lambda pos: int(8 + (w - 16) * pos / max(1, n - 1))  # noqa: E731
        nums = {f: i for i, f in enumerate(self.view.frame_numbers)}
        det_frames = [nums[f] for f, d in self.view.detections.items() if f in nums and len(d)]
        for pos in det_frames:  # faint strip: frames with detections
            cv2.line(canvas, (to_x(pos), mid - 2), (to_x(pos), mid + 2), (120, 120, 120), 1)
        for a in self._view_ann():
            if a["frame"] in nums:
                x = to_x(nums[a["frame"]])
                y0, y1 = (top + 6, mid) if a["kind"] == "start" else (mid, top + TIMELINE_H - 6)
                cv2.line(canvas, (x, y0), (x, y1), object_colour(a["object"]), 3)
        x = to_x(self.pos)
        cv2.rectangle(canvas, (x - 2, top + 3), (x + 2, top + TIMELINE_H - 3), (255, 255, 255), -1)
        cv2.putText(canvas, str(self.view.frame_numbers[0]), (8, top + TIMELINE_H - 4), FONT, 0.38, DIM, 1)
        last = str(self.view.frame_numbers[-1])
        cv2.putText(canvas, last, (w - 8 - 8 * len(last), top + TIMELINE_H - 4), FONT, 0.38, DIM, 1)

    def _button_draw(self, canvas, name, x, y, w, h, text, colour=(70, 70, 70), active=False, scale=0.45) -> None:
        self.buttons[name] = (x, y, x + w, y + h)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour if not active else ACCENT, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (120, 120, 120) if not active else (255, 255, 255), 1)
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
        cv2.putText(canvas, text, (x + (w - tw) // 2, y + (h + th) // 2), FONT, scale, TEXT, 1, cv2.LINE_AA)

    def _table_rows(self, n: int, fit: int) -> int:
        """How many of a table's n rows are drawn when `fit` fit; clamps table_scroll to match."""
        shown = min(n, fit)
        self.table_scroll = min(self.table_scroll, max(0, n - shown))
        return shown

    def _scroll_table(self, step: int) -> None:
        self.table_scroll = max(0, self.table_scroll + step)   # the upper bound is clamped when drawn

    def _draw_table_scroll(self, canvas: np.ndarray, L: int, y_head: int, y: int, bw: int, n: int, fit: int,
                           h: int) -> int:
        """When a table's n rows overflow the `fit` that fit: an up / down column at its right (the mouse
        wheel over the panel scrolls too) and "rows a-b of n" on its header line. Returns the rows' width."""
        if n <= fit:
            return bw
        s, W = self.table_scroll, SCROLL_W
        text = f"{s + 1}-{s + fit} of {n}"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.4, 1)
        cv2.putText(canvas, text, (L + bw - tw, y_head), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
        x, half = L + bw - W, max(10, (fit * h - 3) // 2)
        for name, y0, up, live in (("tbl_up", y, True, s > 0), ("tbl_dn", y + fit * h - 3 - half, False, s + fit < n)):
            self._button_draw(canvas, name, x, y0, W, half, "")
            cx, cy, r = x + W // 2, y0 + half // 2, min(7, half // 2 - 2)
            tri = [(cx - r, cy + r // 2), (cx + r, cy + r // 2), (cx, cy - r // 2 - 2)] if up else \
                  [(cx - r, cy - r // 2), (cx + r, cy - r // 2), (cx, cy + r // 2 + 2)]
            cv2.fillPoly(canvas, [np.array(tri, np.int32)], TEXT if live else (100, 100, 100), cv2.LINE_AA)
        return bw - W - 6

    def _draw_offset_controls(self, canvas: np.ndarray, L: int, y: int, bw: int) -> int:
        """This view's frame offset: -10 -1 [offset box] +1 +10, then "use suggested" / "save as
        default", then how it compares to cameras.json. Returns the height used."""
        if self.view.camera is None:
            cv2.putText(canvas, "no camera for this view - no frame offset", (L, y + 18), FONT, 0.4, DIM, 1,
                        cv2.LINE_AA)
            return 26
        off = self.view.camera.frame_offset
        sw, h = 44, 26
        for i, (name, text) in enumerate((("off_m10", "-10"), ("off_m1", "-1"))):
            self._button_draw(canvas, name, L + i * (sw + 6), y, sw, h, text)
        ex = L + 2 * (sw + 6)
        ew = bw - 4 * (sw + 6) + 6
        txt = (self.offset_input + "_") if self.offset_input is not None else f"frame offset {off:+d}  (o: type)"
        self._button_draw(canvas, "off_edit", ex, y, ew, h, txt, (60, 60, 90), self.offset_input is not None)
        for i, (name, text) in enumerate((("off_p1", "+1"), ("off_p10", "+10"))):
            self._button_draw(canvas, name, ex + ew + 6 + i * (sw + 6), y, sw, h, text)
        y += h + 4
        half = (bw - 10) // 2
        hints = self._offset_hints()
        self._button_draw(canvas, "off_hint", L, y, half, 24,
                          f"use suggested {hints[0][1]:+d} (O)" if hints else "use suggested (O)",
                          (40, 80, 110) if hints else (70, 70, 70))
        self._button_draw(canvas, "off_save", L + half + 10, y, half, 24, "save as default (W)", (70, 70, 70))
        changed = [v.name[:12] for v in self.views if v.camera is not None
                   and v.camera.frame_offset != self.default_offsets.get(v.name, 0)]
        default = self.default_offsets.get(self.view.name, 0)
        note = (f"cameras.json: {default:+d}" + (f"   changed: {', '.join(changed)}" if changed else "   (all unchanged)"))
        cv2.putText(canvas, note[:62], (L, y + 24 + 16), FONT, 0.38, (120, 200, 255) if changed else DIM, 1,
                    cv2.LINE_AA)
        return h + 4 + 24 + 28

    def _draw_panel(self, canvas: np.ndarray) -> None:
        x0, W, H = self.vp[0], PANEL_W, self.canvas_size[1]
        cv2.rectangle(canvas, (x0, 0), (x0 + W, H), PANEL, -1)
        self.buttons, self.rows = {}, []
        L = x0 + 14
        bw = W - 28

        y = 28
        cv2.putText(canvas, self.title or "Guided tracking annotations", (L, y), FONT, 0.5, DIM, 1, cv2.LINE_AA)
        y += 30
        cv2.putText(canvas, f"View {self.vi + 1} of {len(self.views)}: {self.view.name}", (L, y), FONT, 0.65,
                    TEXT, 2, cv2.LINE_AA)
        y += 20
        cv2.putText(canvas, self.view.label[-48:], (L, y), FONT, 0.38, DIM, 1, cv2.LINE_AA)
        y += 8
        y += self._draw_offset_controls(canvas, L, y, bw) + 6
        cv2.putText(canvas, "same instant in:", (L, y), FONT, 0.38, DIM, 1, cv2.LINE_AA)
        y += 16
        same = []
        for v in self.views:
            if v is not self.view:
                f = self._frame_at(v, self._instant())
                inside = v.frame_numbers[0] <= f <= v.frame_numbers[-1]
                same.append(f"{v.name[:12]} {f if inside else 'none'}")
        cv2.putText(canvas, "   ".join(same)[:62], (L, y), FONT, 0.38, TEXT, 1, cv2.LINE_AA)
        for hint, _ in self._offset_hints()[:2]:
            y += 16
            cv2.putText(canvas, hint[:62], (L, y), FONT, 0.38, (120, 200, 255), 1, cv2.LINE_AA)
        for i in range(len(self.views)):   # progress dots
            c = ACCENT if i == self.vi else (TEXT if self.visited[i] else (90, 90, 90))
            cv2.circle(canvas, (x0 + W - 20 - 18 * (len(self.views) - 1 - i), 52), 6, c, -1, cv2.LINE_AA)

        y += 18
        half = (bw - 10) // 2
        self._button_draw(canvas, "start", L, y, half, 34, "START  (s)", (40, 90, 40), self.mode == "start")
        self._button_draw(canvas, "end", L + half + 10, y, half, 34, "END  (e)", (40, 40, 110), self.mode == "end")

        y += 50
        cv2.putText(canvas, "Object  ([ ]  1-9)", (L, y + 4), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        y += 12
        x = L
        for obj in self._objects()[:9]:
            self._button_draw(canvas, f"obj:{obj}", x, y, 34, 30, str(obj), object_colour(obj), obj == self.obj)
            x += 40
        self._button_draw(canvas, "new", min(x, L + bw - 70), y, 70, 30, "+ new (n)")

        y += 46
        cv2.putText(canvas, "Frame  (a / d   A / D   arrows)", (L, y + 4), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        y += 12
        sw = 44
        for i, (name, text) in enumerate((("m10", "-10"), ("m1", "-1"))):
            self._button_draw(canvas, name, L + i * (sw + 6), y, sw, 30, text)
        gx = L + 2 * (sw + 6)
        gw = bw - 4 * (sw + 6) + 6
        txt = (self.frame_input + "_") if self.frame_input is not None else f"go to {self.frame_no}"
        self._button_draw(canvas, "goto", gx, y, gw, 30, txt, (60, 60, 90), self.frame_input is not None)
        for i, (name, text) in enumerate((("p1", "+1"), ("p10", "+10"))):
            self._button_draw(canvas, name, gx + gw + 6 + i * (sw + 6), y, sw, 30, text)

        y += 44
        self._button_draw(canvas, "dets", L, y, half, 28, f"detections: {'on' if self.show_dets else 'off'} (h)",
                          active=False)
        self._button_draw(canvas, "epipolar", L + half + 10, y, half, 28,
                          f"epipolar lines: {'on' if self.show_epipolar else 'off'} (l)")
        y += 34
        self._button_draw(canvas, "sugg_start", L, y, half, 28, "suggest starts (S)", (40, 90, 40))
        self._button_draw(canvas, "sugg_end", L + half + 10, y, half, 28, "suggest ends (E)", (40, 40, 110))
        y += 34
        self._button_draw(canvas, "goto_other", L, y, half, 24, f"#{self.obj} {self.mode} elsewhere (g)")
        self._button_draw(canvas, "clear_view", L + half + 10, y, half, 24, "clear this view", (50, 50, 110))

        # annotations table (this view)
        y += 42
        cv2.putText(canvas, "This view  (click: go to  |  right click: delete)", (L, y), FONT, 0.42, DIM, 1,
                    cv2.LINE_AA)
        rows = sorted(self._view_ann(), key=lambda a: (a["object"], a["kind"] != "start"))
        footer_top = H - 196
        row_h = 24
        y_head, y = y, y + 8
        max_rows = self._table_rows(len(rows), max(1, (footer_top - 120 - y) // row_h))
        rw = self._draw_table_scroll(canvas, L, y_head, y, bw, len(rows), max_rows, row_h)
        if not rows:
            cv2.putText(canvas, "none yet - click on the fly in the image", (L + 6, y + 18), FONT, 0.42,
                        (120, 120, 120), 1, cv2.LINE_AA)
        for i, a in enumerate(rows[self.table_scroll:self.table_scroll + max_rows]):
            ry = y + i * row_h
            sel = a["object"] == self.obj and a["kind"] == self.mode
            here = a["frame"] == self.frame_no
            cv2.rectangle(canvas, (L, ry), (L + rw, ry + row_h - 3), (75, 75, 75) if sel else (55, 55, 55), -1)
            cv2.rectangle(canvas, (L, ry), (L + 8, ry + row_h - 3), object_colour(a["object"]), -1)
            text = (f"#{a['object']}  {a['kind'].upper():<5}  frame {a['frame']:<6}"
                    f"({a['x']:.0f}, {a['y']:.0f}){'  box' if a.get('box') else ''}"
                    f"{'  suggested' if a.get('suggested') else ''}")
            cv2.putText(canvas, text, (L + 16, ry + 15), FONT, 0.42, (255, 255, 255) if here else TEXT, 1,
                        cv2.LINE_AA)
            self.rows.append(((L, ry, L + rw, ry + row_h - 3), a))

        # all objects across views: start / end frame per view
        y = footer_top - 118
        cv2.putText(canvas, "All views  (start / end frame)", (L, y), FONT, 0.42, DIM, 1, cv2.LINE_AA)
        col_w = (bw - 34) // max(1, len(self.views))
        for i, v in enumerate(self.views):
            cv2.putText(canvas, v.name[:12], (L + 34 + i * col_w, y + 20), FONT, 0.38,
                        TEXT if i == self.vi else DIM, 1, cv2.LINE_AA)
        for r, obj in enumerate(o for o in self._objects() if self._has_any(o)):
            ry = y + 40 + r * 19
            if r >= 4:
                cv2.putText(canvas, "...", (L, ry), FONT, 0.4, DIM, 1)
                break
            cv2.circle(canvas, (L + 5, ry - 4), 5, object_colour(obj), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"#{obj}", (L + 12, ry), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
            for i, v in enumerate(self.views):
                fr = {a["kind"]: a["frame"] for a in self.ann if a["object"] == obj and a["view"] == v.name}
                cv2.putText(canvas, f"{fr.get('start', '-')} / {fr.get('end', '-')}", (L + 34 + i * col_w, ry),
                            FONT, 0.4, TEXT, 1, cv2.LINE_AA)

        # footer: navigation, help, status
        y = footer_top
        third = (bw - 20) // 3
        last = self.vi == len(self.views) - 1
        self._button_draw(canvas, "back", L, y, third, 36, "< back", (70, 70, 70))
        self._button_draw(canvas, "cancel", L + third + 10, y, third, 36, "cancel (esc)", (50, 50, 110))
        self._button_draw(canvas, "continue", L + 2 * (third + 10), y, third, 36,
                          "finish (enter)" if last else "continue >", (40, 110, 40))
        help_lines = [f"circles: {self.detections_label}"[:60] + (" (clicks snap to them)" if self.show_dets else " (hidden)"),
                      "left click: point (snaps to detection)   drag: box",
                      "right click: delete   wheel: zoom   middle drag: pan",
                      "timeline: click / drag to seek"]
        for i, line in enumerate(help_lines):
            cv2.putText(canvas, line, (L, y + 62 + i * 18), FONT, 0.38, DIM, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (L, H - 44), (L + bw, H - 14), (30, 30, 30), -1)
        cv2.putText(canvas, self.status[:58], (L + 8, H - 24), FONT, 0.42, (180, 220, 255), 1, cv2.LINE_AA)


def run_annotation_gui(views: list[ViewInput], annotations: list[dict] | None = None, title: str = "",
                       detections_label: str = "detections", suggest_gate_px: float = 15.0,
                       default_offsets: dict[str, int] | None = None,
                       save_defaults: Callable[[dict[str, int]], str] | None = None
                       ) -> tuple[list[dict] | None, dict[str, int]]:
    """Annotate every view in turn. Returns (annotations, or None if cancelled; the frame offsets
    the views ended with, camera -> offset). The views' cameras are not changed.
    detections_label says what the detection circles are (e.g. the model name).
    suggest_gate_px: how far (pixels) a detection may be from the epipolar line / reprojected
    point to be suggested (views need a .camera for suggestions).
    default_offsets: cameras.json's offsets, shown next to the current ones (default: the views' own).
    save_defaults(offsets) -> status message: called by "save (W)" to make the offsets the defaults."""
    gui = AnnotationGUI(views, annotations, title, detections_label, suggest_gate_px, default_offsets, save_defaults)
    result = gui.run()
    return result, gui.offsets
