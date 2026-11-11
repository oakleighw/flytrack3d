"""Correction annotator for a past factor-graph result (step 10, main_human_correction.py).

The same window as the start / end annotator (annotation_tool.py), which it extends: the
views of the session one after another, the same zoom, pan, seeking and epipolar lines.
Instead of START / END points you place CORRECTION points: where the selected track's fly
really is, in any view, at any frame, as many as needed. They become trusted anchors of a
new factor-graph solve of that track (tracking3d/correction.py), like the start / end clicks.

    image      circles            YOLO detections (clicks snap to them)
               yellow ring        the selected track here: dot on top of the ring = married (fitted to a
                                  detection in this view), no dot = inferred by the motion prior;
                                  red ring and dot = low confidence (sigma > the low-confidence threshold)
               yellow line        the selected track over the frames around this one (t)
               small rings        the other tracks, in their colours
               green ring / line  the selected track as corrected by an earlier run of this step
               green cross        a correction (small green circles: the track's corrections on
                                  other frames, with their frame number)
               + / x              the source result's annotated START / END (step 8, fixed here),
                                  in the track's colour (also as ticks on the timeline)
               left click         correction for the selected track on this frame (replaces its
                                  correction on this frame, if any)
               left drag          the same with a box (its centre is used)
               right click        delete the nearest correction on this frame
               wheel / middle     zoom / pan
    timeline   white              frames with detections in this view
               yellow             frames of the selected track: bright = married here, dim = inferred
               red                frames where the selected track has low confidence
               green (thin)       the earlier corrected track; green ticks = corrections
    keys       a / d   A / D      previous / next frame, -10 / +10 (also the arrow keys)
               1-9, [ / ]         select a track, previous / next track
               r / R              start of the next / previous low-confidence (red) stretch
               c / C              next / previous correction of the selected track in this view
               - / +              lower / raise the low-confidence threshold
               h / l / t          show / hide detections, epipolar lines, the track's trail
               m                  show / hide the corrections and START / END on the image (the
                                  track, the earlier correction and the detections stay), to see
                                  the fly under them; clicks still place corrections
               S                  suggest a correction here from the selected track's corrections
                                  at this instant in other views (epipolar geometry)
               Enter              continue to the next view (finish on the last)
               Backspace          back to the previous view
               Esc                cancel everything

Frame offsets are those the source result was made with, and cannot be changed here: the
tracks shown were built with them.

Corrections are returned as plain dicts, like the start / end annotations:
    {"object": 1, "view": "344-above", "kind": "correction", "frame": 914,
     "x": 1032.5, "y": 455.0, "box": [x, y, w, h] or None}
with x, y in raw (distorted) pixels, frame the label frame number and object the track id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .annotation_tool import (ACCENT, DIM, FONT, LABEL_SCALE, PANEL, PANEL_W, SMALL_SCALE, SNAP_PX, TEXT,
                              TIMELINE_H, KEY_LEFT, KEY_RIGHT, AnnotationGUI, ViewInput, draw_cross, object_colour)
from .suggest import aligned, suggest

KIND = "correction"
# BGR
TRACK_C = (0, 220, 255)       # yellow: the selected track, married
TRACK_DIM = (0, 125, 150)     # dim yellow: inferred
LOW_C = (50, 50, 235)         # red: low confidence
CORR_C = (60, 210, 60)        # green: corrections and the corrected track
DET_TICK = (225, 225, 225)    # white: frames with detections
TRAIL = 40                    # frames either side drawn as the selected track's trail
SIGMA_STEP = 1.5              # - / + scale the low-confidence threshold by this


@dataclass
class TrackView:
    """A factor-graph track as seen in one view, per label frame number of that view."""
    frames: np.ndarray            # (N,) label frame numbers, ascending
    xy: np.ndarray                # (N, 2) raw pixels; nan where behind the camera or outside the image
    married: np.ndarray           # (N,) bool: fitted to a detection in this view on that frame
    sigma: np.ndarray             # (N,) 1-sigma 3D uncertainty (world units), nan if unknown
    index: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {int(f): i for i, f in enumerate(self.frames)}

    def at(self, frame: int) -> int | None:
        return self.index.get(int(frame))

    def low(self, threshold: float) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.nan_to_num(self.sigma, nan=-1.0) > threshold


def low_stretches(low: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of every run of True."""
    if not low.any():
        return []
    d = np.diff(np.r_[0, low.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


class CorrectionGUI(AnnotationGUI):
    window = "Correct factor-graph tracks"

    def __init__(self, views: list[ViewInput], tracks: dict[int, dict[str, TrackView]],
                 corrections: list[dict] | None = None, fixed: list[dict] | None = None,
                 corrected: dict[int, dict[str, TrackView]] | None = None, title: str = "",
                 detections_label: str = "detections", suggest_gate_px: float = 15.0, low_sigma: float = 10.0):
        super().__init__(views, corrections, title, detections_label, suggest_gate_px)
        self.tracks = tracks                 # source result: track id -> view name -> TrackView
        self.corrected = corrected or {}     # an earlier corrected result, same layout (green)
        self.fixed = fixed or []             # the source's start / end annotations (shown, not editable)
        self.low_sigma = float(low_sigma)
        self.show_trail = True
        self.show_marks = True              # corrections and START / END on the image (m)
        self.mode = KIND
        self.obj = min(tracks) if tracks else 1
        self._strips: dict[tuple, np.ndarray] = {}
        self._pos_of: dict[int, dict[int, int]] = {}
        # Open on the first low-confidence stretch of the first track, else on its first frame.
        tv = self._tv(view=self.views[0])
        if tv is not None and len(tv.frames):
            runs = low_stretches(tv.low(self.low_sigma))
            frame = int(tv.frames[runs[0][0] if runs else 0])
            pos = self._pos(frame, 0)
            if pos is not None:
                self.positions[0] = pos

    def run(self) -> list[dict] | None:
        result = super().run()
        return None if result is None else sorted(result, key=lambda a: (a["object"], a["view"], a["frame"]))

    # ------------------------------------------------------------------ lookups

    def _pos(self, frame: int, vi: int | None = None) -> int | None:
        """Position in the video of a label frame number (None if it is not in it)."""
        vi = self.vi if vi is None else vi
        if vi not in self._pos_of:
            self._pos_of[vi] = {int(f): i for i, f in enumerate(self.views[vi].frame_numbers)}
        return self._pos_of[vi].get(int(frame))

    def _tv(self, obj: int | None = None, view: ViewInput | None = None, source: dict | None = None
            ) -> TrackView | None:
        source = self.tracks if source is None else source
        return source.get(self.obj if obj is None else obj, {}).get((view or self.view).name)

    def _objects(self) -> list[int]:
        return sorted(set(self.tracks) | {a["object"] for a in self.ann})

    def _seek_frame(self, frame: int) -> None:
        pos = self._pos(frame)
        if pos is None:
            nums = np.asarray(self.view.frame_numbers)
            pos = int(np.abs(nums - frame).argmin())
        self._seek(pos)

    # ------------------------------------------------------------------ editing

    def _place(self, x: float, y: float, box: list[float] | None) -> None:
        if box is None and self.show_dets:
            d = self.view.detections.get(self.frame_no)
            if d is not None and len(d):
                dist = np.hypot(d[:, 0] - x, d[:, 1] - y)
                j = int(dist.argmin())
                if dist[j] * self.scale <= SNAP_PX:
                    x, y = float(d[j, 0]), float(d[j, 1])
        old = self._here(self.obj)
        if old:
            self.ann.remove(old)
        self.ann.append({"object": self.obj, "view": self.view.name, "kind": KIND, "frame": self.frame_no,
                         "x": round(x, 2), "y": round(y, 2), "box": box})
        self.status = f"#{self.obj} correction at frame {self.frame_no}" + (" (replaced)" if old else "")

    def _here(self, obj: int) -> dict | None:
        return next((a for a in self.ann if a["object"] == obj and a["view"] == self.view.name
                     and a["frame"] == self.frame_no), None)

    def _suggest_here(self) -> None:
        cam = self.view.camera
        if cam is None:
            self.status = "Suggestions need camera calibrations"
            return
        index = {v.name: v for v in self.views}
        others = [a for a in self.ann if a["object"] == self.obj and a["view"] != self.view.name
                  and a["view"] in index and index[a["view"]].camera is not None and not a.get("suggested")
                  and aligned(a["frame"], index[a["view"]].camera) == self._instant()]
        if not others:
            self.status = f"No #{self.obj} correction at this instant in another view to suggest from"
            return
        new, notes = suggest(self.views, self.vi, others, KIND, self.suggest_gate_px)
        print(f"\nSuggested correction for #{self.obj} in {self.view.name}:"
              + "".join(f"\n  {n}" for n in notes or ["none"]))
        if not new:
            self.status = "No suggestion - details in terminal"
            return
        old = self._here(self.obj)
        if old:
            self.ann.remove(old)
        self.ann += new
        self.status = f"#{self.obj} correction suggested (?) - check it, details in terminal"

    def _clear_track_here(self) -> None:
        n = sum(a["object"] == self.obj for a in self._view_ann())
        self.ann = [a for a in self.ann if not (a["object"] == self.obj and a["view"] == self.view.name)]
        self.status = f"Deleted {n} correction(s) of #{self.obj} in {self.view.name}"

    def _jump_low(self, step: int) -> None:
        tv = self._tv()
        runs = low_stretches(tv.low(self.low_sigma)) if tv is not None else []
        if not runs:
            self.status = f"#{self.obj} has no low-confidence frames in {self.view.name} (sigma > {self.low_sigma:g})"
            return
        starts = [int(tv.frames[a]) for a, _ in runs]
        f = self.frame_no
        target = next((s for s in starts if s > f), None) if step > 0 else next((s for s in starts[::-1] if s < f), None)
        if target is None:
            self.status = f"No {'later' if step > 0 else 'earlier'} low-confidence stretch of #{self.obj}"
            return
        k = starts.index(target)
        self._seek_frame(target)
        a, b = runs[k]
        self.status = (f"#{self.obj} low-confidence stretch {k + 1}/{len(runs)}: frames {target}-{int(tv.frames[b])}"
                       f" (max sigma {np.nanmax(tv.sigma[a:b + 1]):.1f})")

    def _jump_correction(self, step: int) -> None:
        frames = sorted(a["frame"] for a in self._view_ann() if a["object"] == self.obj)
        f = self.frame_no
        target = next((x for x in frames if x > f), None) if step > 0 else next((x for x in frames[::-1] if x < f), None)
        if target is None:
            self.status = f"No {'later' if step > 0 else 'earlier'} correction of #{self.obj} in this view"
            return
        self._seek_frame(target)
        self.status = f"#{self.obj} correction at frame {target}"

    def _scale_sigma(self, factor: float) -> None:
        self.low_sigma = float(np.clip(self.low_sigma * factor, 1e-3, 1e6))
        self.status = f"Low confidence: sigma > {self.low_sigma:.3g}"

    def _toggle_trail(self) -> None:
        self.show_trail = not self.show_trail

    def _delete_near(self, x: float, y: float) -> None:
        if not self.show_marks:   # would delete a correction that cannot be seen
            self.status = "Marks hidden - press m before deleting (or use the table)"
            return
        super()._delete_near(x, y)

    def _toggle_marks(self) -> None:
        self.show_marks = not self.show_marks
        self.status = ("Corrections and START / END shown" if self.show_marks else
                       "Marks hidden (m: show) - clicks still place corrections")

    # ------------------------------------------------------------------ input

    def _on_key(self, key: int) -> None:
        if self.frame_input is not None:
            return super()._on_key(key)   # typing a frame number
        if key in KEY_LEFT:
            return self._seek(self.pos - 1)
        if key in KEY_RIGHT:
            return self._seek(self.pos + 1)
        ch = key & 0xFF
        actions = {
            ord("a"): lambda: self._seek(self.pos - 1), ord("d"): lambda: self._seek(self.pos + 1),
            ord("A"): lambda: self._seek(self.pos - 10), ord("D"): lambda: self._seek(self.pos + 10),
            ord("["): lambda: self._step_object(-1), ord("]"): lambda: self._step_object(1),
            ord("r"): lambda: self._jump_low(1), ord("R"): lambda: self._jump_low(-1),
            ord("c"): lambda: self._jump_correction(1), ord("C"): lambda: self._jump_correction(-1),
            ord("-"): lambda: self._scale_sigma(1 / SIGMA_STEP), ord("+"): lambda: self._scale_sigma(SIGMA_STEP),
            ord("="): lambda: self._scale_sigma(SIGMA_STEP),
            ord("h"): self._toggle_dets, ord("l"): self._toggle_epipolar, ord("t"): self._toggle_trail,
            ord("m"): self._toggle_marks,
            ord("S"): self._suggest_here,
            13: self._continue, 10: self._continue, 8: self._back, 27: self._cancel,
        }
        if ord("1") <= ch <= ord("9"):
            self._select(ch - ord("0"))
        elif ch in actions:
            actions[ch]()

    def _select(self, obj: int) -> None:
        if obj not in self._objects():
            self.status = f"The source result has no track #{obj}"
            return
        self.obj = obj
        self.status = f"Track #{obj} selected"

    def _button(self, name: str) -> None:
        if name.startswith("obj:"):
            return self._select(int(name[4:]))
        {"prev_obj": lambda: self._step_object(-1), "next_obj": lambda: self._step_object(1),
         "m10": lambda: self._seek(self.pos - 10), "m1": lambda: self._seek(self.pos - 1),
         "p1": lambda: self._seek(self.pos + 1), "p10": lambda: self._seek(self.pos + 10),
         "goto": self._start_input, "low_prev": lambda: self._jump_low(-1), "low_next": lambda: self._jump_low(1),
         "fix_prev": lambda: self._jump_correction(-1), "fix_next": lambda: self._jump_correction(1),
         "dets": self._toggle_dets, "epipolar": self._toggle_epipolar, "trail": self._toggle_trail, "marks": self._toggle_marks,
         "suggest": self._suggest_here, "sig_m": lambda: self._scale_sigma(1 / SIGMA_STEP),
         "sig_p": lambda: self._scale_sigma(SIGMA_STEP), "sig_box": lambda: None,
         "clear_track": self._clear_track_here, "back": self._back, "continue": self._continue,
         "cancel": self._cancel, "tbl_up": lambda: self._scroll_table(-1),
         "tbl_dn": lambda: self._scroll_table(1)}[name]()

    # ------------------------------------------------------------------ drawing

    def _disp_poly(self, xy: np.ndarray) -> np.ndarray:
        xy = xy[np.isfinite(xy).all(axis=1)]
        return np.array([self._to_disp(x, y) for x, y in xy], np.int32).reshape(-1, 2)

    def _draw_track(self, vp: np.ndarray, tv: TrackView, colour, selected: bool, corrected: bool,
                    tid: int | None = None) -> None:
        i = tv.at(self.frame_no)
        if selected and self.show_trail and len(tv.frames):
            lo = int(np.searchsorted(tv.frames, self.frame_no - TRAIL))
            hi = int(np.searchsorted(tv.frames, self.frame_no + TRAIL, side="right"))
            poly = self._disp_poly(tv.xy[lo:hi])
            if len(poly) >= 2:
                cv2.polylines(vp, [poly], False, colour if corrected else TRACK_DIM, 1, cv2.LINE_AA)
        if i is None or not np.isfinite(tv.xy[i]).all():
            return
        p = self._to_disp(*tv.xy[i])
        if not selected:
            cv2.circle(vp, p, 8, colour, 1, cv2.LINE_AA)
            cv2.putText(vp, f"#{tid}", (p[0] + 10, p[1] + 4), FONT, SMALL_SCALE, colour, 1, cv2.LINE_AA)
            return
        if corrected:
            cv2.circle(vp, p, 11, CORR_C, 1, cv2.LINE_AA)
            self._label(vp, "corrected", (p[0] - 26, p[1] + 28), CORR_C, SMALL_SCALE)
            return
        low = bool(tv.low(self.low_sigma)[i])
        col = LOW_C if low else TRACK_C
        cv2.circle(vp, p, 17, col, 1, cv2.LINE_AA)
        if tv.married[i]:   # on the ring, not the centre: the centre is where the fly should be
            cv2.circle(vp, (p[0], p[1] - 17), 3, col, -1, cv2.LINE_AA)
        sig = f"  sigma {tv.sigma[i]:.1f}" if np.isfinite(tv.sigma[i]) else ""
        self._label(vp, f"#{self.obj} {'married' if tv.married[i] else 'inferred'}{sig}{'  LOW' if low else ''}",
                    (p[0] + 21, p[1] - 14), col)

    def _draw_viewport(self, canvas: np.ndarray) -> None:
        vp = self._draw_image(canvas)
        self._draw_epipolar(vp)
        for tid in self._objects():
            tv = self._tv(tid)
            if tv is not None and tid != self.obj:
                self._draw_track(vp, tv, object_colour(tid), False, False, tid)
        tv = self._tv()
        if tv is not None:
            self._draw_track(vp, tv, TRACK_C, True, False)
        ctv = self._tv(source=self.corrected)
        if ctv is not None:
            self._draw_track(vp, ctv, CORR_C, True, True)

        for a in self.fixed:   # the source result's start / end annotations
            if not self.show_marks or a["view"] != self.view.name or a["frame"] != self.frame_no:
                continue
            p = self._to_disp(a["x"], a["y"])
            start = a["kind"] == "start"
            draw_cross(vp, p, object_colour(a["object"]), tilted=not start)
            self._label(vp, f"#{a['object']} {a['kind'].upper()} (step 8)", (p[0] + 16, p[1] + 22),
                        object_colour(a["object"]), SMALL_SCALE)

        for a in self._view_ann() if self.show_marks else []:
            col = object_colour(a["object"])
            p = self._to_disp(a["x"], a["y"])
            q = "?" if a.get("suggested") else ""
            if a["frame"] != self.frame_no:
                if a["object"] == self.obj:   # this track's corrections on other frames
                    cv2.circle(vp, p, 5, CORR_C, 1, cv2.LINE_AA)
                    cv2.putText(vp, f"{a['frame']}{q}", (p[0] + 7, p[1] + 4), FONT, SMALL_SCALE, CORR_C, 1,
                                cv2.LINE_AA)
                continue
            if a.get("box"):
                bx, by, bw, bh = a["box"]
                cv2.rectangle(vp, self._to_disp(bx, by), self._to_disp(bx + bw, by + bh), col, 1, cv2.LINE_AA)
            draw_cross(vp, p, CORR_C)
            cv2.circle(vp, p, 14, col, 1, cv2.LINE_AA)
            self._label(vp, f"#{a['object']} correction{q}", (p[0] + 17, p[1] + 26), col, LABEL_SCALE)
        if self.drag_start and self.drag_now:
            cv2.rectangle(vp, self.drag_start, self.drag_now, object_colour(self.obj), 1, cv2.LINE_AA)
        self._label(vp, f"{self.view.name}   frame {self.frame_no}   ({self.pos + 1}/{len(self.source)})"
                        f"{'   zoom x%.1f' % self.zoom if self.zoom > 1 else ''}", (10, 22), TEXT, 0.5)

    def _timeline_strip(self, w: int) -> np.ndarray:
        """The parts of the timeline that only change with the view, the track or the threshold."""
        key = (self.vi, self.obj, self.low_sigma, w)
        if key in self._strips:
            return self._strips[key]
        n = len(self.source)
        strip = np.zeros((TIMELINE_H, w, 3), np.uint8)
        strip[:] = PANEL
        to_x = lambda pos: np.round(8 + (w - 16) * np.asarray(pos) / max(1, n - 1)).astype(int)  # noqa: E731
        strip[4:19, 8:w - 8] = (62, 62, 62)          # track lane
        strip[26:29, 8:w - 8] = (90, 90, 90)         # detection lane
        det_pos = [p for f, d in self.view.detections.items() if len(d) and (p := self._pos(f)) is not None]
        if det_pos:
            strip[24:31, np.unique(to_x(det_pos))] = DET_TICK

        def lane(tv: TrackView | None, mask_colours, y0, y1):
            if tv is None or not len(tv.frames):
                return
            pos = np.array([self._pos(f) if self._pos(f) is not None else -1 for f in tv.frames])
            for mask, colour in mask_colours(tv):
                m = mask & (pos >= 0)
                if m.any():
                    strip[y0:y1, np.unique(to_x(pos[m]))] = colour

        lane(self._tv(), lambda tv: ((~tv.married, TRACK_DIM), (tv.married, TRACK_C), (tv.low(self.low_sigma), LOW_C)),
             4, 19)
        lane(self._tv(source=self.corrected), lambda tv: ((np.ones(len(tv.frames), bool), CORR_C),), 20, 23)
        self._strips = {key: strip}   # one cached strip is enough
        return strip

    def _draw_timeline(self, canvas: np.ndarray) -> None:
        top, w, n = self.vp[1], self.vp[0], len(self.source)
        canvas[top:top + TIMELINE_H, :w] = self._timeline_strip(w)
        to_x = lambda pos: int(8 + (w - 16) * pos / max(1, n - 1))  # noqa: E731
        for a in self.fixed:
            if a["view"] == self.view.name and a["object"] == self.obj and (p := self._pos(a["frame"])) is not None:
                cv2.line(canvas, (to_x(p), top + 33), (to_x(p), top + 43), object_colour(a["object"]), 3)
        for a in self._view_ann():
            if (p := self._pos(a["frame"])) is not None:
                sel = a["object"] == self.obj
                cv2.line(canvas, (to_x(p), top + 32 if sel else 38), (to_x(p), top + 44), CORR_C if sel else DIM,
                         2 if sel else 1)
        x = to_x(self.pos)
        cv2.rectangle(canvas, (x - 2, top + 2), (x + 2, top + TIMELINE_H - 12), (255, 255, 255), -1)
        cv2.putText(canvas, str(self.view.frame_numbers[0]), (8, top + TIMELINE_H - 2), FONT, 0.36, DIM, 1)
        last = str(self.view.frame_numbers[-1])
        cv2.putText(canvas, last, (w - 8 - 8 * len(last), top + TIMELINE_H - 2), FONT, 0.36, DIM, 1)

    def _state_lines(self) -> list[tuple[str, tuple]]:
        """What the selected track (and its corrected version) does on this frame."""
        lines = []
        tv = self._tv()
        if tv is None:
            lines.append((f"#{self.obj} is not seen in this view", DIM))
        else:
            i = tv.at(self.frame_no)
            if i is None:
                lines.append((f"#{self.obj} does not run through frame {self.frame_no}", DIM))
            else:
                low = bool(tv.low(self.low_sigma)[i])
                sig = f"sigma {tv.sigma[i]:.2f}" if np.isfinite(tv.sigma[i]) else "sigma unknown"
                what = "married to a detection here" if tv.married[i] else "inferred (no detection here)"
                lines.append((f"#{self.obj}: {what}, {sig}{', LOW' if low else ''}", LOW_C if low else TRACK_C))
            runs = low_stretches(tv.low(self.low_sigma))
            lines.append((f"track frames {int(tv.frames[0])}-{int(tv.frames[-1])} here; {len(runs)} low stretch(es),"
                          f" {int(tv.low(self.low_sigma).sum())} fr" if len(tv.frames) else "track empty", DIM))
        ctv = self._tv(source=self.corrected)
        if ctv is not None:
            i = ctv.at(self.frame_no)
            lines.append(("earlier correction: " + ("not on this frame" if i is None else
                          ("married" if ctv.married[i] else "inferred")), CORR_C))
        return lines

    def _draw_panel(self, canvas: np.ndarray) -> None:
        x0, W, H = self.vp[0], PANEL_W, self.canvas_size[1]
        cv2.rectangle(canvas, (x0, 0), (x0 + W, H), PANEL, -1)
        self.buttons, self.rows = {}, []
        L, bw = x0 + 14, W - 28
        half = (bw - 10) // 2

        y = 28
        cv2.putText(canvas, (self.title or "Track corrections")[:60], (L, y), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        y += 30
        cv2.putText(canvas, f"View {self.vi + 1} of {len(self.views)}: {self.view.name}", (L, y), FONT, 0.65,
                    TEXT, 2, cv2.LINE_AA)
        y += 20
        cv2.putText(canvas, self.view.label[-48:], (L, y), FONT, 0.38, DIM, 1, cv2.LINE_AA)
        y += 18
        cv2.putText(canvas, f"frame offset {self._offset(self.view):+d} (the source result's)", (L, y), FONT, 0.38,
                    DIM, 1, cv2.LINE_AA)
        y += 16
        same = []
        for v in self.views:
            if v is not self.view:
                f = self._frame_at(v, self._instant())
                same.append(f"{v.name[:12]} {f if v.frame_numbers[0] <= f <= v.frame_numbers[-1] else 'none'}")
        cv2.putText(canvas, ("same instant: " + "   ".join(same))[:62], (L, y), FONT, 0.38, TEXT, 1, cv2.LINE_AA)
        for i in range(len(self.views)):   # progress dots
            c = ACCENT if i == self.vi else (TEXT if self.visited[i] else (90, 90, 90))
            cv2.circle(canvas, (x0 + W - 20 - 18 * (len(self.views) - 1 - i), 52), 6, c, -1, cv2.LINE_AA)

        y += 26
        cv2.putText(canvas, "Track  ([ ]  1-9)", (L, y + 4), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        y += 12
        x = L
        for obj in self._objects()[:9]:
            self._button_draw(canvas, f"obj:{obj}", x, y, 34, 30, str(obj), object_colour(obj), obj == self.obj)
            x += 40

        y += 44
        cv2.putText(canvas, "Frame  (a / d   A / D   arrows)", (L, y + 4), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        y += 12
        sw = 44
        for i, (name, text) in enumerate((("m10", "-10"), ("m1", "-1"))):
            self._button_draw(canvas, name, L + i * (sw + 6), y, sw, 30, text)
        gx, gw = L + 2 * (sw + 6), bw - 4 * (sw + 6) + 6
        txt = (self.frame_input + "_") if self.frame_input is not None else f"go to {self.frame_no}"
        self._button_draw(canvas, "goto", gx, y, gw, 30, txt, (60, 60, 90), self.frame_input is not None)
        for i, (name, text) in enumerate((("p1", "+1"), ("p10", "+10"))):
            self._button_draw(canvas, name, gx + gw + 6 + i * (sw + 6), y, sw, 30, text)

        y += 40
        self._button_draw(canvas, "low_prev", L, y, half, 28, "< low conf. (R)", (40, 40, 120))
        self._button_draw(canvas, "low_next", L + half + 10, y, half, 28, "low conf. > (r)", (40, 40, 120))
        y += 34
        self._button_draw(canvas, "fix_prev", L, y, half, 28, "< correction (C)", (40, 90, 40))
        self._button_draw(canvas, "fix_next", L + half + 10, y, half, 28, "correction > (c)", (40, 90, 40))
        y += 34
        self._button_draw(canvas, "sig_m", L, y, 44, 26, "-")
        self._button_draw(canvas, "sig_box", L + 50, y, bw - 100, 26, f"low confidence: sigma > {self.low_sigma:.3g}",
                          (60, 60, 90))
        self._button_draw(canvas, "sig_p", L + bw - 44, y, 44, 26, "+")
        y += 32
        third, quarter = (bw - 20) // 3, (bw - 18) // 4
        for i, (name, text, on, key) in enumerate((("dets", "dets", self.show_dets, "h"),
                                                   ("epipolar", "epipolar", self.show_epipolar, "l"),
                                                   ("trail", "trail", self.show_trail, "t"),
                                                   ("marks", "marks", self.show_marks, "m"))):
            self._button_draw(canvas, name, L + i * (quarter + 6), y, quarter, 26,
                              f"{text} {'on' if on else 'off'} ({key})", (70, 70, 70) if on else (45, 45, 45),
                              scale=0.38)
        y += 32
        self._button_draw(canvas, "suggest", L, y, half, 26, "suggest here (S)", (40, 90, 40))
        self._button_draw(canvas, "clear_track", L + half + 10, y, half, 26, f"delete #{self.obj} in view",
                          (50, 50, 110))

        y += 46
        for text, col in self._state_lines():
            cv2.putText(canvas, text[:62], (L, y), FONT, 0.4, col, 1, cv2.LINE_AA)
            y += 17

        # corrections table (this view)
        y += 12
        cv2.putText(canvas, "Corrections, this view  (click: go to | right: delete)", (L, y), FONT, 0.4, DIM, 1,
                    cv2.LINE_AA)
        rows = sorted(self._view_ann(), key=lambda a: (a["object"] != self.obj, a["object"], a["frame"]))
        footer_top = H - 196
        row_h = 22
        y_head, y = y, y + 8
        max_rows = self._table_rows(len(rows), max(1, (footer_top - 100 - y) // row_h))
        rw = self._draw_table_scroll(canvas, L, y_head, y, bw, len(rows), max_rows, row_h)
        if not rows:
            cv2.putText(canvas, "none yet - click on the fly in the image", (L + 6, y + 18), FONT, 0.42,
                        (120, 120, 120), 1, cv2.LINE_AA)
        for i, a in enumerate(rows[self.table_scroll:self.table_scroll + max_rows]):
            ry = y + i * row_h
            here = a["frame"] == self.frame_no
            cv2.rectangle(canvas, (L, ry), (L + rw, ry + row_h - 3),
                          (75, 75, 75) if a["object"] == self.obj else (55, 55, 55), -1)
            cv2.rectangle(canvas, (L, ry), (L + 8, ry + row_h - 3), object_colour(a["object"]), -1)
            text = (f"#{a['object']}  frame {a['frame']:<6}({a['x']:.0f}, {a['y']:.0f})"
                    f"{'  box' if a.get('box') else ''}{'  suggested' if a.get('suggested') else ''}")
            cv2.putText(canvas, text, (L + 16, ry + 14), FONT, 0.42, (255, 255, 255) if here else TEXT, 1, cv2.LINE_AA)
            self.rows.append(((L, ry, L + rw, ry + row_h - 3), a))

        # corrections per track and view
        y = footer_top - 96
        cv2.putText(canvas, "All views  (corrections per track)", (L, y), FONT, 0.42, DIM, 1, cv2.LINE_AA)
        col_w = (bw - 34) // max(1, len(self.views))
        for i, v in enumerate(self.views):
            cv2.putText(canvas, v.name[:12], (L + 34 + i * col_w, y + 18), FONT, 0.38,
                        TEXT if i == self.vi else DIM, 1, cv2.LINE_AA)
        for r, obj in enumerate(self._objects()):
            ry = y + 36 + r * 17
            if r >= 3:
                cv2.putText(canvas, "...", (L, ry), FONT, 0.4, DIM, 1)
                break
            cv2.circle(canvas, (L + 5, ry - 4), 5, object_colour(obj), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"#{obj}", (L + 12, ry), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
            for i, v in enumerate(self.views):
                n = sum(a["object"] == obj and a["view"] == v.name for a in self.ann)
                cv2.putText(canvas, str(n) if n else "-", (L + 34 + i * col_w, ry), FONT, 0.4, TEXT, 1, cv2.LINE_AA)

        # footer: navigation, help, status
        y = footer_top
        last = self.vi == len(self.views) - 1
        self._button_draw(canvas, "back", L, y, third, 36, "< back", (70, 70, 70))
        self._button_draw(canvas, "cancel", L + third + 10, y, third, 36, "cancel (esc)", (50, 50, 110))
        self._button_draw(canvas, "continue", L + 2 * (third + 10), y, third, 36,
                          "finish (enter)" if last else "continue >", (40, 110, 40))
        help_lines = [f"circles: {self.detections_label}"[:60],
                      "yellow: track (top dot = married)  red: low confidence",
                      "left click: correction here (snaps)   drag: box",
                      "right click: delete   wheel: zoom   middle drag: pan",
                      "m: hide / show corrections and START / END",
                      "timeline: white dets, yellow track, red low, green fixes"]
        for i, line in enumerate(help_lines):
            cv2.putText(canvas, line, (L, y + 58 + i * 17), FONT, 0.38, DIM, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (L, H - 40), (L + bw, H - 12), (30, 30, 30), -1)
        cv2.putText(canvas, self.status[:58], (L + 8, H - 21), FONT, 0.42, (180, 220, 255), 1, cv2.LINE_AA)


def run_correction_gui(views: list[ViewInput], tracks: dict[int, dict[str, TrackView]],
                       corrections: list[dict] | None = None, fixed: list[dict] | None = None,
                       corrected: dict[int, dict[str, TrackView]] | None = None, title: str = "",
                       detections_label: str = "detections", suggest_gate_px: float = 15.0,
                       low_sigma: float = 10.0) -> tuple[list[dict] | None, float]:
    """Place corrections on every view in turn. Returns (corrections, or None if cancelled; the
    low-confidence threshold the window ended with).
    tracks: the source result's tracks, track id -> view name -> TrackView; corrected: an earlier
    corrected result in the same layout (drawn green); fixed: the source's start / end annotations."""
    gui = CorrectionGUI(views, tracks, corrections, fixed, corrected, title, detections_label, suggest_gate_px,
                        low_sigma)
    return gui.run(), gui.low_sigma
