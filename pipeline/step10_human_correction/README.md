# Step 10: human correction

Part of the [pipeline](../README.md), which explains how to run the scripts. The `--source gt` results and the fruit / leaves / arena overlays are described [there](../README.md#shared-by-the-3d-scripts-steps-7-to-9).

## `main_human_correction.py`

Fixes the few tracks that a guided (or unguided) step 8 result gets wrong. You don't tune parameters: you click where the fly really is. Those clicks are correction points, and the factor graph treats them like the guided START / END annotations: trusted anchors it must pass through.

Settings are in `configs/tracking.json`, as for step 8.

### Choosing what to correct

In the terminal, choose a run, then one of its guided or unguided results (`--kind`, `--result latest` skip the lists). That result's tracking settings and frame offsets are used throughout.

### The correction window

It opens on each camera's video in turn and works like the step 8 annotator (zoom, pan, seeking, epipolar lines, `Enter` / `Backspace` between views).

What it shows:

| On the image | Means |
| --- | --- |
| white circles | YOLO detections (clicks snap to them) |
| yellow ring | the selected track on this frame: a filled dot = married (fitted to a detection in this view), no dot = inferred by the motion prior |
| red ring | the same, where the track's confidence is low |
| yellow line | the selected track over the 40 frames either side (`t`) |
| small rings | the other tracks, in their colours |
| green ring / line | the track as corrected the last time this step ran on this result |
| green cross | a correction; small green circles mark the track's corrections on other frames |
| + / x | the result's START / END annotations from step 8 (fixed here) |

| On the timeline | Means |
| --- | --- |
| white | frames with detections in this view |
| yellow | frames of the selected track: bright = married here, dim = inferred |
| red | low confidence: the track's 3D uncertainty (sigma) is above the threshold |
| thin green | the earlier corrected track; green ticks = corrections |

The red stretches are where to look. Click on the fly at a frame or two inside a stretch, ideally in two views at the same instant, to put the track back on it.

| Key | Does |
| --- | --- |
| left click / drag | correction for the selected track on this frame (a box uses its centre) |
| right click | delete the nearest correction on this frame (or a row in the table) |
| `1`-`9`, `[` `]` | select a track |
| `a` `d`, `A` `D`, arrows | previous / next frame, -10 / +10 |
| `r` / `R` | start of the next / previous red stretch |
| `c` / `C` | next / previous correction of the track in this view |
| `-` / `+` | lower / raise the low-confidence threshold |
| `S` | suggest the correction in this view from the track's corrections at this instant in the other views |
| `h` / `l` / `t` | detections, epipolar lines, trail on / off |
| `Enter`, `Backspace`, `Esc` | next view (finish on the last), previous view, cancel |

The frame offsets can't be changed here: the tracks shown were built with the result's offsets. Change them in step 8.

### Saved corrections

When you finish, the corrections are saved to `<run>/3d/corrections.json`, under the result they correct.

Running the step again on the same result loads them, so you can add to them or delete them. If that result has none yet but another result of the run does, the terminal offers to copy them. Guided results number their tracks by annotated object, so corrections carry over between them.

`--no-gui` solves the saved corrections without opening the window.

### Solving

After the window closes, every track with corrections is solved again, with a progress bar (`tracking3d/correction.py`):

- **Anchors:** the corrections, plus the result's START / END annotations.
- **Starting guess:** the old track, except within `correction_seed_window` frames of a correction, where the corrections are interpolated instead.
- **Frame range:** a guided track keeps its START-END range, so corrections outside it are left out, with a warning. An unguided track grows or shrinks as in step 8, always covering its corrections.
- **Detections:** a corrected track takes its detections first. Tracks without corrections are kept exactly as they were.

### Output

Results go to `<run>/3d/<model>/factor_graph/corrected_<datetime>/`:

| File | Holds |
| --- | --- |
| `tracks_3d.csv`, `tracks.json`, `summary.txt`, `params.json` | every track (corrected or kept), as in step 8; `params.json` and `tracks.json` also hold the source result and the corrections |
| `source_tracks_3d.csv` | the tracks before correction |
| `corrections.json` | the corrections used |
| `corrected_observations.csv` | what each corrected track is fitted to: its YOLO detections and the corrections (raw pixels) |
| `tracks_3d.png` | the 3D plot: the source tracks in their colours, the corrected tracks in dark green, the corrections as green diamonds |
| `reprojection/<camera>.png` | the same over each camera's first frame |
| `reprojection/<camera>.mp4` | the back-projection video: a corrected track's dot is green where the correction moved it (over 3 px from the source track in that view, `MOVED_PX` in `tracking3d/evaluate.py`) and red, like the other tracks, where it did not; the correction clicks are green rings; `--no-video` skips it |
| `track_plots_backproj/<input>.png` | the 2D track plot of each view, plus the corrected tracks before correction (grey dots), where the correction moved them (dark green dots) and the correction clicks (dark green rings) |
| `evaluation_*.txt` | the evaluation, when the run has labels |

The ground truth is drawn as a wide, pale green band, and a corrected track as a thin, solid dark green line.

### Settings

`tracking3d` in the config:

| Setting | Default | Does |
| --- | --- | --- |
| `correction_low_sigma` | 10 | a track frame with a 3D sigma above this (world units) is low confidence (red); `--low-sigma` overrides it |
| `correction_seed_window` | 30 | frames either side of a correction where the old track is not used as the starting guess |
| `anchor_pixel_sigma`, `guided_iterations` | as step 8 | how strongly the corrections pull, and the solve rounds |

### Viewing it again

[Step 9](../step9_visualise3d/README.md) lists corrected results as their own kind:

- In `main_visualise_past_3d.py`, choose "corrected tracks" to see a result with its correction overlaid in green. Choose the guided / unguided result to see it as it was.
- `main_visualise_all.py` draws a run's corrected result first if it has one (`"kinds"` in its config sets the order).
