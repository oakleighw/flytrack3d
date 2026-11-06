# Step 9: visualise 3D

Part of the [pipeline](../README.md), which explains how to run the scripts. The `--source gt` flag and the fruit / leaves / arena overlays are described [there](../README.md#shared-by-the-3d-scripts-steps-7-to-9).

## `main_visualise_past_3d.py`

Opens the interactive 3D plot of a past step 7 / 8 result again, without recomputing it. Nothing is written to the result's folder.

- **Choosing a result:** in the terminal, choose a run, then what to view (its triangulated points, a past unguided run or a past guided run), then which one if there are several.
- **Read from the result:** the points / tracks come from the saved csv.
- **Rebuilt:** the cameras, ground-truth overlay, arena and the grey points behind the tracks are rebuilt using the settings and frame offsets saved in the result's `params.json`.
- **Calibration:** it is not saved with a result, so those parts follow the current `cameras` config.

## `main_visualise_all.py`

Draws the 3D tracks of every run in a folder in one plot, and compares folders of runs (groups) in different colours, e.g. `cladding` in pink against `control` in blue.

- **Config:** `configs/visualise_all.json` (template: `configs/visualise_all.example.json`). Its `groups` list each group's name, folder and colour. `--group NAME FOLDER [COLOUR]` (repeatable) replaces them from the command line.
- **Runs:** every folder holding a `3d/` folder, up to 3 levels below a group's folder. Each run's latest guided result is drawn, or its latest unguided one if it has none (`kinds`).
- **Tracks:** the line is the factor-graph track. The dots mark the frames where it was fitted to YOLO detections; where the line runs without dots, the motion prior bridged a gap.
- **Alignment:** runs are recorded with the rig in different places and with different reference cameras, so their raw coordinates cannot be compared. Averaging them would not fix this, since they differ by a rotation as well as a shift. Each run is instead moved by a rigid transform into a frame centred on its fruit. The origin is the fruit's base on the wall; x points right, y towards the wall and z up, in mm. The transform is built from the run's cached fruit fit (`3d/fruit/fruit.json`) and the calibration it was fitted with, the same wall frame the leaves and arena frame are built on. No video is needed. A run without a fruit fit is skipped, and the terminal says why.
- **Arena:** one fruit (median size), every run's leaves overlaid (darker where they agree) and the arena frame.
- **Saved** to `output/visualise_all/<groups>_<datetime>/`:
  - `aligned_tracks.png`: the plot.
  - `side_by_side.png`: with several groups, one column per group (its tracks alone) with the same views on the same axes, to compare them. `"side_by_side": false` in the config turns it off.
  - `aligned_tracks.csv`: every track point in the shared frame.
  - `plot.json`: the colours, title, track style and aligned arena, so `main_visualise_past_all.py` can redraw the plot.
  - `alignment.json`: each run's result, its transform and its fruit size, plus the skipped runs.

  `--no-show` saves without opening the window.

## `main_visualise_past_all.py`

Opens the interactive 3D plot of a past `main_visualise_all.py` output again, without finding or aligning the runs again.

- **Choosing:** in the terminal, choose an output from `output/visualise_all/` (`--output <folder>` or `--output latest` skips the list). If it has several groups, choose all of them together, all side by side (`--side-by-side`) or one (`--group <name>`).
- **Read from the output:** everything comes from its `aligned_tracks.csv` and `plot.json`. The run folders don't need to exist any more, and nothing is written.
- **Older outputs:** outputs saved before `plot.json` existed cannot be shown. Run `main_visualise_all.py` again for those.

## `main_visualise_external_points.py`

The same 3D plot for files that come without their run.

- **Config:** it has its own, `configs/external_points.json` (template: `configs/external_points.example.json`), which holds the inputs and includes the shared calibration (`cameras`) and `tracking3d` / `arena_plotting` settings.
- **Inputs:** set in the config's `inputs` section or with the flags below.

### Inputs

| Flag | Give it |
| --- | --- |
| `--csv` | a `points.csv`, a `tracks_3d.csv`, or any csv with `frame, x, y, z` columns; a `track` column makes it a track plot |
| `--labels` | YOLO labels of the detector's predictions: a folder with one sub-folder per camera, named after it |
| `--fg-labels` | YOLO labels of the boxes the factor graph inserted, in the same layout |
| `--videos` | a folder with one video per camera, needed for the arena overlays |

Give it a csv, or labels (`--labels` and / or `--fg-labels`).

- **Labels:** they are pooled, shifted by each camera's `frame_offset`, and matched and triangulated as in [step 7](../step7_points3d/README.md). `--link` also links the points into tracks (no factor graph).
- **Arena overlays:** the fruit, leaves and arena frame are drawn only with `--videos`, since they are found in the images. Their fits are cached in `output/external/<videos folder>/3d/`.
- **Saving:** `--save plot.png` also saves the figure.
