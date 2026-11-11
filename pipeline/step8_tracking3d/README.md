# Step 8: 3D tracking

Part of the [pipeline](../README.md), which explains how to run the scripts. The `--source gt` flag and the fruit / leaves / arena overlays are described [there](../README.md#shared-by-the-3d-scripts-steps-7-to-10).

Both scripts work on the run folder that step 5 or 6 wrote, and their settings are in `configs/tracking.json`.

## `main_tracks3d.py`

Builds 3D tracks automatically.

How it works:

1. Detections at fixed pixel positions (background clutter) are removed.
2. Triangulated points are linked into tracklets.
3. Each tracklet seeds a factor-graph track.
4. Every 2D detection is associated with tracks iteratively, including detections seen by one camera only.

Results go to `<run>/3d/<model>/factor_graph/tracks_<datetime>/`:

| File | Holds |
| --- | --- |
| `tracks_3d.csv` | the tracks |
| `params.json` | the settings |
| plots | 3D and reprojection plots |
| `evaluation_3d.txt`, `evaluation_2d.txt` | the evaluation, when the run has labels |

## `main_tracks3d_guided.py`

3D tracks for hand-picked flies. Each annotated fly gets exactly one track: the clicks fix its frame range and act as trusted constraints.

### Annotating

- An annotator opens each camera's video, where you click where each fly starts and / or ends.
- Later views can suggest the points from earlier views using the epipolar geometry.
- Annotations are saved and can be reused with `--no-gui`.

### Keeping the views in sync

The annotator keeps the views in sync using each camera's `frame_offset`:

- Switching views opens the same instant.
- `g` jumps to the selected fly's point in another view.
- If the same START / END is annotated in two views, the panel shows the offset that would line them up.

To find an unknown offset, annotate one sharp event, like a take-off, in every view.

You can change the offset of the view you are on without leaving the annotator. The panel, epipolar lines and suggestions update straight away.

- `,` / `.` changes it by -1 / +1, and `<` / `>` by -10 / +10. The buttons in the panel do the same.
- `o`, or clicking the offset box, lets you type a value. Press Enter to apply it.
- `O` applies the offset the panel suggests from a START / END annotated in two views.
- `W` saves the current offsets to `cameras.json` as the defaults for every run.

When you finish, any offsets that differ from `cameras.json` are kept for this run only, in `<run>/3d/frame_offsets.json`. Every step (`main_points3d.py`, `main_tracks3d.py`, the guided tracker) uses that file instead of `cameras.json` for this run. To go back to `cameras.json`, delete the file. Suggestions made before an offset change are not redone, so press `S` / `E` again to refresh them.

### Output

Outputs and evaluation are the same as for `main_tracks3d.py`, in `factor_graph/guided_<datetime>/`.
