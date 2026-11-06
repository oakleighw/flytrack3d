# Configs

Each pipeline stage has one config in `configs/`, and the settings that several stages need are in `configs/shared/`, where they are set once.

## Basics

- The real configs are gitignored; only the `*.example.json` templates are tracked. `python setup_configs.py` creates the real ones from the templates (see the [main README](../README.md#setup)).
- Every config section has a `_readme` entry that explains its settings.
- Relative paths in a config are resolved against the folder that config is in.
- Results go to `output/` in the project root unless a config sets `output_dir`.

## Config files

Stage configs:

| Config | Used by | Holds |
| --- | --- | --- |
| `video.json` | step 0 | the videos folder and the slice window |
| `image_dataset.json` | step 1 | one section per script |
| `image_tiling.json` | step 2 | the tiling settings |
| `dataset_pruning.json` | step 3 | one section per script |
| `training.json` | step 4 | the `training` section |
| `detection.json` | steps 5 and 6 | the `models`; includes the detection source, the evaluation settings and `training.json` |
| `tracking.json` | steps 7 to 9 | only includes: cameras, tracking3d, arena plotting, evaluation |
| `external_points.json` | step 9 (external points) | the `inputs`; includes cameras, tracking3d, arena plotting |
| `visualise_all.json` | step 9 (all runs) | the groups of runs to compare; includes arena plotting |

Shared configs:

| Config | Holds |
| --- | --- |
| `shared/paths.json` | the folders several configs point at, by name |
| `shared/detection_source.json` | what steps 5 and 6 run the detectors on: a YOLO split, a video, or a folder of camera videos (`detection_source.batch.example.json`) |
| `shared/evaluation.json` | the `evaluation` and `visualisation` settings |
| `shared/cameras.json` | the camera calibration |
| `shared/tracking3d.json` | the 3D matching and factor graph settings |
| `shared/arena_plotting.json` | the fruit, leaves and arena frame |

## Including other configs

A config pulls in other files with `"_include"`, a list of paths relative to its own folder:

```json
{
  "_include": ["shared/detection_source.json", "shared/evaluation.json", "training.json"],
  "models": [...]
}
```

- The included files are merged in order and the config's own keys win, section by section. A stage can therefore override one shared setting without copying the rest, e.g. `"tracking3d": {"n_tracks": 3}` in `tracking.json`.
- To keep a second set-up, such as another detection source, copy the stage config under a new name, point its `_include` at the other shared file and pass it with `--config`.
- A run folder's `config.json` is the merged result, so it records everything the run used.

## Named paths

Because the same folders come up in several configs, `shared/paths.json` gives them names:

```json
{
  "paths": {
    "full_dataset": "/data/full_dataset",
    "pruned_dataset": "/data/pruned_dataset",
    "session_videos": "/data/session_01/videos"
  }
}
```

A config that includes it can write `${name}` in any setting, e.g. `"data_yaml": "${pruned_dataset}/data.yaml"`, so a new pruned dataset is one edit. The names are free to choose, and an entry may use another one.

## Two meanings of "dataset"

"Dataset" means two things in this project, kept in different places:

- **A YOLO dataset on disk** (`images/<split>`, `labels/<split>`, `data.yaml`) is what steps 1 to 4 build, prune and train on. Each of their configs names it in its own setting (`input_dir`, `training.data_yaml`).
- **The detection source** is what steps 5 and 6 run the detectors on, which may be a split of such a dataset or the camera videos of one session.

Steps 7 to 9 read neither: they work on the run folder that step 5 or 6 wrote.
