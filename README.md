# flytrack3d

Deep-learning fly detection and 3D trajectory tracking from multi-camera video, built for noisy real-world scenes. Flies are detected with trained YOLO models rather than classical image processing, and a factor graph turns the detections into smooth 3D trajectories that hold up when cameras miss the fly. Starts from recorded videos and an existing camera calibration; capture and calibration are not part of this repo.

- **Datasets:** cut videos, then build, merge, convert, check, tile and prune YOLO datasets.
- **Detection:** train YOLO detectors (optionally with SAHI slicing), run them on videos or image folders, cache the predictions, and score them against ground-truth labels: COCO-style AP (mAP with several classes), precision / recall / F1, track plots and annotated videos.
- **3D tracking:** match detections across cameras, triangulate them, and refine them into smooth 3D tracks with a factor graph (gtsam). The factor graph combines reprojection factors with a constant-velocity motion prior, so it bridges frames where cameras miss the fly. Tracks can also be guided by hand-annotated start / end points.
- **Evaluation and visualisation:** score tracks against the ground truth in 3D (RMSE in mm) and in 2D, by projecting them back into each camera, and reopen or plot past 3D results.

## Setup

```bash
pip install -r requirements.txt
python setup_configs.py
```

This needs Python 3.12 or newer. For GPU torch, or on Windows (where gtsam comes from conda-forge), first follow the notes at the top of [requirements.txt](requirements.txt).

`setup_configs.py` copies every template in `configs/` to the config it stands for (`configs/training.example.json` to `configs/training.json`, and so on), never touching a config that already exists, and lists the ones to fill in. Then:

1. Fill in your paths, models and camera calibration. Start with `configs/shared/paths.json`, which most of the others refer to.
2. Delete the `"_example"` line of each config you filled in. The scripts refuse to run a config that still has it.

How the configs fit together is described in [configs/README.md](configs/README.md).

## Pipeline

The scripts to run are the `main_*.py` files in `pipeline/`, grouped into numbered steps in the order you would normally use them. [pipeline/README.md](pipeline/README.md) explains how to run them and summarises each step; each step folder has its own README with the full description of its scripts.

| Step | Folder | What it does | Config |
| --- | --- | --- | --- |
| 0 | [step0_video_editing](pipeline/step0_video_editing/README.md) | cut a time window out of a folder of videos | `video.json` |
| 1 | [step1_image_dataset_creation](pipeline/step1_image_dataset_creation/README.md) | build, merge, convert and check YOLO datasets | `image_dataset.json` |
| 2 | [step2_image_tiling](pipeline/step2_image_tiling/README.md) | tile a full-frame dataset into training crops | `image_tiling.json` |
| 3 | [step3_dataset_pruning](pipeline/step3_dataset_pruning/README.md) | find near-identical frames and build a pruned dataset | `dataset_pruning.json` |
| 4 | [step4_training](pipeline/step4_training/README.md) | train a YOLO detector | `training.json` |
| 5 | [step5_evaluation](pipeline/step5_evaluation/README.md) | run detectors and score them against ground truth | `detection.json` |
| 6 | [step6_inference](pipeline/step6_inference/README.md) | run detectors on new recordings, without ground truth | `detection.json` |
| 7 | [step7_points3d](pipeline/step7_points3d/README.md) | match detections across cameras and triangulate them | `tracking.json` |
| 8 | [step8_tracking3d](pipeline/step8_tracking3d/README.md) | build 3D tracks with the factor graph, automatic or guided | `tracking.json` |
| 9 | [step9_visualise3d](pipeline/step9_visualise3d/README.md) | reopen past 3D results, plot points from outside a run, or compare all runs aligned on the fruit | `tracking.json`, `external_points.json`, `visualise_all.json` |

## Layout

| Folder | Holds |
| --- | --- |
| `pipeline/` | the `main_*.py` scripts, one folder per step. They only parse settings and call into the packages below. |
| `configs/` | one config per pipeline stage, and the shared settings in `configs/shared/` |
| `common/` | what every package below builds on: config loading, frames and datasets, YOLO dataset reading, track linking |
| `dataset_tools/` | dataset creation, cleaning, conversion and pruning |
| `detection/` | the 2D detector: training and evaluation |
| `tracking3d/` | matching, triangulation, the factor graph, 3D evaluation and the guided-track annotator |
| `scene3d/` | the static arena objects in the 3D plots: fruit, leaves and arena frame |
| `visualise/` | everything drawn or laid out for people to look at: plots, boxes, charts and metric tables |

<details>
<summary>Per-file breakdown</summary>

**`common/`** imports none of the other packages.

- `config.py`: config loading
- `datasets.py`: frames and datasets
- `yolo_format.py`: reading YOLO datasets from disk
- `linking.py`: the frame-to-frame track linker

**`dataset_tools/`**

- `tile.py`: tiles a full-frame dataset into training crops
- `cvat.py`: rebuilds a dataset from a CVAT export and its video
- `merge.py`: merges a folder of datasets into one without duplicate images
- `stats.py`: computes the per-split statistics of a dataset (videos, images, instances, box sizes)
- `convert_yolo_to_coco.py`, `convert_coco_to_yolo.py`: the YOLO -> COCO and COCO -> YOLO converters
- `slice_videos.py`: cuts a time window out of a folder of videos
- `prune/`: the dataset pruning (step 3)
  - `redundancy.py`: ranks a video's bbox sizes and decides with SSIM which boxes are kept or redundant
  - `plan.py`: works out which whole frames a run drops
  - `runs.py`: keeps the manifests and the purge log
  - `build.py`: builds the pruned dataset
  - `guided.py`: the resumable per-video workflow
  - `review.py`: chooses the frames to look at
  - `dataset.py`, `metrics.py`, `prompts.py`, `paths.py`: the label-file, counting, terminal-prompt and output-folder helpers they share

**`detection/`**

- `training.py`: builds the YOLO model (version, size, P2 head, pretrained weights) and runs Ultralytics training
- the rest is its evaluation: detectors, metrics, saved predictions and their cache (`predictions.py`) and the evaluation pipeline

**`tracking3d/`**

- `camera.py`: the camera model
- cross-view matching and triangulation, run / camera loading, the factor graph (unguided and guided), evaluation, output files and loading past results
- `external.py`: results from outside a run
- `annotator/`: the start / end annotator of the guided tracks and its epipolar suggestions

**`scene3d/`** (its settings are the config's `arena_plotting` section)

- `fruit/`, `leaves/`: each has `hsv/` (segmentation), `mesh/` (reconstruction and drawing) and `locate.py` (per-run glue and cache)
- `arena/`: `mesh/` and `locate.py` only, since nothing is segmented
- `objects.py`: bundles them for the plots
- `geometry.py`, `drawing.py`, `frames.py`, `settings.py`: the code they share

**`visualise/`** The other packages compute the results and call into it.

- `plots2d/`: the frame-sized 2D graphs
  - `track_plots.py`: ground-truth and predicted 2D tracks
  - `reprojection.py`: 3D tracks projected back into each camera
- `plots3d/scene.py`: draws the 3D points and tracks with their overlays
- `boxes/`: draws bounding boxes
  - `yolo_dataset.py`: boxes on an image and random samples of a dataset, used by the tiling and dataset analysis steps to check the labels, and boxes outlined on matplotlib axes
  - `pruning.py`: the dataset pruning checks, sample frames of a bbox size rank and dropped frames next to the kept frame they matched
  - `video.py`: annotated videos of ground truth, detections and back-projected boxes
- `charts/dataset.py`: the dataset analysis: pie charts of a dataset's splits and videos, and its box position and size plots
- `metrics/`: lays the metrics out as text tables
  - `detection.py`: the 2D model reports, comparison and batch table
  - `tracking.py`: the 3D track summary and the 3D / 2D track evaluation
  - `dataset.py`: the dataset analysis report
- `common.py`: the colours and helpers they share

</details>
