# Pipeline

The scripts to run are the `main_*.py` files in this folder, grouped into numbered steps in the order you would normally use them. This page summarises each step; the README in each step folder describes its scripts in full.

## Running a script

Each script can be run in two ways:

- **From the IDE:** edit its `SETTINGS` block and press Run.
- **From the command line:** e.g. `python pipeline/step5_evaluation/main_evaluate2d.py`. Flags override the `SETTINGS` block, and `--help` lists them.

The scripts work from any working directory. Each reads its stage's config from `configs/` (see [configs/README.md](../configs/README.md)); `--config` gives another. The evaluation and 3D scripts each write to a timestamped `output/run_<datetime>/` folder, or into the run folder they work on.

## Steps

### Step 0: video editing

| Script | What it does |
| --- | --- |
| `main_slice_videos.py` | cuts the same time window out of every video in a folder |

Config: `configs/video.json`. Details: [step0_video_editing](step0_video_editing/README.md).

### Step 1: image dataset creation

| Script | What it does |
| --- | --- |
| `main_reconstruct_cvat_dataset.py` | rebuilds a YOLO dataset from a CVAT export and its video |
| `main_merge_datasets.py` | merges a folder of YOLO datasets into one, without duplicate images |
| `main_convert_dataset_format.py` | converts a dataset between the YOLO and COCO formats |
| `main_analyse_dataset.py` | reports what a YOLO dataset holds, with figures |
| `main_check_bboxes.py` | shows random images with their boxes drawn, to check the labels |

Config: `configs/image_dataset.json`, one section per script. Details: [step1_image_dataset_creation](step1_image_dataset_creation/README.md).

### Step 2: image tiling

| Script | What it does |
| --- | --- |
| `main_tile_dataset.py` | tiles a full-frame YOLO dataset into small square crops for SAHI-style training |

Config: `configs/image_tiling.json`. Details: [step2_image_tiling](step2_image_tiling/README.md).

### Step 3: dataset pruning

| Script | What it does |
| --- | --- |
| `main_analyse_redundancy.py` | finds the near-identical frames of a video; changes nothing |
| `main_guided_prune.py` | walks a split one video at a time and builds a pruned copy of the dataset |

Config: `configs/dataset_pruning.json`, one section per script. Details: [step3_dataset_pruning](step3_dataset_pruning/README.md).

### Step 4: YOLO training

| Script | What it does |
| --- | --- |
| `main_train_detector.py` | trains a YOLO detector with Ultralytics, optionally with a P2 head |

Config: `configs/training.json`. Details: [step4_training](step4_training/README.md).

### Step 5: evaluation

| Script | What it does |
| --- | --- |
| `main_evaluate2d.py` | runs the enabled detectors on the detection source and scores them against the ground truth |

Config: `configs/detection.json`. Details: [step5_evaluation](step5_evaluation/README.md).

### Step 6: inference

| Script | What it does |
| --- | --- |
| `main_inference2d.py` | runs the detectors on new recordings, without ground truth |

Config: `configs/detection.json`. Details: [step6_inference](step6_inference/README.md).

### Step 7: 3D points

| Script | What it does |
| --- | --- |
| `main_points3d.py` | matches each frame's detections across the cameras and triangulates them, with no tracking |

Config: `configs/tracking.json`. Details: [step7_points3d](step7_points3d/README.md).

### Step 8: 3D tracking

| Script | What it does |
| --- | --- |
| `main_tracks3d.py` | builds 3D tracks automatically with the factor graph |
| `main_tracks3d_guided.py` | builds one 3D track per hand-annotated fly |

Config: `configs/tracking.json`. Details: [step8_tracking3d](step8_tracking3d/README.md).

### Step 9: visualise 3D

| Script | What it does |
| --- | --- |
| `main_visualise_past_3d.py` | reopens the interactive 3D plot of a past step 7 / 8 result |
| `main_visualise_external_points.py` | the same 3D plot for files that come without their run |
| `main_visualise_all.py` | draws every run's tracks in a folder together, aligned on the fruit, and compares folders of runs |
| `main_visualise_past_all.py` | reopens the interactive plot of a past `main_visualise_all.py` output |

Config: `configs/tracking.json`, `configs/external_points.json` and `configs/visualise_all.json`. Details: [step9_visualise3d](step9_visualise3d/README.md).

## Shared by the 3D scripts (steps 7 to 9)

### Running on the ground truth

Use `--source gt` on the 3D scripts to run them on the ground-truth labels instead of the predictions.

### Fruit, leaves and arena frame

The 3D scripts also draw the red fake fruit, the green leaves around it and the arena frame, with no annotation needed. The fruit and leaves come from each view's median frame (frames sampled across the video, so the flies vanish), segmented in HSV:

- **Fruit:** round, saturated red blobs are candidates. The combination of one blob per view that triangulates consistently is the fruit, which gives the centre of its base on the wall. Its radius is measured from the blobs. It is drawn as a shaded hemisphere bulging out from the wall, towards the cameras.
- **Leaves:** the green region around the fruit in each view is laid flat onto the fruit's wall (camera rays meet the wall plane), and kept where at least two views agree. The terminal prints how well each view agrees (IoU), a quick check of the calibration.
- **Arena frame:** the tapered box of pipes is grown out from the fruit using its measured size (`arena_plotting.arena`): the back face on the fruit's wall, upright, at a fixed offset from the fruit, and the front face `depth` towards the cameras. It is only as accurate as the fruit's placement on the back wall, so check the dashed frame in the reprojection images.

Caching and settings:

- The fruit and leaves are fitted once per run and cached in `<run>/3d/fruit/` and `<run>/3d/leaves/`, each with one debug image per camera (`hsv/<camera>.jpg`). The frame needs no fitting.
- The settings are in `configs/shared/arena_plotting.json`.
- Use `--no-arena` to skip all three.
