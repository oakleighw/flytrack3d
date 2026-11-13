# Step 5: evaluation

Part of the [pipeline](../README.md), which explains how to run the scripts.

## `main_evaluate2d.py`

Runs the enabled detectors on the detection source (`configs/shared/detection_source.json`) and scores them against the ground-truth labels. The models are the `models` of `configs/detection.json`.

- **Metrics:** AP@0.2 / 0.5 / 0.5:0.95, P/R/F1, and a side-by-side model comparison. AP is reported as "AP" when the ground truth has one class and as "mAP" (the mean over classes) when it has more.
- **Confidence:** the models infer at a low `confidence_threshold` (0.001) and save every box with its confidence (`class xc yc w h conf`), so AP covers the whole precision-recall curve. The operating point is applied afterwards: P/R/F1 at `evaluation.pr_conf_threshold` (0.25), the boxes drawn at `visualisation.conf_threshold` (null = the same 0.25), and steps 7 to 9 track with `tracking3d.conf_threshold` (0.25).
- **Charts:** `summary.png` next to each `summary.txt`: AP per IoU and by object size, P/R/F1, and (batch) AP per input.
- **MMDetection models:** `"type": "mmdet_sahi"` evaluates a DINO / RT-DETR / ... checkpoint with the same SAHI slicing and merging as the YOLO models, and the same reports. The ground truth stays in YOLO format; no COCO conversion is needed.
- **Optional:** track plots and annotated videos.
- **Cache:** predictions are cached and reused while the model and frames are unchanged.
- **Batch source:** with a batch source (`shared/detection_source.batch.example.json`), each camera video becomes one input of the run, and that run is what the 3D scripts (steps 7 to 9) work on.
