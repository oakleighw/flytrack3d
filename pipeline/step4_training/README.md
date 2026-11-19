# Step 4: training

Part of the [pipeline](../README.md), which explains how to run the scripts.

## `main_train_detector.py`

Trains a YOLO detector with Ultralytics from the `training` section of `configs/training.json`.

- **Settings:** the dataset yaml, GPU, W&B project / group, YOLO version and size, the optional P2 head (an extra small-object detection layer, from `p2_model_yaml`) and the training arguments.
- **Pretrained weights:** with `load_pretrained`, the standard model's pretrained weights are loaded, also into the P2 architecture (backbone and P3-P5 heads; the P2 layers start random).
- **Output:** runs go to `output/training/<wandb_project>/<run name>/`, named like the W&B run, e.g. `yolo_s_12_base` (`_2`, `_3`, ... if the name is taken).

### Evaluating a trained model

Set a model's `model_path` in the `models` of `configs/detection.json` to the run name (e.g. `"model_path": "yolo_s_12_base"`) instead of a path. It resolves to that run's `weights/best.pt`. Then run [step 5](../step5_evaluation/README.md).

## `main_train_mmdet.py`

Trains an MMDetection detector (DINO, RT-DETR, ...) from the `mmdet_training` section of `configs/mmdet_training.json`, picked with `--model` from its `models`.

- **Recipe:** each model starts from its upstream mmdet config (`recipe`). Only the dataset, batch, gradient accumulation, LR, epochs, schedule, seed, checkpoints and W&B are changed; the model, pipeline, augmentation and optimizer stay as the recipe has them.
- **Batch and LR:** the LR is scaled linearly from the recipe's total batch (`auto_scale_lr.base_batch_size`, the batch over all its GPUs, e.g. 16 for an `8xb2` recipe) to `batch_size` x `accumulative_counts`. On one GPU, accumulating up to the recipe's total keeps the recipe's LR (DINO: 2 x 8 = 16 at 1e-4).
- **Schedule:** `"recipe"` keeps the recipe's LR schedulers and stage switches, with epoch milestones and switch epochs stretched to `epochs` (DINO's milestone 11 of 12 becomes 46 of 50, RT-DETR's stage-2 switch 117 of 120 becomes 49 of 50); iteration-based warmup is unchanged. `"warmup_cosine"` replaces them with a 1000-iteration warmup and a cosine decay.
- **Checkpoints:** every epoch, keeping the latest `max_keep_ckpts` plus `best_coco_fitness_epoch_N.pth`, the best epoch by the Ultralytics fitness (0.1 mAP50 + 0.9 mAP50-95), which is logged as `coco/fitness`.
- **Environment:** mmdet can't be installed in this repo's env, so training runs in `detection/mmdet_train_worker.py` on the openmmlab env's Python (`python`). `gpu` is the GPU index as nvidia-smi numbers it; the worker sees only that GPU.
- **Output:** runs go to `output/training/<wandb_project>/<run name>/` (`_2`, `_3`, ... if the name is taken), with the merged config, the checkpoints and the logs.

```bash
python pipeline/step4_training/main_train_mmdet.py --model dino
python pipeline/step4_training/main_train_mmdet.py --model dino --dump-config /tmp/dino.py   # dry run: write the merged config, don't train
python pipeline/step4_training/main_train_mmdet.py --model dino --resume                     # continue after a crash
```

`--resume` continues the named run itself, from its `last_checkpoint`, not the pretrained weights.

### Evaluating a trained model

Add an `mmdet_sahi` model to the detection config with the run name as its `model_path` (e.g. `"model_path": "dino_run_fixed_lr"`), and include `mmdet_training.json` in that config. It resolves to the run's `best_coco_fitness_epoch_N.pth`, and `config_path` (if not set) to the merged config saved in the run.
