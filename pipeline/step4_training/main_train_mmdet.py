"""Train an MMDetection detector (DINO, RT-DETR, ...) from the config's "mmdet_training" section.

The upstream recipe, dataset, batch / gradient accumulation, GPU, epochs, LR schedule, seed and
W&B project come from the config, per model under "models". The LR is scaled linearly from the
recipe's total batch to batch x accumulation. Training runs in the openmmlab env's Python. The run
is written to output/training/<wandb_project>/<run name>/ (_2, _3, ... if that name is taken),
keeping the latest epochs and best_coco_fitness_epoch_N.pth (best by 0.1 mAP50 + 0.9 mAP50-95).
To evaluate it, give the run name as the model_path of an "mmdet_sahi" model in the config.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step4_training/main_train_mmdet.py --model dino
    python pipeline/step4_training/main_train_mmdet.py --model rtdetr --gpu 0
    python pipeline/step4_training/main_train_mmdet.py --model dino --resume                 # after a crash
    python pipeline/step4_training/main_train_mmdet.py --model dino --dump-config /tmp/dino.py   # dry run
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from common.config import load_config  # noqa: E402
from detection.mmdet_training import train_mmdet  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "configs" / "mmdet_training.json"   # "mmdet_training" holds the models and shared settings
MODEL = None               # a key of mmdet_training.models, e.g. "dino" or "rtdetr"
GPU = None                 # GPU index as nvidia-smi numbers it; None = the model's
RUN_NAME = None            # None = the model's run_name
RESUME = False             # True = continue RUN_NAME from its last checkpoint
DUMP_CONFIG = None         # a path = write the merged mmdet config there and exit (dry run)
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Train an MMDetection detector.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--model", default=MODEL, required=MODEL is None,
                        help="A key of mmdet_training.models, e.g. dino or rtdetr.")
    parser.add_argument("--gpu", type=int, default=GPU, help="GPU index as nvidia-smi numbers it.")
    parser.add_argument("--run-name", default=RUN_NAME, help="Run folder and W&B run name.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME,
                        help="Continue the run from its last checkpoint.")
    parser.add_argument("--dump-config", default=DUMP_CONFIG,
                        help="Write the merged mmdet config to this path and exit without training.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_mmdet(config, args.model, overrides={"gpu": args.gpu, "run_name": args.run_name},
                resume=args.resume, dump_config=args.dump_config)


if __name__ == "__main__":
    main()
