"""Train an MMDetection detector (DINO, RT-DETR, ...) from the config's "mmdet_training" section.

mmcv / mmdet cannot be installed in this env, so the training itself runs in
`mmdet_train_worker.py`, started with the openmmlab env's Python ("python" in the config). That
process is started with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=<gpu>, so "gpu" is
the GPU index as nvidia-smi numbers it and the training sees it as cuda:0.

A run is written to <output_dir>/training/<wandb_project>/<run name>/ (_2, _3, ... if the name is
taken; resume continues the named run itself). It holds the merged config (<recipe name>.py),
epoch_N.pth for the latest epochs, best_coco_fitness_epoch_N.pth and the logs. To evaluate it,
give the run name as the model_path of an "mmdet_sahi" model in the detection config.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .training import free_run_name

WORKER = Path(__file__).with_name("mmdet_train_worker.py")
SHARED_KEYS = ("data_root", "train_ann", "val_ann", "img_prefix", "classes", "wandb_project",
               "wandb_group", "epochs", "seed", "schedule", "log_interval", "max_keep_ckpts")


def mmdet_run_dir(output_dir: str | Path, wandb_project: str, run: str) -> Path:
    return Path(output_dir) / "training" / wandb_project / run


def build_spec(config: dict, model: str, overrides: dict | None = None) -> dict:
    """The worker's settings: the section's shared keys with the model's own keys on top."""
    if "mmdet_training" not in config:
        raise SystemExit("config has no 'mmdet_training' section. See configs/mmdet_training.example.json.")
    section = config["mmdet_training"]
    models = {k: v for k, v in section.get("models", {}).items() if not k.startswith("_")}
    if model not in models:
        raise SystemExit(f"mmdet_training.models has no '{model}' (has: {', '.join(models) or 'none'}).")
    spec = {k: section[k] for k in SHARED_KEYS if k in section}
    spec.update({k: v for k, v in models[model].items() if not k.startswith("_")})
    spec.update({k: v for k, v in (overrides or {}).items() if v is not None})
    spec.setdefault("run_name", f"{model}_run")
    spec["python"] = spec.get("python") or section.get("python")
    for key in ("recipe", "data_root", "python"):
        if not spec.get(key) or not Path(spec[key]).exists():
            raise SystemExit(f"mmdet_training ({model}): {key} not found: {spec.get(key)}")
    return spec


def train_mmdet(config: dict, model: str, overrides: dict | None = None,
                resume: bool = False, dump_config: str | None = None) -> Path:
    """Train mmdet_training.models[model] (keys in overrides win). Returns the run folder."""
    spec = build_spec(config, model, overrides)
    project_dir = Path(config["output_dir"]) / "training" / spec["wandb_project"]
    if resume:
        work_dir = project_dir / spec["run_name"]
        if not (work_dir / "last_checkpoint").is_file():
            raise SystemExit(f"Nothing to resume: {work_dir / 'last_checkpoint'} not found.")
    else:
        spec["run_name"] = free_run_name(project_dir, spec["run_name"])
        work_dir = project_dir / spec["run_name"]
    spec["work_dir"], spec["resume"] = str(work_dir), resume

    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(spec["gpu"]))
    cmd = [spec["python"], "-I", str(WORKER), "--spec", json.dumps(spec)]
    if dump_config:
        cmd += ["--dump-config", str(Path(dump_config).resolve())]

    print("==================================================")
    print(f" {'Resuming' if resume else 'Dry run of' if dump_config else 'Launching'} "
          f"{model} on GPU {spec['gpu']} (nvidia-smi index)")
    print(f" Recipe: {spec['recipe']}")
    print(f" Data:   {spec['data_root']}")
    print(f" W&B:    {spec['wandb_project']} / {spec['wandb_group']} / {spec['run_name']}")
    print(f" Output: {work_dir}")
    print("==================================================", flush=True)
    code = subprocess.run(cmd, env=env).returncode
    if code:
        raise SystemExit(f"mmdet training worker exited with code {code}")
    if not dump_config:
        print(f"\nBest checkpoint: {work_dir}/best_coco_fitness_epoch_N.pth\n"
              f'To evaluate it, add an "mmdet_sahi" model with "model_path": "{spec["run_name"]}"')
    return work_dir
