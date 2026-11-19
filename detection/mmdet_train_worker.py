"""MMDetection training worker, run in a separate Python that has mmdet installed.

mmcv / mmdet cannot be installed in this repo's env (Python 3.14), so `detection/mmdet_training.py`
starts this script with the openmmlab env's interpreter, with CUDA_VISIBLE_DEVICES already set to
the one GPU to train on. It imports nothing from this repository and must stay Python 3.10-compatible.

    <openmmlab python> -I mmdet_train_worker.py --spec '<json>' [--dump-config merged_cfg.py]

The spec (built by mmdet_training.py from configs/mmdet_training.json) names the upstream recipe
config and says what to change in it: the dataset, the batch / accumulation (the LR is scaled
linearly against the recipe's total batch), the epochs and LR schedule, the seed, the checkpoints
and W&B. Everything else (model, pipeline, augmentation, optimizer) stays as the recipe has it.
With --dump-config the merged config is written there and nothing is trained.

Also defines FitnessHook, which adds the Ultralytics fitness score (0.1 mAP50 + 0.9 mAP50-95) to the
validation metrics as coco/fitness, so CheckpointHook can keep the best epoch by it.
"""

import argparse
import json
import os
import sys

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class FitnessHook(Hook):
    """metrics[out_key] = w_map50 * mAP50 + w_map * mAP after each validation.

    Priority NORMAL runs after RuntimeInfoHook (VERY_HIGH), which has already copied the metrics to
    the message hub, and before LoggerHook (BELOW_NORMAL) and CheckpointHook (VERY_LOW). The score is
    therefore written to both the metrics dict (CheckpointHook's save_best reads it) and the message
    hub (the text log and W&B read it).
    """

    priority = 'NORMAL'

    def __init__(self, map50_key='coco/bbox_mAP_50', map_key='coco/bbox_mAP',
                 out_key='coco/fitness', w_map50=0.1, w_map=0.9):
        self.map50_key, self.map_key, self.out_key = map50_key, map_key, out_key
        self.w_map50, self.w_map = w_map50, w_map

    def after_val_epoch(self, runner, metrics=None):
        if not metrics:
            return
        missing = [k for k in (self.map50_key, self.map_key) if k not in metrics]
        if missing:
            raise KeyError(f'FitnessHook: {missing} not in the metrics ({sorted(metrics)})')
        fitness = (self.w_map50 * float(metrics[self.map50_key]) +
                   self.w_map * float(metrics[self.map_key]))
        metrics[self.out_key] = fitness
        runner.message_hub.update_scalar(f'val/{self.out_key}', fitness)


def inner_dataset(ds):
    """The dataset itself, under any wrappers (RepeatDataset, ...)."""
    while 'dataset' in ds and 'ann_file' not in ds:
        ds = ds['dataset']
    return ds


def set_num_classes(module, n):
    if isinstance(module, dict):
        if 'num_classes' in module:
            module['num_classes'] = n
        for v in module.values():
            set_num_classes(v, n)
    elif isinstance(module, list):
        for v in module:
            set_num_classes(v, n)


def build_config(spec):
    from mmengine.config import Config

    cfg = Config.fromfile(spec['recipe'])
    recipe_epochs = cfg.train_cfg.max_epochs
    epochs = spec['epochs']

    # Pretrained weights: a whole detector (load_from) or only the backbone (init_cfg)
    if spec.get('backbone_pretrained'):
        cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=spec['backbone_pretrained'])
        cfg.load_from = None
    else:
        cfg.load_from = spec.get('load_from')
    if spec.get('custom_imports'):
        cfg.custom_imports = dict(imports=list(spec['custom_imports']), allow_failed_imports=False)

    # Dataset: the COCO export, the same annotation files for val and test
    data_root, classes = spec['data_root'], tuple(spec['classes'])
    cfg.data_root = data_root
    cfg.metainfo = dict(classes=classes, palette=[(220, 20, 60)] * len(classes))
    ann = {'train': os.path.join(data_root, spec['train_ann']),
           'val': os.path.join(data_root, spec['val_ann'])}
    for split, ann_split in (('train', 'train'), ('val', 'val'), ('test', 'val')):
        loader = cfg.get(f'{split}_dataloader')
        if not loader:
            continue
        ds = inner_dataset(loader.dataset)
        ds.data_root = data_root
        ds.metainfo = cfg.metainfo
        ds.ann_file = ann[ann_split]
        if ds.get('data_prefix') is not None:
            ds.data_prefix.img = spec['img_prefix']
        else:
            ds.data_prefix = dict(img=spec['img_prefix'])
    for name in ('val_evaluator', 'test_evaluator'):
        if cfg.get(name):
            cfg[name].ann_file = ann['val']
    set_num_classes(cfg.model, len(classes))

    # Batch, accumulation and the linearly scaled LR. The recipe's LR is tuned for its TOTAL batch
    # over all GPUs (auto_scale_lr.base_batch_size, e.g. 16 for 8xb2), not the per-GPU batch.
    if spec.get('batch_size'):
        cfg.train_dataloader.batch_size = spec['batch_size']
    if spec.get('val_batch_size'):
        cfg.val_dataloader.batch_size = spec['val_batch_size']
    accum = int(spec.get('accumulative_counts') or 1)
    if accum > 1:
        cfg.optim_wrapper.accumulative_counts = accum
    if spec.get('amp'):
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
    batch = cfg.train_dataloader.batch_size
    recipe_total = cfg.auto_scale_lr.base_batch_size
    base_lr = cfg.optim_wrapper.optimizer.lr
    cfg.optim_wrapper.optimizer.lr = base_lr * (batch * accum) / recipe_total

    # Epochs and LR schedule
    cfg.train_cfg.max_epochs = epochs
    cfg.train_cfg.val_interval = 1
    if 'max_epochs' in cfg:
        cfg.max_epochs = epochs  # the recipe's helper variable; only for a readable dump
    if spec['schedule'] == 'recipe':
        # The recipe's schedulers; iteration-based ones (warmup) as they are, epoch-based
        # milestones and ends stretched from the recipe's length to `epochs`.
        for sch in cfg.param_scheduler:
            if not sch.get('by_epoch', True):
                continue
            if 'milestones' in sch:
                sch['milestones'] = [int(round(m * epochs / recipe_epochs)) for m in sch['milestones']]
            if sch.get('end') is not None and sch['end'] >= recipe_epochs:
                sch['end'] = epochs
        # Stage switches (RT-DETR's last-epochs pipeline / data preprocessor) stretched the same
        # way, e.g. 117 of 120 -> 49 of 50. They fire once runner.epoch (0-based) >= switch_epoch.
        for hook in cfg.get('custom_hooks', []):
            if 'switch_epoch' in hook:
                hook['switch_epoch'] = int(round(hook['switch_epoch'] * epochs / recipe_epochs))
        if 'stage2_switch_epoch' in cfg:  # the recipe's helper variable; only for a readable dump
            cfg.stage2_switch_epoch = int(round(cfg.stage2_switch_epoch * epochs / recipe_epochs))
    elif spec['schedule'] == 'warmup_cosine':
        cfg.param_scheduler = [
            dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
            dict(type='CosineAnnealingLR', begin=0, end=epochs, by_epoch=True, T_max=epochs,
                 convert_to_iter_based=True)]
    else:
        raise SystemExit(f"schedule must be 'recipe' or 'warmup_cosine', not {spec['schedule']!r}")

    if spec.get('seed') is not None:
        cfg.randomness = dict(seed=spec['seed'])

    # Logging, checkpoints (every epoch, the latest max_keep_ckpts + the best by coco/fitness)
    cfg.default_hooks.logger.update(type='LoggerHook', interval=spec['log_interval'],
                                    log_metric_by_epoch=True)
    cfg.default_hooks.checkpoint.update(
        type='CheckpointHook', interval=1, by_epoch=True, max_keep_ckpts=spec['max_keep_ckpts'],
        save_best='coco/fitness', rule='greater')
    cfg.custom_hooks = list(cfg.get('custom_hooks', [])) + [dict(type='FitnessHook', priority='NORMAL')]

    cfg.visualizer.vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend', init_kwargs=dict(
            project=spec['wandb_project'], group=spec['wandb_group'], name=spec['run_name']))]
    cfg.work_dir = spec['work_dir']

    cfg.resume = bool(spec.get('resume'))
    if cfg.resume:
        # MMEngine resumes FROM load_from when it is set, which for DINO is the COCO-pretrained
        # detector, so point it at this run's own latest checkpoint instead.
        with open(os.path.join(spec['work_dir'], 'last_checkpoint')) as fh:
            cfg.load_from = fh.read().strip()

    summary = (f"batch {batch} x accumulation {accum} = effective {batch * accum} "
               f"(recipe total {recipe_total}); lr {base_lr:g} x {batch * accum / recipe_total:g} "
               f"= {cfg.optim_wrapper.optimizer.lr:g}; {epochs} epochs; schedule {cfg.param_scheduler}")
    return cfg, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True, help='JSON settings from mmdet_training.py')
    ap.add_argument('--dump-config', default=None, help='write the merged config here and exit')
    args = ap.parse_args()
    spec = json.loads(args.spec)

    if spec.get('repo'):  # a source checkout of mmdet first on sys.path (rtdetr-mmdet for RT-DETR)
        sys.path.insert(0, os.path.abspath(spec['repo']))
    import mmdet

    cfg, summary = build_config(spec)
    print(f"[mmdet_train_worker] mmdet from {mmdet.__file__}", flush=True)
    print(f"[mmdet_train_worker] {summary}", flush=True)
    if args.dump_config:
        cfg.dump(args.dump_config)
        print(f"[mmdet_train_worker] merged config written to {args.dump_config}; not training.")
        return

    import torch
    from mmengine.runner import Runner
    print(f"[mmdet_train_worker] CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"-> {torch.cuda.device_count()} visible GPU: {torch.cuda.get_device_name(0)}", flush=True)
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
