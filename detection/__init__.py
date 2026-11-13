"""The 2D detector: training and evaluation (main_train_detector.py, main_evaluate2d.py,
main_inference2d.py).

    training         build the YOLO model (version / size / P2 head / pretrained weights) and
                     run Ultralytics training
    detectors        SAHI / Ultralytics / MMDetection detector wrappers
    mmdet_remote     runs MMDetection models in a worker on another Python env (mmdet_worker.py)
    metrics          COCO-style AP (mAP with several classes) and precision / recall
    predictions      saved predictions (YOLO labels per frame) and their cache key
    evaluate_models  the evaluation pipeline: cached inference, scoring, reports

The datasets it evaluates are loaded by common/datasets.py. Its track plots, annotated videos and text reports are in visualise/.
"""
