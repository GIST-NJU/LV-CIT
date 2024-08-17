import traceback
import warnings
import time
import argparse
import os
import math

from default_runner import runner
from dataloaders import *

warnings.filterwarnings('ignore')


# data info
root = "./data"
data_info = [
    {
        "data_name": "voc",
        "data": os.path.join(root, "voc"),
        "phase": "test",
        "num_classes": 20,
        "res_path": os.path.join(root, "voc", "results"),
        "inp_name": "data/voc/voc_glove_word2vec.pkl",
        "graph_file": "data/voc/voc_adj.pkl",
    },
    {
        "data_name": "coco",
        "data": os.path.join(root, "coco", "coco"),
        "phase": "val",
        "num_classes": 80,
        "res_path": os.path.join(root, "coco", "results"),
        "inp_name": "data/coco/coco_glove_word2vec.pkl",
        "graph_file": "data/coco/coco_adj.pkl",
    },
]

# model info
model_info = [
    # 0 msrn
    {
        "model_name": "msrn",
        "image_size": 448,
        "batch_size": 8,
        "threshold": 0.5,
        "workers": 4,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": "checkpoints/org/msrn/voc_checkpoint.pth.tar",
        "evaluate": True,
        "pretrained": 1,
        "pretrain_model": "pretrained/resnet101_for_msrn.pth.tar",
        "pool_ratio": 0.2,
        "backbone": "resnet101",
        "save_model_path": "checkpoints/save/msrn",
    },
    # 1 ml gcn
    {
        "model_name": "mlgcn",
        "image_size": 448,
        "batch_size": 8,
        "threshold": 0.5,
        "workers": 4,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "lrp": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": "checkpoints/org/ml_gcn/voc_checkpoint.pth.tar",
        "evaluate": True,
        "save_model_path": "checkpoints/save/mlgcn",
    },
    # 3 asl
    {
        "model_name": "asl",
        "model_type": "tresnet_xl",
        "model_path": "checkpoints/org/asl/PASCAL_VOC_TResNet_xl_448_96.0.pth",
        "workers": 4,
        "image_size": 448,
        "threshold": 0.8,
        "batch_size": 8,
        "print_freq": 64,
    },
]

TASKS = [
    # voc msrn
    {
        "task_name": "voc_msrn",
        "args": {**data_info[0], **model_info[0]},
        "dataloader": Voc2007Classification,
    },
    # voc mlgcn
    {
        "task_name": "voc_mlgcn",
        "args": {**data_info[0], **model_info[1]},
        "dataloader": Voc2007Classification,
    },
    # voc asl
    {
        "task_name": "voc_asl",
        "args": {**data_info[0], **model_info[3]},
        "dataloader": Voc2007Classification2,
    },

    # coco msrn
    {
        "task_name": "coco_msrn",
        "args": {
            **data_info[1], **model_info[0],
            "pool_ratio": 0.05,
            "resume": "checkpoints/org/msrn/coco_checkpoint.pth.tar",
        },
        "dataloader": COCO2014Classification,
    },
    # coco mlgcn
    {
        "task_name": "coco_mlgcn",
        "args": {
            **data_info[1], **model_info[1],
            "resume": "checkpoints/org/ml_gcn/coco_checkpoint.pth.tar",
        },
        "dataloader": COCO2014Classification,
    },
    # coco asl
    {
        "task_name": "coco1_asl",
        "args": {
            **data_info[1], **model_info[3],
            "batch_size": 1,
            "model_type": "tresnet_l",
            "model_path": "checkpoints/org/asl/MS_COCO_TRresNet_L_448_86.6.pth",
            "part": 10,
            "print_freq": 640,
        },
        "dataloader": COCO2014Classification2,
    },
]


if __name__ == "__main__":
    with open("errors.txt", 'w') as f:
        f.write("")
    for task in TASKS:
        print("task: {} started".format(task["task_name"]))
        start = time.time()
        args = argparse.Namespace(**task["args"])
        args.dataloader = task["dataloader"]
        try:
            runner(args)
        except Exception as e:
            with open("errors.txt", 'a') as f:
                f.write(task["task_name"])
                traceback.print_exc()
                f.write(traceback.format_exc())
                f.write("\n")
        print("task: {} finished, time:{}".format(task["task_name"], time.time() - start))
