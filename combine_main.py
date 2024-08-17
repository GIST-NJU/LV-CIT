import traceback
import warnings
import time
import argparse
import os

from combine_runner import runner
from dataloaders import *

warnings.filterwarnings('ignore')

# data info
root = os.path.join("data", "combine")
combine_arg = "_v6_random_255_255_255_255_s1a0"

checkpoints_dir = os.path.join("checkpoints", "org")
checkpoints_save_dir = os.path.join("checkpoints", "save")
data_info = [
    {
        "data_name": "voc",
        "data": os.path.join(root, "3combine_img", f"VOC_20{combine_arg}"),
        "covering_array_type": [
            f"adaptive random_20_{k}_{tau}" for k in [4] for tau in [2]
        ],
        "num_classes": 20,
        "phase": "predict",
        "res_path": os.path.join(root, "4results", f"VOC_20{combine_arg}"),
        "inp_name": "data/voc/voc_glove_word2vec.pkl",
        "graph_file": "data/voc/voc_adj.pkl",
    },
    {
        "data_name": "coco",
        "data": os.path.join(root, "3combine_img", f"COCO_80{combine_arg}"),
        "covering_array_type": [
            f"adaptive random_80_{k}_{tau}" for k in [4] for tau in [2]
        ],
        "num_classes": 80,
        "phase": "predict",
        "res_path": os.path.join(root, "4results", f"COCO_80{combine_arg}"),
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
        "batch_size": 64,
        "threshold": 0.5,
        "workers": 1,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": os.path.join(checkpoints_dir, "msrn", "voc_checkpoint.pth.tar"),
        "evaluate": True,
        "pretrained": 1,
        "pretrain_model": "pretrained/resnet101_for_msrn.pth.tar",
        "pool_ratio": 0.2,
        "backbone": "resnet101",
        "save_model_path": os.path.join(checkpoints_save_dir, "msrn"),
    },
    # 1 ml gcn
    {
        "model_name": "mlgcn",
        "image_size": 448,
        "batch_size": 64,
        "threshold": 0.5,
        "workers": 2,
        "epochs": 20,
        "epoch_step": [30],
        "start_epoch": 0,
        "lr": 0.1,
        "lrp": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "print_freq": 0,
        "resume": os.path.join(checkpoints_dir, "mlgcn", "voc_checkpoint.pth.tar"),
        "evaluate": True,
        "save_model_path": os.path.join(checkpoints_save_dir, "mlgcn"),
    },
    # 2 asl
    {
        "model_name": "asl",
        "model_type": "tresnet_xl",
        "model_path": os.path.join(checkpoints_dir, "asl", "PASCAL_VOC_TResNet_xl_448_96.0.pth"),
        "workers": 1,
        "image_size": 448,
        "threshold": 0.8,
        "batch_size": 64,
        "print_freq": 64,
        # TODO save path
    },
]

TASKS = [
    # voc msrn
    {
        "task_name": "voc_msrn",
        "args": {**data_info[0], **model_info[0]},
        "dataloader": CombineVoc,
    },
    # voc mlgcn
    {
        "task_name": "voc_mlgcn",
        "args": {**data_info[0], **model_info[1]},
        "dataloader": CombineVoc,
    },
    # voc asl
    {
        "task_name": "voc_asl",
        "args": {**data_info[0], **model_info[3]},
        "dataloader": CombineVoc2,
    },

    # coco msrn
    {
        "task_name": "coco_msrn",
        "args": {
            **data_info[1], **model_info[0],
            "pool_ratio": 0.05,
            "resume": os.path.join(checkpoints_dir, "msrn", "coco_checkpoint.pth.tar"),
            "batch_size": 10,
        },
        "dataloader": CombineCoco,
    },
    # coco mlgcn
    {
        "task_name": "coco_mlgcn",
        "args": {
            **data_info[1], **model_info[1],
            "resume": os.path.join(checkpoints_dir, "mlgcn", "coco_checkpoint.pth.tar"),
            "batch_size": 80,
        },
        "dataloader": CombineCoco,
    },
    # coco asl
    {
        "task_name": "coco_asl",
        "args": {
            **data_info[1], **model_info[3],
            "model_type": "tresnet_l",
            "model_path": os.path.join(checkpoints_dir, "asl", "MS_COCO_TRresNet_L_448_86.6.pth"),
            "print_freq": 640,
            "batch_size": 64,
        },
        "dataloader": CombineCoco2,
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
        args.data = os.path.join(args.data, args.model_name)
        args.res_path = os.path.join(args.res_path, args.model_name)
        try:
            runner(args)
        except Exception:
            with open("errors.txt", 'a') as f:
                f.write(task["task_name"])
                traceback.print_exc()
                f.write(traceback.format_exc())
                f.write("\n")
        print("task: {} finished, time:{}".format(task["task_name"], time.time() - start))
