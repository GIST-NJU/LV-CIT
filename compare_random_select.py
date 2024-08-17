import glob
import os
import pandas as pd
import joblib
from util import cal_score


VERSION = "_v6_random_255_255_255_255_s1a0"
select_num = 10

src_root = os.path.join("data")
dst_root = os.path.join("data", "combine", "4results")
ca_root = os.path.join("data", "combine", "1covering_array")
dst_dir_name = {
    "voc": "VOC_20",
    "coco": "COCO_80"
}
ca_types = [f"_{k}_{tau}" for k in [4] for tau in [2]]
num_classes = {
    "voc": 20,
    "coco": 80
}
cat2idxes = {
    "voc": {
        'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
        'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
        'cow': 9, 'dining table': 10, 'dog': 11, 'horse': 12,
        'motorbike': 13, 'person': 14, 'potted plant': 15,
        'sheep': 16, 'sofa': 17, 'train': 18, 'tv': 19
    },
    "coco": {
        "airplane": 0, "apple": 1, "backpack": 2, "banana": 3, "baseball bat": 4,
        "baseball glove": 5, "bear": 6, "bed": 7, "bench": 8, "bicycle": 9,
        "bird": 10, "boat": 11, "book": 12, "bottle": 13, "bowl": 14,
        "broccoli": 15, "bus": 16, "cake": 17, "car": 18, "carrot": 19,
        "cat": 20, "cell phone": 21, "chair": 22, "clock": 23, "couch": 24,
        "cow": 25, "cup": 26, "dining table": 27, "dog": 28, "donut": 29,
        "elephant": 30, "fire hydrant": 31, "fork": 32, "frisbee": 33, "giraffe": 34,
        "hair dryer": 35, "handbag": 36, "horse": 37, "hot dog": 38, "keyboard": 39,
        "kite": 40, "knife": 41, "laptop": 42, "microwave": 43, "motorcycle": 44,
        "mouse": 45, "orange": 46, "oven": 47, "parking meter": 48, "person": 49,
        "pizza": 50, "potted plant": 51, "refrigerator": 52, "remote": 53, "sandwich": 54,
        "scissors": 55, "sheep": 56, "sink": 57, "skateboard": 58, "skis": 59,
        "snowboard": 60, "spoon": 61, "sports ball": 62, "stop sign": 63, "suitcase": 64,
        "surfboard": 65, "teddy bear": 66, "tennis racket": 67, "tie": 68, "toaster": 69,
        "toilet": 70, "toothbrush": 71, "traffic light": 72, "train": 73, "truck": 74,
        "tv": 75, "umbrella": 76, "vase": 77, "wine glass": 78, "zebra": 79
    }
}
models = [
    ("msrn", "voc"),
    ("mlgcn", "voc"),
    ("asl", "voc"),
    ("msrn", "coco"),
    ("mlgcn", "coco"),
    ("asl", "coco"),
]


if __name__ == "__main__":
    for i in range(5):
        selected = {}
        for model, data in models:
            res_df = joblib.load(f"data/{data}/res_{data}_{model}_val.pkl")
            res_df["filename"] = res_df["img"]
            res_df["labels_gt"] = res_df["title"].apply(
                lambda x: "|".join(sorted([str(i) for i in x]))
            )
            res_df["labels"] = res_df["pred"].apply(
                lambda x: "|".join(sorted([str(i) for i in x]))
            )
            res_df["pass"] = res_df.apply(
                lambda x: 1 if x["labels_gt"] == x["labels"] else 0, axis=1
            )
            res_df = res_df[['filename', 'labels_gt', 'labels', 'pass']]
            for ca_type in ca_types:
                ca_type = f"adaptive random_{num_classes[data]}" + ca_type
                dst_dir = os.path.join(dst_root, dst_dir_name[data] + VERSION, "random", f"{ca_type}_No{i+1}")
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                print()
                ca_path = glob.glob(os.path.join(ca_root, ca_type.split("_")[0], f"ca_{ca_type}*.csv"))[i]
                ca_df = pd.read_csv(ca_path)
                img_num = len(ca_df) * select_num
                if f"{data}_{ca_type}" not in selected:
                    res = res_df.sample(img_num).reset_index(drop=True)
                    selected[f"{data}_{ca_type}"] = res[["filename"]]
                else:
                    res = pd.merge(selected[f"{data}_{ca_type}"]["filename"], res_df, on="filename", how="left")
                res["score"] = res.apply(
                    lambda x: cal_score(
                        x["labels_gt"], x["labels"], num_classes[data], int(ca_type.split("_")[-1]), cat2idxes[data]
                    ), axis=1
                )
                print(res)
                res.to_csv(os.path.join(dst_dir, f"res_{data}_{model}_{ca_type}_cmp_random_{i+1}.csv"), index=False)
