import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle

urls = {'train_img': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_img': 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}
cat2idx = {
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


def download_coco2014(root, phase):
    if not os.path.exists(root):
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/')
    data = os.path.join(root, '../data/')
    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    else:
        raise ValueError('phase should be train or val')
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    # extract file
    img_data = os.path.join(data, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.Popen('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(data, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014Classification(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.img_path = os.path.join(self.root, '../data', '{}2014'.format(self.phase))
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] COCO2014Classification Set=%s number of classes=%d  number of images=%d' % (
            self.phase, self.num_classes, len(self.img_list)
        ))

    def get_anno(self):
        list_path = os.path.join(self.root, '../data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        # self.cat2idx = json.load(open(os.path.join(self.root, '../data', 'category.json'), 'r'))
        self.cat2idx = cat2idx

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.img_path, filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, torch.tensor(self.inp)), target

    def get_number_classes(self):
        return self.num_classes

    def get_cat2id(self):
        return self.cat2idx
