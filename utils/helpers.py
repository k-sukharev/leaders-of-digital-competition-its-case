import os
import json
import pickle
import random

import cv2
import imgaug as ia
import numpy as np
import torch

from pytorch_lightning import seed_everything


def fix_seed(seed: int):
    """Seeds and fixes every possible random state."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    ia.random.seed(seed)
    seed_everything(seed)


def dump(obj, filepath):
    with open(filepath, "wb") as fout:
        pickle.dump(obj, fout, -1)


def load(filepath):
    with open(filepath, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def get_img_idxs(path):
    img_idxs = sorted(
        [
            x.split('.')[0]
            for x in os.listdir(path)
            if 'png' in x
        ]
    )
    return img_idxs


def process_json(path):
    with open(path) as inf:
        layout = json.load(inf)

    h, w = layout['imageHeight'], layout['imageWidth']
    mask = np.zeros((h, w), np.uint8)

    for shape in layout['shapes']:
        label = shape['label']
        polygon = np.array([point[::-1] for point in shape['points']])
        cv2.fillPoly(mask, [polygon[:, [1, 0]]], 1)

    label = layout['shapes'][0]['label']

    return mask.astype(np.int32), label
