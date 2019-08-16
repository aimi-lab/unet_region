from __future__ import print_function
import torch.utils.data as data
import os
from PIL import Image
import numpy as np


class VOCSegmentation(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]

    def __init__(self,
                 root,
                 mode='train'):

        if(not ((mode == 'train') or (mode == 'val') or (mode == 'trainval'))):
            raise Exception('mode must be train, val, or trainval')
        self.root = root
        _voc_root = os.path.join(self.root, 'VOC2012')
        _list_dir = os.path.join(_voc_root, 'list')

        if mode == 'train':
            files = [os.path.join(_list_dir, 'train_aug.txt')]
        elif mode == 'val':
            files = [os.path.join(_list_dir, 'val.txt')]
        else:
            files = [os.path.join(_list_dir, 'val.txt'),
                     os.path.join(_list_dir, 'train_aug.txt')]

        self.images = []
        self.masks = []
        for f in files:
            with open(f, 'r') as lines:
                for line in lines:
                    _image = _voc_root + line.split()[0]
                    _mask = _voc_root + line.split()[1]
                    assert os.path.isfile(_image)
                    assert os.path.isfile(_mask)
                    self.images.append(_image)
                    self.masks.append(_mask)

    def __getitem__(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        _target = np.array(Image.open(self.masks[index]))

        _target[_target == 255] = 0

        return np.array(_img), np.array(_target)

    def __len__(self):
        return len(self.images)
