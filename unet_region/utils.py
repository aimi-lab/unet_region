import os
from os.path import join as pjoin
import yaml
import shutil
import torch
import logging
import pandas as pd
import numpy as np
from itertools import product
from PIL import Image, ImageDraw


def read_locs_csv(csvName,
                  seqStart=None,
                  seqEnd=None):
    """
    """

    out = np.loadtxt(
        open(csvName, "rb"), delimiter=";", skiprows=5)[seqStart:seqEnd, :]
    if ((seqStart is not None) or (seqEnd is not None)):
        out[:, 0] = np.arange(0, seqEnd - seqStart)
    out = {
        k: v
        for k, v in zip(['frame', 'x', 'y'],
                        [out[:, 0], out[:, -2], out[:, -1]])
    }

    out = pd.DataFrame.from_dict(out)
    out['frame'] = out['frame'].astype(int)

    return out

def get_opt_box(truth, rotations=False):

    n_coords = 20
    n_angles = 16

    shape = truth.shape

    loc = [truth.shape[0] // 2, truth.shape[1] // 2]

    half_width = np.unique(
        np.linspace(5, truth.shape[0] // 2 - 1, n_coords, dtype=int))

    if rotations:
        angle = np.linspace(0, np.pi, n_angles)
    else:
        angle = [0.]

    dims = product(half_width, half_width, angle)
    boxes = []
    
    for hh, ww, a in dims:
        R = np.array([[np.cos(a), -np.sin(a)],
                      [np.sin(a), np.cos(a)]])
        top_left = np.dot(np.array((-ww, +hh)), R)
        bottom_right = np.dot(np.array((+ww, -hh)), R)

        i = sorted((int(top_left[1] + loc[0]), int(bottom_right[1] + loc[0])))
        j = sorted((int(top_left[0] + loc[1]), int(bottom_right[0] + loc[1])))
        boxes.append(BoundingBox(corners=(i, j),
                     orig_shape=truth.shape[:2]))

    boxes = [{
        'box': b,
        'mask': b.get_ellipse_mask(shape).astype(bool)
    } for b in boxes]

    boxes = [
        b for b in boxes
        if (not np.any(b['mask'] * np.logical_not(truth)))
    ]

    if(len(boxes) == 0):
        box = BoundingBox(
            corners=((loc[0] - 1, loc[0] + 1),
                        (loc[0] - 1, loc[0] + 1)),
            orig_shape=truth.shape[:2])
        boxes = [{'box': box,
                    'mask': box.get_ellipse_mask(shape).astype(bool)}]

    biggest = np.argmax([np.sum(b['mask']) for b in boxes])
    return boxes[biggest]

def setup_logging(log_path,
                  conf_path='logging.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = conf_path

    # Get absolute path to logging.yaml
    path = pjoin(os.path.dirname(__file__), path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            config['handlers']['info_file_handler']['filename'] = pjoin(
                log_path, 'info.log')
            config['handlers']['error_file_handler']['filename'] = pjoin(
                log_path, 'error.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def save_checkpoint(dict_,
                    is_best,
                    path,
                    fname_cp='checkpoint.pth.tar',
                    fname_bm='best_model.pth.tar'):

    cp_path = os.path.join(path, fname_cp)
    bm_path = os.path.join(path, fname_bm)

    if (not os.path.exists(path)):
        os.makedirs(path)

    try:
        state_dict = dict_['model'].module.state_dict()
    except AttributeError:
        state_dict = dict_['model'].state_dict()

    torch.save(state_dict, cp_path)

    if (is_best):
        shutil.copyfile(cp_path, bm_path)


def load_checkpoint(path, model, gpu=False):

    if (gpu):
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
    else:
        # checkpoint = torch.load(path)
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    return model

class BoundingBox:
    def __init__(self,
                 corners,
                 orig_shape,
                 expand_n_pix=10):

        i, j = corners
        if ((len(i) != 2) | (len(j) != 2)):
            raise Exception(
                'BoundingBox of shape ({},{}) but must be (2, 2)'.format(
                    len(i), len(j)))

        self.i = np.array(i)
        self.j = np.array(j)

        self.expand_n_pix = expand_n_pix
        self.orig_shape = orig_shape

    @classmethod
    def fromdict(cls, dict_):
        # create dummy box (will be overwritten)
        bbox = cls(corners=((0, 0), (0, 0)), orig_shape=-1)

        return bbox.load_state_dict(dict_)

    def contract_x(self):

        center = np.mean(self.j)

        new_j = np.array(self.j) + (self.expand_n_pix//2, -self.expand_n_pix//2)

        curr_dim = new_j[1] - new_j[0]

        if(curr_dim < 1):
            self.j = (center - (1, -1)).tolist()
            return False

        self.j = new_j.tolist()

        return True

    def contract_y(self):
        
        center = np.mean(self.i)

        new_i = np.array(self.i) + (self.expand_n_pix//2, -self.expand_n_pix//2)

        curr_dim = new_i[1] - new_i[0]

        if(curr_dim < 1):
            self.i = (center - (1, -1)).tolist()
            return False

        self.i = new_i.tolist()

        return True

    def __str__(self):
        str_ = '(i, j): ({}, {})'.format(self.i, self.j)
        return str_
    
    def expand_x(self):
        self.j = ((self.j - np.mean(self.j)) + (
            -self.expand_n_pix//2, self.expand_n_pix//2) + np.mean(self.j)).tolist()

    def expand_y(self):
        self.i = ((self.i - np.mean(self.i)) + (
            -self.expand_n_pix//2, self.expand_n_pix//2) + np.mean(self.i)).tolist()

    @property
    def coords(self):
        return [tuple(np.round(np.sort(self.i)).astype(int).tolist()),
                tuple(np.round(np.sort(self.j)).astype(int).tolist())]

    @property
    def center(self):
        return [np.mean(self.coords[0]),
                np.mean(self.coords[1])]

    def apply_actions(self, actions):
        bbox_out = BoundingBox(
            corners=(self.i, self.j),
            orig_shape=self.orig_shape,
            expand_n_pix=self.expand_n_pix)

        # flag checks for "illegal" actions
        ok = True
        for a in actions:
            if (a == 'ex'):
                bbox_out.expand_x()
            elif (a == 'ey'):
                bbox_out.expand_y()
            elif (a == 'cx'):
                ok = bbox_out.contract_x()
            elif (a == 'cy'):
                ok = bbox_out.contract_y()
            elif (a == 'cxy'):
                ok = bbox_out.contract_y()
                ok = bbox_out.contract_x()

        return bbox_out, ok

    @property
    def shape(self):
        return (self.i[1] - self.i[0] + 1,
                self.j[1] - self.j[0] + 1)

    def resize(self, new_shape):
        factor_i = self.orig_shape[0] / new_shape[0]
        factor_j = self.orig_shape[1] / new_shape[1]

        self.i = (self.i / factor_i).astype(int)
        self.j = (self.j / factor_j).astype(int)

        self.orig_shape = new_shape
        
        return self

    def paste_patch(self, patch):
        if(patch.shape[:2] != self.shape):
            raise Exception('patch must be same as bounding box shape')

        # generate mask on padded
        out = self.get_mask(shape=np.array(self.orig_shape) + np.array(self.shape),
                            offset=(self.shape[0]//2, self.shape[1]//2))

        np.place(out, out, patch)

        out = out[self.shape[0] // 2 : -self.shape[0] // 2,
                  self.shape[1] // 2 : -self.shape[1] // 2]

        return out
            
    def get_mask(self, shape=None, offset=(0, 0)):

        if(shape is None):
            if(self.orig_shape is None):
                raise Exception('set shape or orig_shape')
            else:
                shape = self.orig_shape

        coords = np.array(self.coords).T
        rr, cc = draw.rectangle(coords[0, :] + offset[0],
                                coords[1, :] + offset[1],
                                shape=shape[:2])

        out = np.zeros(shape[:2]).astype(int)
        out[rr, cc] = 1
        return out

    def get_ellipse_mask(self, shape=None):

        if(shape is None):
            shape = self.orig_shape

        box_ = np.array(self.coords).T.ravel().tolist()
        mask = Image.new('RGB', (shape[:2]))
        draw = ImageDraw.Draw(mask)
        draw.ellipse(box_, fill=1)
        del draw
        return np.array(mask).transpose((1, 0, 2))[..., 0]

    def apply_mask(self, im, color=(0, 0, 1), alpha=0.2, mode='box'):
        if not ((mode != 'box') | (mode != 'ellipse')):
            raise Exception('mode must be box or ellipse')

        if (mode == 'box'):
            mask = self.get_mask(im.shape)
        else:
            mask = self.get_ellipse_mask(im.shape)

        mask = np.array([mask * c for c in color]).transpose((1, 2, 0)).astype(float)
        im_ = im.copy()
        im_ = (1 - alpha) * im_ + alpha * mask

        return im_

    def concat_mask(self, im):
        im_ = im.copy()
        mask = self.get_mask(im_.shape[:2])[..., np.newaxis]

        if(im_.dtype == np.uint8):
            mask = mask.astype(np.uint8)*255

        im_ = np.concatenate((im_, mask), axis=-1)
        return im_

    def crop_resize_image(self, im, resize_shape=None, margin_ratio=0.05):
        im_ = im.copy()
        coords = self.coords

        range_i = np.arange(coords[0][0], coords[0][1])
        range_j = np.arange(coords[1][0], coords[1][1])

        # deal with zero width/height box
        if (((coords[0][1] - coords[0][0]) < 1)):
            range_i = np.arange(coords[0][0], coords[0][0] + 2)
        if (((coords[1][1] - coords[1][0]) < 1)):
            range_j = np.arange(coords[1][0], coords[1][0] + 2)

        im_ = im_.take(range_i, axis=0, mode='clip')
        im_ = im_.take(range_j, axis=1, mode='clip')
        if (resize_shape is not None):
            im_ = transform.resize(
                im_, resize_shape, preserve_range=True).astype(int)

        return im_

    def state_dict(self):
        return {
            'i': self.i,
            'j': self.j,
            'expand_n_pix': self.expand_n_pix,
            'orig_shape': self.orig_shape,
        }

    @staticmethod
    def load_state_dict(dict_):
        return BoundingBox((dict_['i'], dict_['j']),
                           dict_['orig_shape'],
                           expand_n_pix=dict_['expand_n_pix'])
