from unet_region.bounding_box import BoundingBox, get_opt_box
from unet_region.dsac_utils import make_spline_contour
from unet_region.loader import Loader
import numpy as np
import itertools
import matplotlib.pyplot as plt
from torch.utils import data
import torch
import imgaug as ia
from skimage import transform, segmentation
from itertools import product
import glob
from os.path import join as pjoin
from PIL import Image
import pandas as pd
from scipy import interpolate
from scipy.ndimage import distance_transform_edt


class PatchLoader(Loader, data.Dataset):
    def __init__(self,
                 root_path,
                 truth_type,
                 ksp_pm_thr=0.8,
                 fix_frames=None,
                 locs=None,
                 fake_len=None,
                 augmentation=None,
                 normalization=None,
                 late_fn=None,
                 patch_rel_size=0.3,
                 cuda=False):

        Loader.__init__(self, root_path, truth_type, ksp_pm_thr, fix_frames)
        self.locs = locs

        self.late_fn = late_fn

        # when no locs are given, generate random locs
        self.fake_len = fake_len

        self.augmentation = augmentation
        self.normalization = normalization

        self.patch_rel_size = patch_rel_size

        self.ignore_collate = ['image_unnormalized', 'box', 'rel_size', 'loc', 'idx']

        self.device = torch.device('cuda' if cuda else 'cpu')

        if((self.fake_len is None) and (self.locs is None)):
            raise Exception('fake_len and locs cannot be both None')


    def __len__(self):

        if self.locs is None:
            return self.fake_len
        else:
            return self.locs.shape[0]

    def __getitem__(self, index):
            
        if(self.locs is None):
            # sample = super().__getitem__(super().__len__() % self.fake_len)
            sample = Loader.__getitem__(self,
                                        (Loader.__len__(self) % self.fake_len) - 1)
        else:
            sample = Loader.__getitem__(self,
                                        self.locs['frame'][index])

        truth = sample['label/segmentation']
        im = sample['image']

        orig_shape = im.shape[:2]

        patch_width = int(max(truth.shape) * self.patch_rel_size)
        patch_width += not patch_width % 2

        if (self.locs is None):
            loc = np.random.choice(np.nonzero(truth.ravel())[0])
            loc = np.array(np.unravel_index(loc, truth.shape))
        else:
            j = int(np.round(self.locs['x'][index] * (im.shape[1] - 1), 0))
            i = int(np.round(self.locs['y'][index] * (im.shape[0] - 1), 0))
            loc = np.array([i, j])

        if (im.max() > 255):
            im = (im / im.max() * 255).astype(np.uint8)
        if (len(im.shape) < 3):
            im = np.repeat(im[..., np.newaxis], 3, -1)

        if (im.shape[-1] > 3):
            im = im[..., 0:3]

        im = extract_patch(im, patch_width, loc.copy())
        truth = extract_patch(truth, patch_width, loc.copy())

        box = BoundingBox(((loc[0] - patch_width//2, loc[0] + patch_width//2),
                           (loc[1] - patch_width//2, loc[1] + patch_width//2)),
                          orig_shape)

        if (self.augmentation is not None):
            truth = ia.SegmentationMapOnImage(
                truth, shape=truth.shape, nb_classes=1 + 1)
            seq_det = self.augmentation.to_deterministic()
            im = seq_det.augment_image(im)
            truth = seq_det.augment_segmentation_maps(
                [truth])[0].get_arr_int()[..., np.newaxis]

        out = {
            'image_unnormalized': im,
            'image': im,
            'idx': index,
            'box': box,
            'label/segmentation': truth,
            'rel_size': self.patch_rel_size,
            'loc': loc
        }

        if(self.normalization is not None):
            out['image'] = self.normalization.augment_image(im)

        if(self.late_fn is not None):
            out = self.late_fn(out)

        return out

    def collate_fn(self, samples):
        out = {k: [dic[k] for dic in samples] for k in samples[0]}

        for k in out.keys():
            if(k not in self.ignore_collate):
                out[k] = np.array(out[k])
                out[k] = np.rollaxis(out[k], -1, 1)
                out[k] = torch.from_numpy(out[k]).float()

        return out

    @staticmethod
    def worker_init_fn_dummy(pid):
        pass

    @staticmethod
    def worker_init_fn(pid):
        np.random.seed(torch.initial_seed() % (2**31-1))


def extract_patch(arr, patch_width, loc):

    hw = patch_width // 2

    if (len(arr.shape) < 3):
        arr = arr[..., np.newaxis]

    # pad array
    arr = np.pad(
        arr, (((hw, ), (hw, ), (0, ))), 'constant', constant_values=0.)
    # adjust loc
    loc += hw

    patch = arr[loc[0] - hw:loc[0] + hw + 1, loc[1] - hw:loc[1] + hw + 1, ...]

    if ((patch.shape[0] != patch_width) | (patch.shape[1] != patch_width)):
        raise Exception('patch.shape= {} but must be {}'.format(
            patch.shape, patch_width))

    return patch
