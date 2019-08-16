from unet_region.utils import get_opt_box
from unet_region import acm_utils as autls
import numpy as np
import itertools
import matplotlib.pyplot as plt
from skimage.draw import polygon
from sklearn.metrics import confusion_matrix
import torch
from skimage import transform, segmentation, measure
import imgaug as ia
from unet_region.VOC2012 import VOC2012
from scipy.ndimage import distance_transform_edt


class pascalVOCLoaderPatch(VOC2012):
    def __init__(self,
                 root,
                 augmentations=None,
                 normalization=None,
                 late_fn=None,
                 patch_rel_size=0.3):
        # seed=None):
        """
        """

        super().__init__(root)
        self.patch_rel_size = patch_rel_size
        self.augmentations = augmentations
        self.normalization = normalization
        self.late_fn = late_fn

        self.ignore_collate = ['image_unnormalized', 'loc']

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):

        im, tgt = super().__getitem__(index)

        # select random truth
        lbl = np.random.choice(np.unique(tgt)[1:])

        truth = tgt == lbl

        patch_width = int(max(truth.shape) * self.patch_rel_size)
        patch_width += not patch_width % 2

        # select random positive location
        loc = np.random.choice(np.nonzero(truth.ravel())[0])
        loc = np.array(np.unravel_index(loc, truth.shape))

        im = extract_patch(im, patch_width, loc.copy())
        truth = extract_patch(truth, patch_width, loc.copy())

        # select closed region centered
        truth = measure.label(truth, return_num=False)
        truth = truth == truth[truth.shape[0] // 2, truth.shape[1] // 2]

        if (self.augmentations is not None):
            aug_det = self.augmentations.to_deterministic()
            im = aug_det.augment_image(im)
            truth = ia.SegmentationMapOnImage(
                truth, shape=truth.shape, nb_classes=2)
            truth = aug_det.augment_segmentation_maps(
                [truth])[0].get_arr_int()[..., np.newaxis]

        out = {
            'image': im,
            'image_unnormalized': im,
            'label/segmentation': truth,
            'loc': loc,
        }

        if (self.normalization is not None):
            out['image'] = self.normalization.augment_image(im)

        if (self.late_fn is not None):
            out = self.late_fn(out)

        return out

    def collate_fn(self, samples):

        out = {k: [dic[k] for dic in samples] for k in samples[0]}

        for k in out.keys():
            if (k not in self.ignore_collate):
                out[k] = np.array(out[k])
                out[k] = np.rollaxis(out[k], -1, 1)
                out[k] = torch.from_numpy(out[k]).float()

        return out

    @staticmethod
    def worker_init_fn_dummy(pid):
        pass

    @staticmethod
    def worker_init_fn(pid):
        np.random.seed(torch.initial_seed() % (2**31 - 1))


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
