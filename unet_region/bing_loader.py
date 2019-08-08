from unet_region.loader import Loader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import torch
import imgaug as ia
from skimage import transform, io
import glob
from os.path import join as pjoin
import pandas as pd
from scipy import interpolate
from imgaug.augmentables import Keypoint, KeypointsOnImage


class BingLoader(data.Dataset):
    def __init__(self, root_path, L=30,
                 augmentations=None, normalization=None):

        # when no locs are given, generate random locs
        self.root_path = root_path
        self.csv = pd.read_csv(
            pjoin(root_path, 'building_coords.csv'),
            header=None).drop(columns=0)

        self.img_paths = glob.glob(pjoin(root_path, 'images', '*.png'))
        self.img_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.mask_paths = glob.glob(pjoin(root_path, 'masks', '*.png'))
        self.mask_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.L = L
        
        self.augmentations = augmentations
        self.normalization = normalization

    def __len__(self):

        return len(self.img_paths)

    def __getitem__(self, index):

        img = io.imread(self.img_paths[index])
        mask = io.imread(self.mask_paths[index])
        mask = mask / mask.max()
        mask = mask.astype(np.uint8)

        corners = self.csv.iloc[index]
        poly = np.zeros([5, 2])
        nodes = np.zeros([self.L, 2])
        for c in range(4):
            poly[c, 0] = np.float(corners[1 + 2 * c])
            poly[c, 1] = np.float(corners[2 + 2 * c])
        poly[4, :] = poly[0, :]
        [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]],
                                       s=2,
                                       k=1,
                                       per=1)
        [nodes[:, 0], nodes[:, 1]] = interpolate.splev(
            np.linspace(0, 1, self.L), tck)
        
        sample = {'image': img,
                  'label/segmentation': mask,
                  'label/nodes': nodes}

        # do image augmentations
        if (self.augmentations is not None):
            orig_shape = sample['image'].shape
            aug_det = self.augmentations.to_deterministic()
            sample['image'] = aug_det.augment_image(sample['image'])

            truth = sample['label/segmentation']
            truth = ia.SegmentationMapOnImage(
                truth, shape=truth.shape, nb_classes=2)
            truth = aug_det.augment_segmentation_maps(
                [truth])[0].get_arr_int()[..., np.newaxis]
            sample['label/segmentation'] = truth

            if ('label/nodes' in sample.keys()):
                kp = sample['label/nodes']
                kp = KeypointsOnImage([Keypoint(x=r[1],
                                                y=r[0]-orig_shape[0])
                                       for r in kp],
                                      shape=orig_shape)
                sample['label/nodes'] = aug_det.augment_keypoints(
                    kp).to_xy_array()

        # do image normalization
        sample['image_unnormalized'] = sample['image']
        sample['image'] = self.normalization.augment_image(sample['image'])

        return sample

    @staticmethod
    def collate_fn(samples):

        im = np.array([np.rollaxis(s['image'], -1) for s in samples])
        im = torch.from_numpy(im).type(torch.float)

        im_unnorm = [s['image_unnormalized'] for s in samples]

        lbl_segm = np.array(
            [np.rollaxis(s['label/segmentation'], -1) for s in samples])
        lbl_nodes = [s['label/nodes'] for s in samples]

        lbl_segm = torch.from_numpy(lbl_segm.astype(int)).type(torch.float)

        out = {
            'image': im,
            'image_unnormalized': im_unnorm,
            'label/segmentation': lbl_segm,
            'label/nodes': lbl_nodes
        }

        return out

    @staticmethod
    def worker_init_fn_dummy(pid):
        pass

    @staticmethod
    def worker_init_fn(pid):
        np.random.seed(torch.initial_seed() % (2**31 - 1))
