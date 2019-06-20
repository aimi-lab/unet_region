import torch
from torch.utils import data
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt


class LearningLoader(data.Dataset):
    """Data loader that deals with augmentation 
    """

    def __init__(self, loader, augmentations=None, normalization=None):
        """
        """

        self.loader = loader
        self.augmentations = augmentations
        self.normalization = normalization

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):

        sample = self.loader[index]

        import pdb; pdb.set_trace()
        # plt.imshow(sample['image']);plt.scatter(sample['label/nodes'][:, 1], sample['label/nodes'][:, 0]);plt.show()

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
        im = np.array([s['image'] for s in samples]).transpose((0, -1, 1, 2))
        im_unnorm = np.array([s['image_unnormalized']
                              for s in samples]).transpose((0, -1, 1, 2))
        lbl_segm = np.array([s['label/segmentation']
                             for s in samples]).transpose((0, -1, 1, 2))

        if ('label/opt_box' in samples[0].keys()):
            lbl_opt_box = np.array([
                s['label/opt_box']['mask'].astype(np.uint8) for s in samples
            ]).transpose((0, -1, 1, 2))
            lbl_opt_box = torch.from_numpy(lbl_opt_box).type(torch.float)
        else:
            lbl_opt_box = None
        im = torch.from_numpy(im).type(torch.float)
        im_unnorm = torch.from_numpy(im_unnorm).type(torch.float)
        lbl_segm = torch.from_numpy(lbl_segm).type(torch.float)

        loc = np.array([s['loc'] for s in samples])

        out = {
            'image': im,
            'image_unnormalized': im_unnorm,
            'label/segmentation': lbl_segm,
            'loc': loc
        }

        return out

    @staticmethod
    def worker_init_fn_dummy(pid):
        pass

    @staticmethod
    def worker_init_fn(pid):
        np.random.seed(torch.initial_seed() % (2**31 - 1))
