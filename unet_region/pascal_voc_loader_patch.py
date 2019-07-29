from pytorch_utils.utils import get_opt_box
from unet_region import acm_utils as autls
import numpy as np
import itertools
import matplotlib.pyplot as plt
from skimage.draw import polygon
from sklearn.metrics import confusion_matrix
import torch
from skimage import transform, segmentation
import imgaug as ia
from unet_region.VOC2012 import VOC2012
from scipy.ndimage import distance_transform_edt


class pascalVOCLoaderPatch(VOC2012):

    def __init__(self,
                 root,
                 augmentations=None,
                 normalization=None,
                 patch_rel_size=0.3,
                 make_opt_box=False,
                 make_edt=False,
                 initial_ls_range=5,
                 tsdf_thr=None):
                 # seed=None):
        """
        """

        super().__init__(root)
        self.patch_rel_size = patch_rel_size
        self.make_opt_box = make_opt_box
        self.make_edt = make_edt
        self.tsdf_thr = tsdf_thr
        self.augmentations = augmentations
        self.normalization = normalization
        self.initial_ls_range = initial_ls_range

        self.to_collate_keys = ['image',
                                'label/segmentation']

        if(tsdf_thr is not None):
            self.to_collate_keys.append('label/tsdf')
            self.to_collate_keys.append('phi_tilda')
            self.to_collate_keys.append('label/U')

        if(make_edt is not None):
            self.to_collate_keys.append('label/edt_D')
            self.to_collate_keys.append('label/edt_beta')
            self.to_collate_keys.append('label/edt_kappa')

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

        if(self.augmentations is not None):
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

        if(self.normalization is not None):
            out['image'] = self.normalization.augment_image(im)


        if (self.make_opt_box):
            opt_box = get_opt_box(truth > 0)
            # opt_box['mask'] = opt_box['mask'][..., np.newaxis]
            out['label/opt_box'] = opt_box['box']

        if(self.make_edt is not None):
            truth_contour = (segmentation.find_boundaries(truth > 0))
            data = distance_transform_edt(np.logical_not(truth_contour))
            beta = data.copy()
            beta[truth > 0] = 0
            kappa = data.copy()
            kappa[truth == 0] = 0

            out['label/edt_D'] = data
            out['label/edt_beta'] = beta
            out['label/edt_kappa'] = kappa

        if(self.tsdf_thr is not None):
            out['label/tsdf'], contour = autls.make_sdf(truth[..., 0] > 0,
                                                        thr=self.tsdf_thr,
                                                        return_contour=True) 
            out['label/tsdf'] = out['label/tsdf'][..., np.newaxis]

            # make "shifted" initial level set
            out['phi_tilda'] = out['label/tsdf'] + \
                np.random.uniform(low=-self.initial_ls_range,
                                  high=self.initial_ls_range)

            dist = distance_transform_edt(np.logical_not(contour))
            dy, dx = np.gradient(dist)
            norm = np.linalg.norm(np.array((dy, dx)), axis=0)
            U = np.array((dy, dx)) / (norm + 1e-7)
            U = -U
            out['label/U'] = U

        return out

    def collate_fn(self, samples):

        out = {k: [dic[k] for dic in samples] for k in samples[0]}

        for k in out.keys():
            if(k in self.to_collate_keys):
                out[k] = np.array(out[k])
                out[k] = out[k].transpose((0, -1, 1, 2))
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



