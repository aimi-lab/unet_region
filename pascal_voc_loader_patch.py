from pytorch_utils.pascal_voc_loader import pascalVOCLoader
from pytorch_utils import im_utils as imutls
from pytorch_utils.utils import get_opt_box
import numpy as np
import itertools
import matplotlib.pyplot as plt
from skimage.draw import polygon
from sklearn.metrics import confusion_matrix
import torch
from skimage import transform


class pascalVOCLoaderPatch(pascalVOCLoader):
    def __init__(self,
                 root,
                 patch_rel_size=0.3,
                 make_opt_box=False):
                 # seed=None):

        super().__init__(
            root)
        self.patch_rel_size = patch_rel_size
        self.make_opt_box = make_opt_box

        # if(seed is not None):
        #     np.random.seed(seed)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):

        sample = super().__getitem__(index)

        # select random truth
        lbl = np.random.choice(np.arange(len(sample['label/segmentations'])))

        truth = sample['label/segmentations'][lbl]
        im = sample['image']

        patch_width = int(max(truth.shape) * self.patch_rel_size)
        patch_width += not patch_width % 2

        # select random positive location
        loc = np.random.choice(np.nonzero(truth.ravel())[0])
        loc = np.array(np.unravel_index(loc, truth.shape))

        im = extract_patch(im, patch_width, loc.copy())
        truth = extract_patch(truth, patch_width, loc.copy())

        out = {
            'image': im,
            'label/segmentation': truth,
            'label/idx': sample['label/idxs'][lbl],
            'label/name': sample['label/names'][lbl],
            'loc': loc,
        }

        if (self.make_opt_box):
            opt_box = get_opt_box(truth > 0)
            # opt_box['mask'] = opt_box['mask'][..., np.newaxis]
            out['label/opt_box'] = opt_box['box']

        return out


    @staticmethod
    def collate_fn(samples):
        im = np.array([s['image'] for s in samples]).transpose((0, -1, 1, 2))
        lbl_segm = np.array([s['label/segmentation'] for s in samples]).transpose(
            (0, -1, 1, 2))

        if('label/opt_box' in samples[0].keys()):
            lbl_opt_box = np.array([s['label/opt_box']['mask'].astype(np.uint8)
                                    for s in samples]).transpose(
                (0, -1, 1, 2))
            lbl_opt_box = torch.from_numpy(lbl_opt_box).type(torch.float)
        else:
            lbl_opt_box = None
        im = torch.from_numpy(im).type(torch.float)
        lbl_segm = torch.from_numpy(lbl_segm).type(torch.float)
        lbl_idx = np.array([s['label/idx'] for s in samples])
        lbl_idx = torch.from_numpy(lbl_idx).type(torch.float)
        lbl_name = [s['label/name'] for s in samples]

        loc = np.array([s['loc'] for s in samples])

        out = {
            'image': im,
            'label/segmentation': lbl_segm,
            'label/opt_box': lbl_opt_box,
            'label/idx': lbl_idx,
            'label/name': lbl_name,
            'loc': loc
        }

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



