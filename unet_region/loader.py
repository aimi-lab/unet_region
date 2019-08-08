import os
from os.path import join as pjoin
from skimage import io
import glob
import numpy as np
from unet_region.utils import get_opt_box


class Loader:
    def __init__(self,
                 root_path=None,
                 truth_type=None,
                 ksp_pm_thr=0.8,
                 fix_frames=None):
        """

        """

        self.root_path = root_path
        self.truth_type = truth_type
        self.ksp_pm_thr = ksp_pm_thr

        exts = ['*.png', '*.jpg', '*.jpeg']
        img_paths = []
        for e in exts:
            img_paths.extend(sorted(glob.glob(pjoin(root_path,
                                           'input-frames',
                                                     e))))
        if(truth_type == 'hand'):
            truth_paths = []
            for e in exts:
                truth_paths.extend(sorted(glob.glob(pjoin(root_path,
                                            'ground_truth-frames',
                                                        e))))
            if(fix_frames is not None):
                self.truth_paths = [truth_paths[i] for i in range(len(truth_paths))
                                    if(i in fix_frames)]
                self.img_paths = [img_paths[i] for i in range(len(img_paths))
                                  if(i in fix_frames)]
            else:
                self.truth_paths = truth_paths
                self.img_paths = img_paths

            self.truths = [
                io.imread(f).astype('bool') for f in self.truth_paths
            ]
            self.truths = [t if(len(t.shape) < 3) else t[..., 0]
                           for t in self.truths]
            self.imgs = [
                io.imread(f) for f in self.img_paths]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        truth = self.truths[idx]
        im = self.imgs[idx]

        return {'image': self.imgs[idx],
                'key': idx,
                'label/segmentation': self.truths[idx]}
