import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from PIL import Image
from tqdm import tqdm
from skimage import transform
from skimage import measure
import imgaug as ia
import pandas as pd


class pascalVOCLoader:
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(
            self,
            root):

        self.sbd_path = os.path.join(root, 'VOC2012', 'benchmark_RELEASE')
        self.root = os.path.join(root, 'VOC2012', 'VOCdevkit', 'VOC2012')
        self.n_classes = 21

        self.files = collections.defaultdict(list)

        # get all image file names
        path = pjoin(self.root, "SegmentationClass/pre_encoded",
                        "*.png")
        self.all_files = sorted(glob.glob(path))
        self.setup_annotations()

        # Find label (category)
        self.files_categories = sorted(glob.glob(pjoin(self.root,
                                                'ImageSets/Main/*_trainval.txt')))
        self.categories = [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        self.file_to_cat = dict()
        for f, c in zip(self.files_categories, self.categories):
            df = pd.read_csv(
                f,
                delim_whitespace=True,
                header=None,
                names=['filename', 'true'])
            self.file_to_cat.update({f_: c for f_ in df[df['true'] == 1]['filename']})

        # get all files for semantic segmentation with segmentation maps

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):

        truth_path = self.all_files[index]
        im_name = os.path.splitext(os.path.split(truth_path)[-1])[0]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")

        im = np.asarray(Image.open(im_path))
        segm = np.asarray(Image.open(truth_path))

        # identify connex labels
        lbls_idx = np.array([l for l in np.unique(segm) if l != 0])
        labels_names = [self.categories[l] for l in lbls_idx]

        # decompose truth
        truth = [(segm == l).astype(np.uint8)
                 for l in np.unique(segm)[1:]]

        # check if some objects have left the frame...
        # idx_ok = [i for i in range(len(truth)) if(np.sum(truth[i])/truth[i].size>0.005)]
        idx_ok = [i for i in range(len(truth)) if(np.sum(truth[i])>0)]
        truth = [t for i,t in enumerate(truth) if(i in idx_ok)]
        lbls_idx = [t for i,t in enumerate(lbls_idx) if(i in idx_ok)]
        labels_names = [t for i,t in enumerate(labels_names) if(i in idx_ok)]

        return {
            'image': im,
            'label/segmentations': truth,
            'label/idxs':  lbls_idx,
            'label/names': labels_names
        }

    def sample_uniform(self, n=1):
        ids = np.random.choice(np.arange(0,
                                         len(self),
                                         size=n,
                                         replace=False))

        out = [self.__getitem__(i) for i in ids] 
        if(n == 1):
            return out[0]
        else:
            return out

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray([
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ])

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        sbd_path = self.sbd_path
        target_path = pjoin(self.root, "SegmentationClass/pre_encoded")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        path = pjoin(sbd_path, "dataset/train.txt")
        sbd_train_list = tuple(open(path, "r"))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
        train_aug = self.files["train"] + sbd_train_list

        pre_encoded = glob.glob(pjoin(target_path, "*.png"))

        if len(pre_encoded) != 9733:
            print("Pre-encoding segmentation masks...")
            for ii in tqdm(sbd_train_list):
                lbl_path = pjoin(sbd_path, "dataset/cls", ii + ".mat")
                data = io.loadmat(lbl_path)
                lbl = data["GTcls"][0]["Segmentation"][0].astype(np.int32)
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(pjoin(target_path, ii + ".png"), lbl)
