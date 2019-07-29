from pascal_voc_loader_patch import pascalVOCLoaderPatch
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from my_augmenters import rescale_augmenter, Normalize
import torch

transf = iaa.Sequential([
    # iaa.Invert(0.5) if 'Dataset1' in cfg.in_dir else iaa.Noop(),
    iaa.Noop(),
    iaa.SomeOf(3, [
        iaa.Affine(rotate=iap.Uniform(-15., 15.)),
        iaa.Affine(shear=iap.Uniform(-15., -15.)),
        iaa.Fliplr(1.),
        iaa.Flipud(1.),
        iaa.GaussianBlur(sigma=iap.Uniform(0.0, 0.05))
    ]),
    iaa.Resize(224), rescale_augmenter
])

normalization = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
dataset = pascalVOCLoaderPatch('/home/ubelix/data/VOC2012',
                               augmentations=transf,
                               normalization=normalization,
                               make_edt=True)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    collate_fn=dataset.collate_fn,
    worker_init_fn=dataset.worker_init_fn)

for sample in train_loader:

    plt.subplot(221)
    plt.imshow(sample['image_unnormalized'][0])
    plt.subplot(222)
    plt.imshow(sample['label/segmentation'][0, 0,...])
    plt.subplot(223)
    plt.imshow(sample['label/edt_D'][0, 0,...])
    plt.subplot(224)
    plt.imshow(sample['label/edt_beta'][0, 0, ...])
    plt.show()
