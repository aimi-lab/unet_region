from os.path import join as pjoin
import yaml
from pytorch_utils.models.unet import UNet
import munch
import torch
import pandas as pd
import numpy as np
from pytorch_utils.pascal_voc_loader_patch import pascalVOCLoaderPatch
from pytorch_utils.pascal_voc_loader_patch import collate_fn_pascal_patch
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import f1_score

root = '/home/laurent.lejeune/medical-labeling'
run_dir = pjoin(root, 'unet_region', 'runs', '2019-04-10_11-38-11')

# cp_path = pjoin(run_dir, 'checkpoints', 'best_model.pth.tar')
cp_path = pjoin(run_dir, 'checkpoints', 'checkpoint.pth.tar')

cp = torch.load(cp_path, map_location=lambda storage, loc: storage)

with open(pjoin(run_dir, 'cfg.yml'), 'r') as stream:
    params = yaml.safe_load(stream)

params = munch.Munch(params)
params.cuda = True
params.num_workers = 8

device = torch.device('cuda' if params.cuda else 'cpu')

batch_to_device = lambda batch: {
    k: v.to(device) if (isinstance(v, torch.Tensor)) else v
    for k, v in batch.items()
}

model = UNet(
    in_channels=3,
    out_channels=1,
    depth=4,
    cuda=params.cuda,
    with_coordconv=params.with_coordconv,
    with_coordconv_r=params.with_coordconv_r,
    with_batchnorm=params.batch_norm)

model.load_state_dict(cp['state_dict'])
model.eval()

# Find threhsold on validation set
val_indices = np.array(pd.read_csv(pjoin(run_dir, 'val_sample.csv')))[:, 1]

valid_sampler = SubsetRandomSampler(val_indices)

valid_loader = pascalVOCLoaderPatch(
    root,
    patch_rel_size=params.patch_rel_size,
    cuda=params.cuda,
    do_reshape=True,
    make_opt_box=True,
    img_size=params.in_shape)

loader = torch.utils.data.DataLoader(
    valid_loader,
    batch_size=params.batch_size,
    num_workers=params.num_workers,
    # num_workers=0,
    collate_fn=collate_fn_pascal_patch,
    sampler=valid_sampler)

# Search these thresholds
thr = np.linspace(0.3, 0.95, 40)

pred_scores_dict = {t: [] for t in thr}
pbar = tqdm.tqdm(total=len(loader))
for i, data in enumerate(loader):
    data = batch_to_device(data)
    pred_scores = []
    pred_ = torch.sigmoid(model(data['image'])).detach().cpu().numpy()
    im_ = data['image'].cpu().numpy().transpose((0, 2, 3, 1))
    truth_ = data['label/opt_box'].cpu().numpy() > 0
    for t in thr:
        # s_ = np.sum(np.logical_and(truth_, pred_)) / np.sum(
        #     np.logical_or(truth_, pred_))
        s_ = f1_score(truth_.ravel(), pred_.ravel() > t)
        pred_scores_dict[t].append(s_)
    pbar.update(1)

pbar.close()

pred_scores_dict = {k: np.mean(v) for k, v in pred_scores_dict.items()}

pred_pd = pd.DataFrame.from_dict(pred_scores_dict, columns=['score'], orient='index')
pred_pd.to_csv(pjoin(run_dir, 'scores.csv'))


        # pred_ = model(data['image']).detach().cpu().numpy()

        # plt.subplot(131)
        # plt.imshow(truth_[0,0,...])
        # plt.subplot(132);plt.imshow(pred_[0,0,...])
        # plt.subplot(133)
        # plt.imshow(im_[0,...])
        # plt.show()
