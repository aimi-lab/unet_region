from pascal_voc_loader_patch import pascalVOCLoaderPatch
from os.path import join as pjoin
from delse import DELSE
from my_augmenters import rescale_augmenter, Normalize
from imgaug import augmenters as iaa
import params
from acm_utils import acm_ls
import torch

root_path = '/home/ubelix'
cp_path = pjoin('runs/acm/runs',
                'pascal_2019-07-05_16-00/checkpoints/checkpoint_ls.pth.tar')


def main(cfg):

    import pdb; pdb.set_trace() ## DEBUG ##
    model = DELSE()
    model.eval()

    dict_ = model.state_dict()
    torch.save(dict_, 'test.pth.tar')

    dict_ = torch.load('test.pth.tar',
                       map_location='cpu')
    model.load_state_dict(dict_)
    dict_keys = dict_.keys()

    checkpoint = torch.load(cfg.checkpoint_path,
                            map_location='cpu')
    cp_keys = checkpoint['state_dict'].keys()
    model.load_state_dict(checkpoint['state_dict'])

    in_shape = [cfg.in_shape] * 2

    transf = iaa.Sequential([
        iaa.Resize(in_shape), rescale_augmenter
    ])

    normalization = Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


    loader = pascalVOCLoaderPatch(
        cfg.in_dir,
        patch_rel_size=cfg.patch_rel_size,
        tsdf_thr=cfg.tsdf_thr,
        augmentations=transf,
        normalization=normalization)



    acm_fun = lambda phi, v, m: acm_ls(
        phi, v, m, 1, cfg.n_iters, vec_field=True)

    batch_to_device = lambda batch: {
        k: v.to(model.device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    for i, data in enumerate(loader):

        data = batch_to_device(data)
        out_mot, out_mod, out_ls = model(
            data['image'])

        import pdb; pdb.set_trace() ## DEBUG ##
        phis = torch.stack([
            acm_fun(out_ls[i].squeeze(),
                    out_mot[i].squeeze(),
                    out_mod[i].squeeze())[-1]
            for i in range(out_ls.shape[0])
        ])


if __name__ == "__main__":

    p = params.get_params()
    # p.add('--out-dir', required=True)
    # p.add('--checkpoint-path', required=True)

    cfg = p.parse_args()

    cfg.in_dir = '/home/ubelix/data/VOC2012/VOCdevkit'
    cfg.checkpoint_path = '/home/ubelix/runs/acm/runs/pascal_2019-07-09_10-00/checkpoints/checkpoint_ls.pth.tar'
    cfg.out_dir = '/home/ubelix/runs/acm/runs/pascal_2019-07-09_10-00'

    # cfg.checkpoint_path = None
    # cfg.n_workers = 0
    # cfg.data_type = 'pascal'
    # cfg.out_dir = '/home/ubelix/data'
    # cfg.in_dir = '/home/ubelix/data/VOCdevkit/'

    # cfg.data_type = 'medical'
    # cfg.out_dir = '/home/ubelix/medical-labeling/unet_region/runs/'
    # cfg.in_dir = '/home/ubelix/medical-labeling/Dataset20'
    # cfg.frames = [30]

    main(cfg)
