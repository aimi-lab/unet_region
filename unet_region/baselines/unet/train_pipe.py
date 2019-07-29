import os
from os.path import join as pjoin
import train
from unet_region.baselines.unet import params

if __name__ == "__main__":
    p = params.get_params()

    p.add('--in-dir', required=True)

    cfg = p.parse_args()

    in_dirs = [
        'Dataset00',
        'Dataset10',
        'Dataset20',
        'Dataset30',
    ]

    frames = [15, 52, 13, 16]

    # pre-train with pascal
    cfg.out_dir = pjoin(cfg.in_dir, 'unet_region')
    cfg.data_type = 'pascal'
    cfg.checkpoint_path = None

    cfg = train.main(cfg)
    cp_pascal = pjoin(cfg.run_dir, 'checkpoints',
                    'best_model.pth.tar')

    cfg.epochs = 50
    for d, f in zip(in_dirs, frames):
        cfg.in_dir = pjoin(cfg.root_dir, d)
        cfg.frames = [f]
        cfg.data_type = 'medical'
        cfg.checkpoint_path = cp_pascal

        train.main(cfg)

