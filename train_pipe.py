import os
from os.path import join as pjoin
import train
import params

root_dir = '/home/laurent.lejeune/medical-labeling'

in_dirs = [
    'Dataset00',
    'Dataset10',
    'Dataset20',
    'Dataset30',
]

frames = [15, 52, 13, 16]

cuda = True

p = params.get_params()
cfg = p.parse_args()

# pre-train with pascal
cfg.out_dir = pjoin(root_dir, 'unet_region')
cfg.in_dir = root_dir
cfg.data_type = 'pascal'
cfg.cuda = cuda
cfg.checkpoint_path = None

cfg = train.main(cfg)
cp_pascal = pjoin(cfg.run_dir, 'checkpoints',
                  'best_model.pth.tar')

for d, f in zip(in_dirs, frames):
    cfg.out_dir = pjoin(root_dir, 'unet_region')
    cfg.in_dir = pjoin(root_dir, d)
    cfg.frames = [f]
    cfg.data_type = 'medical'
    cfg.checkpoint_path = cp_pascal
    cfg.cuda = cuda

    train.main(cfg)

