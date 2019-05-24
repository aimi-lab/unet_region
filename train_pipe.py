import os

root_dir = '/home/laurent.lejeune/medical-labeling'

in_dirs = [
    'Dataset20',
    # 'Dataset10',
]

frames = [15]

for d, f in zip(in_dirs, frames):
    os.system('python train.py \
--out-dir {}/unet_region \
--in-dir {}/{} \
--data-type medical \
--frames {} \
--checkpoint-path {}/unet_region/runs/2019-05-03_13-56-39/checkpoints/best_model.pth.tar --cuda'.format(root_dir, root_dir, d, f, root_dir))
