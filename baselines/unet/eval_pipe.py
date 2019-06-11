import os
from os.path import join as pjoin
import params
import eval

root_dir = '/home/laurent.lejeune/medical-labeling'

dirs = {
    'Dataset00_2019-05-29_19-29': [
        'Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04',
        'Dataset05'
    ],

    'Dataset10_2019-05-29_20-31':
    ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'],

    'Dataset20_2019-05-29_21-33': [
        'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23',
        'Dataset24',
        'Dataset25'
    ],

    'Dataset30_2019-05-29_22-35':
    ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33', 'Dataset34']
}

pascal_dir = 'pascal_2019-05-29_08-20'

cuda = False

p = params.get_params()
p.add('--in-dir')
p.add('--root-dir')
cfg = p.parse_args()
cfg.root_dir = root_dir

for run_dir in dirs.keys():
    for s in dirs[run_dir]:
        cfg.in_dir = pjoin(root_dir, s)
        cfg.cuda = cuda
        cfg.csv_loc_file = pjoin(root_dir, s, 'gaze-measurements',
                                 'video1.csv')

        # with pascal trained
        cfg.run_dir = pjoin(root_dir, 'unet_region', 'runs', pascal_dir)
        cfg.in_dir = pjoin(root_dir, s)
        eval.main(cfg)

        # with self trained
        cfg.run_dir = pjoin(root_dir, 'unet_region', 'runs', run_dir)
        eval.main(cfg)
