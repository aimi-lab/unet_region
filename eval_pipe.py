import os
from os.path import join as pjoin

root_dir = '/home/laurent.lejeune/medical-labeling'

in_dirs = [
    # [
    #     'Dataset00',
    #     'Dataset01',
    #     'Dataset02',
    #     'Dataset03',
    #     'Dataset04',
    #     'Dataset05'],
    [
    # 'Dataset10',
    # 'Dataset11',
    'Dataset12',
    'Dataset13'
    ],
    ['Dataset20',
    'Dataset21',
    'Dataset22',
    'Dataset23',
    'Dataset24',
    'Dataset25'],
    ['Dataset30',
    'Dataset31',
    'Dataset32',
    'Dataset33',
    'Dataset34']
]

run_dirs = [
    'unet_region/runs/Dataset00_2019-04-10_11-38-11',
    'unet_region/runs/Dataset10_2019-05-06_11-44-46',
    'unet_region/runs/Dataset20_2019-05-06_12-39-18',
    'unet_region/runs/Dataset30_2019-05-06_09-59-33'
]

def test_fn(x, y, z):
    print('hihi')


for i in range(len(in_dirs)):
    for j in range(len(in_dirs[i])):

        # with pascal trained
        os.system('python eval.py \
        --run-dir {}/unet_region/runs/2019-04-10_11-38-11 \
        --in-dir {}/{} \
        --cuda \
        --csv-loc-file {}/{}'.format(root_dir,
                                     root_dir,
                                     in_dirs[i][j],
                                     root_dir,
                                     pjoin(in_dirs[i][j],
                                           'gaze-measurements',
                                           'video1.csv')
    ))

        # with self trained
        os.system('python eval.py \
        --run-dir {}/{} \
        --in-dir {}/{} \
        --cuda \
        --csv-loc-file {}/{}'.format(root_dir,
                                     run_dirs[i],
                                     root_dir,
                                     in_dirs[i][j],
                                     root_dir,
                                     pjoin(in_dirs[i][j],
                                           'gaze-measurements',
                                           'video1.csv')
    ))
