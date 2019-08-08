import os
from os.path import join as pjoin
import params
import eval
from unet_region.baselines.chan_vese import main as chan_vese

# with coord-conv
# dirs = {
#     'Dataset00_2019-05-29_19-29': [
#         'Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04',
#         'Dataset05'
#     ],

#     'Dataset10_2019-05-29_20-31':
#     ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'],

#     'Dataset20_2019-05-29_21-33': [
#         'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23',
#         'Dataset24',
#         'Dataset25'
#     ],

#     'Dataset30_2019-05-29_22-35':
#     ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33', 'Dataset34']
# }

# pascal_dir = 'pascal_2019-05-29_08-20'

# without coord-conv
dirs = {
    # 'Dataset30_2019-08-06_17-27':
    # ['Dataset31', 'Dataset30', 'Dataset32', 'Dataset33', 'Dataset34'],
    # 'Dataset30_2019-08-06_18-50':
    # ['Dataset31', 'Dataset30', 'Dataset32', 'Dataset33', 'Dataset34']
    # 'Dataset20_2019-08-06_11-46':
    # ['Dataset21', 'Dataset20', 'Dataset22', 'Dataset23', 'Dataset24', 'Dataset25'],
    'Dataset20_2019-08-06_09-55':
    ['Dataset20']
    # ['Dataset21', 'Dataset20', 'Dataset22', 'Dataset23', 'Dataset24', 'Dataset25']
}

def main(cfg):

    for run_dir in dirs.keys():
        for s in dirs[run_dir]:
            cfg.csv_loc_file = pjoin(cfg.data_dir, 'medical-labeling',
                                     s, 'gaze-measurements',
                                     'video1.csv')

            cfg.in_dir = pjoin(cfg.data_dir, 'medical-labeling', s)

            # with self trained
            cfg.run_dir = pjoin(cfg.runs_dir, run_dir)
            eval.main(cfg)

            cfg.preds_dir = pjoin(cfg.run_dir, s)
            cfg.preds_dir
            print('Running chan-vese on {}'.format(cfg.preds_dir))
            chan_vese(cfg)

if __name__ == "__main__":

    p = params.get_params()
    p.add('--runs-dir')
    p.add('--data-dir')
    p.add('--preds-dir')
    cfg = p.parse_args()

    main(cfg)
