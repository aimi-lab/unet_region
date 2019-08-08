import os
from os.path import join as pjoin
from unet_region.baselines.unet import train
from unet_region.baselines.unet import eval
from unet_region.baselines.chan_vese import main as chan_vese
from unet_region.baselines.unet import params

if __name__ == "__main__":
    p = params.get_params()

    p.add('--data-dir', required=True)
    p.add('--out-dir', required=True)
    p.add('--checkpoint-path')

    cfg = p.parse_args()

    in_dirs = {
        'Dataset20': [
            'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24',
            'Dataset25'
        ],
        # 'Dataset20': [
        #     'Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24',
        #     'Dataset25'
        # ],
        'Dataset00': [
            'Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04',
            'Dataset05'
        ],
        'Dataset10': ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'],
        'Dataset30':
        ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33', 'Dataset34']
    }

    frames = [15, 52, 13, 16]

    # pre-train with pascal
    # cfg.data_type = 'pascal'
    # cfg.checkpoint_path = None

    # cfg.epochs = 20

    # print('pretraining with PASCAL')
    # cfg = train.main(cfg)
    # cp_pascal = pjoin(cfg.run_dir,
    #                   'checkpoints',
    #                   'best_model.pth.tar')

    # to_eval = [item for _, sublist in in_dirs.items() for item in sublist]

    # for dir_to_eval in to_eval:
    #     cfg.csv_loc_file = pjoin(cfg.data_dir,
    #                             'medical-labeling',
    #                             dir_to_eval,
    #                             'gaze-measurements',
    #                             'video1.csv')
    #     cfg.in_dir = pjoin(cfg.data_dir, 'medical-labeling', dir_to_eval)
    #     print('evaluating on {}'.format(dir_to_eval))
    #     eval.main(cfg)
    #     cfg.preds_dir = pjoin(cfg.run_dir, dir_to_eval)
    #     print('Running chan-vese on {}'.format(dir_to_eval))
    #     chan_vese(cfg)

    data_dir = cfg.data_dir

    cfg.epochs = 80

    # cfg.epochs = 1
    for i, train_dir in enumerate(in_dirs.keys()):
        # for in_dir, f in zip(in_dirs, frames):
        cfg.data_dir = pjoin(data_dir, 'medical-labeling', train_dir)
        cfg.frames = [frames[i]]
        cfg.data_type = 'medical'

        print('training on {}'.format(cfg.data_dir))
        cfg = train.main(cfg)

        for eval_dir in in_dirs[train_dir]:
            cfg.csv_loc_file = pjoin(data_dir, 'medical-labeling', eval_dir,
                                     'gaze-measurements', 'video1.csv')
            cfg.in_dir = pjoin(data_dir, 'medical-labeling', eval_dir)
            print('evaluating on {}'.format(eval_dir))
            eval.main(cfg)
            cfg.preds_dir = pjoin(cfg.run_dir, eval_dir)
            print('Running chan-vese on {}'.format(eval_dir))
            chan_vese(cfg)
