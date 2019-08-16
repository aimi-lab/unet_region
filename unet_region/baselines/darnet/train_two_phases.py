from unet_region.baselines.darnet import train
from unet_region.baselines.darnet import params
from os.path import join as pjoin

def main(cfg):

    # train with pascal
<<<<<<< HEAD
    cfg.epochs = 50
    print('Pretraining with PASCAL for {} epochs'.format(cfg.epochs))
    cfg.phase = 'pascal'
    cfg.in_dir = cfg.in_dir_pascal
    cfg = train.main(cfg)
=======
    # cfg.epochs = 50
    # print('Pretraining with PASCAL for {} epochs'.format(cfg.epochs))
    # cfg.phase = 'pascal'
    # cfg.in_dir = cfg.in_dir_pascal
    # cfg = train.main(cfg)
>>>>>>> tmp

    # train without contours
    cfg.epochs = 100
    print('Pretraining with real data for {} epochs'.format(cfg.epochs))
<<<<<<< HEAD
    cfg.checkpoint_path = pjoin(cfg.run_dir, 'checkpoints', 'checkpoint_ls.pth.tar')
=======
    # cfg.checkpoint_path = pjoin(cfg.run_dir, 'checkpoints', 'checkpoint_ls.pth.tar')
>>>>>>> tmp
    cfg.phase = 'data'
    cfg.in_dir = cfg.in_dir_medical
    cfg = train.main(cfg)

    # train with contours
    print('Training contours for {} epochs'.format(cfg.epochs))
    cfg.checkpoint_path = pjoin(cfg.run_dir, 'checkpoints', 'checkpoint_ls.pth.tar')
    cfg.epochs = 100
    cfg = train.main(cfg)


if __name__ == "__main__":

    frames = [15, 52, 13, 16]

    p = params.get_params()

    p.add('--out-dir', required=True)
<<<<<<< HEAD
    p.add('--in-dir-pascal', required=True)
=======
>>>>>>> tmp
    p.add('--in-dir-medical', required=True)
    cfg = p.parse_args()

    # p.add('--out-dir')
    # p.add('--in-dir')
    # p.add('--checkpoint-path',
    #       default=None)
    # p.add('--phase')
    # cfg = p.parse_args()
    # root = '/home/ubelix/data/VOCdevkit/'
    # cfg.n_workers = 0
    # cfg.in_dir = root
    # cfg.out_dir = '/home/ubelix/runs/scratch'
    # cfg.phase = 'pascal'

    # cfg.coordconv = True
    # cfg.coordconv_r = True

    main(cfg)
