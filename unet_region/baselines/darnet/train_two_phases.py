from unet_region.baselines.darnet import train
from unet_region.baselines.darnet import params
from os.path import join as pjoin

def main(cfg):

    # train without contours
    cfg.epochs = 100
    print('Pretraining with real data for {} epochs'.format(cfg.epochs))
    # cfg.checkpoint_path = pjoin(cfg.run_dir, 'checkpoints', 'checkpoint_ls.pth.tar')
    cfg.phase = 'data'
    cfg.in_dir = cfg.in_dir
    cfg = train.main(cfg)

    # train with contours
    print('Training contours for {} epochs'.format(cfg.epochs))
    cfg.checkpoint_path = pjoin(cfg.run_dir, 'checkpoints', 'checkpoint_ls.pth.tar')
    cfg.phase = 'contours'
    cfg.epochs = 20
    cfg = train.main(cfg)


if __name__ == "__main__":

    frames = [15, 52, 13, 16]

    p = params.get_params()

    p.add('--out-dir', required=True)
    p.add('--in-dir', required=True)
    p.add('--in-dirs-test', required=True)
    p.add('-in-dirs-test', nargs='+', required=True)

    p.add('--checkpoint-path', default=None)
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
