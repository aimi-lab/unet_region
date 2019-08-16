import configargparse
<<<<<<< HEAD
=======
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
>>>>>>> tmp

def get_params():
    p = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=['default.yaml'])

    p.add('-v', help='verbose', action='store_true')

    p.add('--data-type')
    p.add('--frames', action='append', type=int)
    p.add('--n-patches', type=int)
<<<<<<< HEAD
    p.add('--epochs-pretrain', type=int)
    p.add('--lr-decay', type=float)
    p.add('--lr', type=float)
=======
    p.add('--epochs', type=int)
    p.add('--lr-decay', type=float)
    p.add('--lr', type=float)
    p.add('--patience', type=int)
    p.add('--gamma', type=float)
>>>>>>> tmp
    p.add('--momentum', type=float)
    p.add('--eps', type=float)
    p.add('--ds-split', type=float)
    p.add('--ds-shuffle', type=bool)
    p.add('--weight-decay', type=float)
    p.add('--batch-size', type=int)
    p.add('--batch-norm', type=bool)
    p.add('--n-workers', type=int)
    p.add('--seed', type=int)
    p.add('--fake-len', type=int)
    p.add('--cuda', default=False, action='store_true')
    p.add('--save-train-examples', default=True, action='store_true')
    p.add('--n-save-train-examples', type=int)
<<<<<<< HEAD
    p.add('--coordconv', type=bool)
    p.add('--coordconv-r', type=bool)
=======
    p.add_argument("--coordconv", type=str2bool, nargs='?',
                        const=True)
    p.add_argument("--coordconv-r", type=str2bool, nargs='?',
                        const=True)
>>>>>>> tmp
    p.add('--in-shape', type=int)
    p.add('--loss-size', type=float)
    p.add('--loss-lambda', type=float)
    p.add('--patch-rel-size', type=float)
    p.add('--aug-noise', type=float)
    p.add('--aug-flip-proba', type=float)
    p.add('--aug-some', type=int)

    p.add('--fix-radius', type=float)

    p.add('--init-radius', type=float)
<<<<<<< HEAD
=======
    p.add('--n-nodes', type=float)
>>>>>>> tmp
    p.add('--tsdf-thr', type=int)
    p.add('--n-iters', type=int)
    p.add('--alpha', type=float)
    p.add('--beta', type=float)


    return p
