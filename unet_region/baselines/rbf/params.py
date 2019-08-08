import configargparse

def get_params():
    p = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=['default.yaml'])

    p.add('-v', help='verbose', action='store_true')

    p.add('--data-type')
    p.add('--frames', action='append', type=int)
    p.add('--n-patches', type=int)
    p.add('--epochs-pretrain', type=int)
    p.add('--lr-decay', type=float)
    p.add('--lr', type=float)
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
    p.add('--coordconv', type=bool)
    p.add('--coordconv-r', type=bool)
    p.add('--in-shape', type=int)
    p.add('--loss-size', type=float)
    p.add('--loss-lambda', type=float)
    p.add('--patch-rel-size', type=float)
    p.add('--aug-noise', type=float)
    p.add('--aug-flip-proba', type=float)
    p.add('--aug-some', type=int)

    p.add('--fix-radius', type=float)

    p.add('--init-radius', type=float)
    p.add('--tsdf-thr', type=int)
    p.add('--n-iters', type=int)
    p.add('--alpha', type=float)
    p.add('--beta', type=float)


    return p
