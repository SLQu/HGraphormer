import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse():
    p = argparse.ArgumentParser("HGraphormer", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--model_name', type=str, default='HGraphormer', help='model name')
    p.add_argument('--data', type=str, default='coauthorship', help='data name (coauthorship/cocitation)')
    p.add_argument('--dataset', type=str, default='cora', help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation)')

    p.add_argument('--nlayer', type=int, default=2, help='number of hidden layers')
    p.add_argument('--nhid', type=int, default=60, help='hidden dimension of HGraphormer')
    p.add_argument('--nhead', type=int, default=3, help='number of attention heads')
    p.add_argument('--d_k', type=int, default=24, help='dimension of key for attention module')
    p.add_argument('--d_v', type=int, default=33, help='dimension of value for attention module')

    p.add_argument('--dropout', type=float, default=0.3, help='dropout probability')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    p.add_argument('--seed', type=int, default=1, help='seed for randomness')
    p.add_argument('--epochs', type=int, default=50, help='number of epochs to train')


    p.add_argument('--nruns', type=int, default=2, help='number of runs for repeated experiments')
    p.add_argument('--split', type=int, default=1, help='choose which train/test split to use')
    p.add_argument('--gpu', type=int, default=0, help='gpu id to use')
    p.add_argument('--out-dir', type=str, default='result', help='output dir')

    p.add_argument('--gamma', type=float, default=0,  help='balance attention matrix and Laplacian matrix, rM + (1-r)L')
    p.add_argument('--residual', type=str2bool, default=True, help='whether use residual layer or not')
    p.add_argument('--snn', type=int, default=5000, help='max node number of mini batch subhypergraph')

    return p.parse_args()











