import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser('Interface for Temporal Graph Representation Learning via Maximal Cliques')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit',
                        choices=['fb_msg', 'sms','contacts','con_dublin', 'wikipedia'],
                        default='wikipedia')
    parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'], help='transductive (t) or inductive (i)')

    # methodology-related hyper-parameters
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--agg', type=str, default='walk', choices=['walk'],
                        help='walk-based flat lstm aggregation')
    parser.add_argument('--walk_pool', type=str, default='attn', choices=['attn', 'sum'], help='how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other walk_ arguments')
    parser.add_argument('--walk_n_head', type=int, default=8, help="number of heads to use for walk attention")
    parser.add_argument('--walk_linear_out', action='store_true', default=False, help="whether to linearly project each node's ")

    parser.add_argument('--attn_agg_method', type=str, default='attn', choices=['attn', 'lstm', 'mean'], help='local aggregation method, we only use the default here')
    parser.add_argument('--attn_mode', type=str, default='prod', choices=['prod', 'map'],
                        help='use dot product attention or mapping based, we only use the default here')
    parser.add_argument('--attn_n_head', type=int, default=2, help='number of heads used in tree-shaped attention layer, we only use the default here')
    parser.add_argument('--time', type=str, default='time', choices=['time', 'pos', 'empty'], help='how to use time information, we only use the default here')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='toleratd margainal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--ngh_cache', action='store_true',
                        help='(currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously calculated to speed up repeated lookup')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--cpu_cores', type=int, default=1, help='number of cpu_cores used for position encoding')
    parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')

    parser.add_argument('--clq_file', type=str, default='', help='clique_file')
    parser.add_argument('--num_walk', type=int, default=10, help='number of the walks generated per node, should be changed according to dataset')
    parser.add_argument('--len_walk', type=int, default=2, help='length of the walks generated per node, should be changed according to dataset')



    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv
