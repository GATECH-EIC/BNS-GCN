import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='BNS-GCN')
    parser.add_argument("--dataset", type=str, default='reddit',
                        help="the input dataset")
    parser.add_argument("--data-path", "--data_path", type=str, default='./dataset/',
                        help="the storage path of datasets")
    parser.add_argument("--graph-name", "--graph_name", type=str, default='')
    parser.add_argument("--model", type=str, default='graphsage',
                        help="model for training")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--sampling-rate", "--sampling_rate", type=float, default=1,
                        help="the sampling rate of BNS-GCN")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--n-epochs", "--n_epochs", type=int, default=200,
                        help="the number of training epochs")
    parser.add_argument("--n-partitions", "--n_partitions", type=int, default=2,
                        help="the number of partitions")
    parser.add_argument("--n-hidden", "--n_hidden", type=int, default=16,
                        help="the number of hidden units")
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2,
                        help="the number of GCN layers")
    parser.add_argument("--n-feat", "--n_feat", type=int, default=0)
    parser.add_argument("--n-class", "--n_class", type=int, default=0)
    parser.add_argument("--n-train", "--n_train", type=int, default=0)
    parser.add_argument("--log-every", "--log_every", type=int, default=10)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0,
                        help="weight for L2 loss")
    parser.add_argument("--norm", choices=['layer', 'batch'], default='layer',
                        help="normalization method")
    parser.add_argument("--partition-obj", "--partition_obj", choices=['vol', 'cut'], default='vol',
                        help="partition objective function ('vol' or 'cut')")
    parser.add_argument("--partition-method", "--partition_method", choices=['metis', 'random'], default='metis',
                        help="the method for graph partition ('metis' or 'random')")
    parser.add_argument("--n-linear", "--n_linear", type=int, default=0,
                        help="the number of linear layers")
    parser.add_argument("--use-pp", "--use_pp", action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--inductive", action='store_true',
                        help="inductive learning setting")
    parser.add_argument("--fix-seed", "--fix_seed", action='store_true',
                        help="fix random seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", type=str, default='gloo')
    parser.add_argument("--port", type=int, default=18118,
                        help="the network port for communication")
    parser.add_argument("--master-addr", "--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--node-rank", "--node_rank", type=int, default=0)
    parser.add_argument("--parts-per-node", "--parts_per_node", type=int, default=10)
    parser.add_argument('--skip-partition', action='store_true',
                        help="skip graph partition")
    parser.add_argument('--eval', action='store_true',
                        help="enable evaluation")
    parser.add_argument('--no-eval', action='store_false', dest='eval',
                        help="disable evaluation")
    parser.set_defaults(eval=True)
    return parser.parse_args()
