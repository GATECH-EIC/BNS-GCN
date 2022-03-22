from helper.parser import *
import random
import torch.multiprocessing as mp
import sys
import subprocess
from helper.utils import *
import train
import warnings

if __name__ == '__main__':

    args = create_parser()
    if args.fix_seed is False:
        if args.parts_per_node < args.n_partitions:
            warnings.warn('Please enable `--fix-seed` for multi-node training.')
        args.seed = random.randint(0, 1 << 31)

    if args.graph_name == '':
        if args.inductive:
            args.graph_name = '%s-%d-%s-%s-induc' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)
        else:
            args.graph_name = '%s-%d-%s-%s-trans' % (args.dataset, args.n_partitions,
                                                     args.partition_method, args.partition_obj)

    if args.skip_partition:
        if args.n_feat == 0 or args.n_class == 0 or args.n_train == 0:
            warnings.warn('Specifying `--n-feat`, `--n-class` and `--n-train` saves data loading time.')
            g, n_feat, n_class = load_data(args.dataset)
            args.n_feat = n_feat
            args.n_class = n_class
            args.n_train = g.ndata['train_mask'].int().sum().item()
    else:
        g, n_feat, n_class = load_data(args.dataset)
        if args.node_rank == 0:
            if args.inductive:
                graph_partition(g.subgraph(g.ndata['train_mask']), args)
            else:
                graph_partition(g, args)
        args.n_class = n_class
        args.n_feat = n_feat
        args.n_train = g.ndata['train_mask'].int().sum().item()

    print(args)

    if args.backend == 'gloo':
        processes = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            n = torch.cuda.device_count()
            devices = [f'{i}' for i in range(n)]
        mp.set_start_method('spawn', force=True)
        start_id = args.node_rank * args.parts_per_node
        for i in range(start_id, min(start_id + args.parts_per_node, args.n_partitions)):
            os.environ['CUDA_VISIBLE_DEVICES'] = devices[i % len(devices)]
            p = mp.Process(target=train.init_processes, args=(i, args.n_partitions, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif args.backend == 'mpi':
        gcn_arg = []
        for k, v in vars(args).items():
            if v is True:
                gcn_arg.append(f'--{k}')
            elif v is not False:
                gcn_arg.extend([f'--{k}', f'{v}'])
        mpi_arg = []
        mpi_arg.extend(['-n', f'{args.n_partitions}'])
        command = ['mpirun'] + mpi_arg + ['python', 'train.py'] + gcn_arg
        print(' '.join(command))
        subprocess.run(command, stderr=sys.stderr, stdout=sys.stdout)
    else:
        raise ValueError
