import os

import scipy
import dgl
from dgl.data import RedditDataset, YelpDataset
from dgl.distributed import partition_graph
from helper.context import *
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import json


class TransferTag:
    NODE = 0
    FEAT = 1
    DEG = 2


def load_ogb_dataset(name, data_path):
    dataset = DglNodePropPredDataset(name=name, root=data_path)
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    n_node = g.num_nodes()
    node_data = g.ndata
    node_data['label'] = label.view(-1).long()
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    return g


def load_data(args):
    if args.dataset == 'reddit':
        data = RedditDataset(raw_dir=args.data_path)
        g = data[0]
    elif args.dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products', args.data_path)
    elif args.dataset == 'ogbn-papers100m':
        g = load_ogb_dataset('ogbn-papers100M', args.data_path)
    elif args.dataset == 'yelp':
        data = YelpDataset(raw_dir=args.data_path)
        g = data[0]
        g.ndata['label'] = g.ndata['label'].float()
        # TODO: remove the following three lines later (see Issue #4806 of DGL).
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, n_feat, n_class


def graph_partition(args):

    g, n_feat, n_class = load_data(args)
    if args.inductive:
        g = g.subgraph(g.ndata['train_mask'])

    n_class = n_class
    n_feat = n_feat
    n_train = g.ndata['train_mask'].int().sum().item()

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')

    # TODO: after being saved, a bool tensor becomes a uint8 tensor (including 'inner_node')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_deg'] = g.in_degrees()
            g.ndata['out_deg'] = g.out_degrees()
            partition_graph(g, args.graph_name, args.n_partitions, graph_dir,  part_method=args.partition_method,
                            balance_edges=False, objtype=args.partition_obj)
    
    with open(os.path.join(graph_dir, 'meta.json'), 'w') as f:
        json.dump({'n_feat': n_feat, 'n_class': n_class, 'n_train': n_train}, f)


def load_partition(args, rank):

    graph_dir = os.path.join(args.part_path, args.graph_name)
    part_config = os.path.join(graph_dir, args.graph_name + '.json')

    print('loading partitions')

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_deg'] = node_feat[node_type + '/in_deg']
    node_feat['out_deg'] = node_feat[node_type + '/out_deg']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_deg')
    node_feat.pop(node_type + '/out_deg')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')
    subg.ndata.clear()
    subg.edata.clear()

    with open(os.path.join(graph_dir, 'meta.json'), 'r') as f:
        meta = json.load(f)
        args.n_feat = meta['n_feat']
        args.n_class = meta['n_class']
        args.n_train = meta['n_train']

    return subg, node_feat, gpb


def get_layer_size(n_feat, n_hidden, n_class, n_layers):
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)
    return layer_size


def get_boundary(node_dict, gpb):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()

    return boundary


_send_cpu, _recv_cpu = {}, {}


def data_transfer(data, recv_shape, tag, dtype=torch.float):

    rank, size = dist.get_rank(), dist.get_world_size()
    msg, res = [None] * size, [None] * size

    for i in range(1, size):
        idx = (rank + i) % size
        key = 'dst%d_tag%d' % (idx, tag)
        if key not in _recv_cpu:
            _send_cpu[key] = torch.zeros_like(data[idx], dtype=dtype, device='cpu', pin_memory=True)
            _recv_cpu[key] = torch.zeros(recv_shape[idx], dtype=dtype, pin_memory=True)
        msg[idx] = _send_cpu[key]
        res[idx] = _recv_cpu[key]

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        msg[right].copy_(data[right])
        req = dist.isend(msg[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda(non_blocking=True)
        req.wait()

    return res


def merge_feature(feat, recv):
    size = len(recv)
    for i in range(size - 1, 0, -1):
        if recv[i] is None:
            recv[i] = recv[i - 1]
            recv[i - 1] = None
    recv[0] = feat
    return torch.cat(recv)


def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test


def minus_one_tensor(size, device=None):
    if device is not None:
        return torch.zeros(size, dtype=torch.long, device=device) - 1
    else:
        return torch.zeros(size, dtype=torch.long) - 1


def nonzero_idx(x):
    return torch.nonzero(x, as_tuple=True)[0]


def print_memory(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.synchronize()
    print('(rank %d) ' % rank + s +
          ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(torch.cuda.memory_allocated() / 1024 / 1024,
                                                                   torch.cuda.max_memory_allocated() / 1024 / 1024,
                                                                   torch.cuda.memory_reserved() / 1024 / 1024))


@contextmanager
def timer(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    t = time.time()
    yield
    print('(rank %d) running time of %s: %.3f seconds' % (rank, s, time.time() - t))
