from module.model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score
from helper.parser import create_parser
import os
import torch.nn.functional as F


def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


@torch.no_grad()
def evaluate_induc(name, model, g, mode, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    model.eval()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, result_file_name=None):
    model.eval()
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    val_acc = calc_acc(val_logits, val_labels)
    test_acc = calc_acc(test_logits, test_labels)
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, val_acc


def move_to_cuda(graph, in_graph, out_graph, node_dict, boundary):
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i != rank:
            boundary[i] = boundary[i].cuda()
    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'.format()))
    in_graph = in_graph.int().to(torch.device('cuda'.format()))
    out_graph = out_graph.int().to(torch.device('cuda'.format()))
    return graph, in_graph, out_graph, node_dict, boundary


def get_in_out_graph(graph, node_dict):
    in_graph = dgl.node_subgraph(graph, node_dict['inner_node'].bool())
    in_graph.ndata.clear()
    in_graph.edata.clear()

    out_graph = graph.clone()
    out_graph.ndata.clear()
    out_graph.edata.clear()
    in_nodes = torch.arange(in_graph.num_nodes())
    out_graph.remove_edges(out_graph.out_edges(in_nodes, form='eid'))
    return in_graph, out_graph


def get_pos(node_dict, gpb):
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            part_size = gpb.partid2nids(i).shape[0]
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, 'cuda')
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos


def get_send_size(boundary, prob):
    rank, size = dist.get_rank(), dist.get_world_size()
    res, ratio = [], []
    for i, b in enumerate(boundary):
        if i == rank:
            res.append(0)
            ratio.append(0)
            continue
        s = int(prob * b.shape[0])
        res.append(s)
        # TODO: ratio.append(1 if args.model == 'gat' else s / b.shape[0])
        ratio.append(s / b.shape[0])
    return res, ratio


def get_recv_size(node_dict, prob):
    rank, size = dist.get_rank(), dist.get_world_size()
    res = []
    for i in range(size):
        if i == rank:
            res.append(0)
            continue
        tot = (node_dict['part_id'] == i).int().sum().item()
        res.append(int(prob * tot))
    return res


def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)
    return construct_graph(part, graph, pos, one_hops)


def collect_out_degree(node_dict, boundary):
    rank, size = dist.get_rank(), dist.get_world_size()
    out_deg = node_dict['out_deg']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
            continue
        else:
            send_info.append(out_deg[b])
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
            continue
        else:
            s = (node_dict['part_id'] == i).int().sum()
            recv_shape.append(torch.Size([s]))
    recv_out_deg = data_transfer(send_info, recv_shape, tag=TransferTag.DEG, dtype=torch.long)
    return merge_feature(out_deg, recv_out_deg)


def precompute(part, graph, node_dict, boundary, model, gpb, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    graph = order_graph(part, graph, gpb, node_dict, pos)
    feat = node_dict['feat']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
            continue
        else:
            send_info.append(feat[b])
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
            continue
        else:
            s = (node_dict['part_id'] == i).int().sum()
            recv_shape.append(torch.Size([s, feat.shape[1]]))
    recv_feat = data_transfer(send_info, recv_shape, tag=TransferTag.FEAT, dtype=torch.float)
    if model == 'gcn':
        in_norm = torch.sqrt(node_dict['in_deg'])
        out_norm = torch.sqrt(node_dict['out_deg'])
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph.nodes['_U'].data['h'] /= out_norm.unsqueeze(-1)
            graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                   fn.sum(msg='m', out='h'),
                                   etype='_E')
            return graph.nodes['_V'].data['h'] / in_norm.unsqueeze(-1)
    elif model == 'graphsage':
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                   fn.mean(msg='m', out='h'),
                                   etype='_E')
            mean_feat = graph.nodes['_V'].data['h']
        return torch.cat([feat, mean_feat], dim=1)
    elif model == 'gat':
        return merge_feature(feat, recv_feat)
    else:
        raise Exception


def create_model(layer_size, args):
    if args.model == 'gcn':
        return GCN(layer_size, F.relu, norm=args.norm, use_pp=args.use_pp, dropout=args.dropout,
                   train_size=args.n_train, n_linear=args.n_linear)
    elif args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, norm=args.norm, use_pp=args.use_pp, dropout=args.dropout,
                         train_size=args.n_train, n_linear=args.n_linear)
    elif args.model == 'gat':
        return GAT(layer_size, F.relu, use_pp=True, heads=args.heads, norm=args.norm, dropout=args.dropout)


def select_node(boundary, send_size):
    rank, size = dist.get_rank(), dist.get_world_size()
    selected = []
    for i in range(size):
        if i == rank:
            selected.append(None)
            continue
        b = boundary[i]
        idx = torch.as_tensor(np.random.choice(b.shape[0], send_size[i], replace=False),
                              dtype=torch.long, device='cuda')
        selected.append(b[idx])
    return selected


def reduce_hook(param, name, n_train):
    def fn(grad):
        ctx.reducer.reduce(param, name, grad, n_train)
    return fn


def construct_out_norm(num, norm, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    out_norm_list = [norm[0:num]]
    for i in range(size):
        if i == rank:
            continue
        else:
            out_norm_list.append(norm[pos[i][one_hops[i]]])
    return torch.cat(out_norm_list)


def construct_graph(part, graph, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    tot = part.num_nodes()
    u, v = part.edges()
    u_list, v_list = [u], [v]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')

    return g


def construct_feat(num, feat, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    res = [feat[0:num]]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            res.append(feat[u])

    return torch.cat(res)


def run(graph, node_dict, gpb, args):

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    rank, size = dist.get_rank(), dist.get_world_size()
    in_graph, out_graph = get_in_out_graph(graph, node_dict)

    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/', exist_ok=True)

    if args.eval:
        if args.inductive is False:
            val_g, _, _ = load_data(args)
            test_g = val_g
        else:
            g, _, _ = load_data(args)
            _, val_g, test_g = inductive_split(g)
    else:
        val_g = test_g = None

    boundary = get_boundary(node_dict, gpb)
    layer_size = get_layer_size(args.n_feat, args.n_hidden, args.n_class, args.n_layers)

    graph, in_graph, out_graph, node_dict, boundary = move_to_cuda(graph, in_graph, out_graph, node_dict, boundary)

    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges '
          f'{in_graph.num_nodes()} inner nodes, and {in_graph.num_edges()} inner edges.')

    torch.manual_seed(args.seed)
    model = create_model(layer_size, args)
    model.cuda()

    ctx.reducer.init(model)

    for i, (name, param) in enumerate(model.named_parameters()):
        param.register_hook(reduce_hook(param, name, args.n_train))

    labels = node_dict['label']
    part_train = node_dict['train_mask'].int().sum().item()

    pos = get_pos(node_dict, gpb)
    send_size, ratio = get_send_size(boundary, args.sampling_rate)
    recv_size = get_recv_size(node_dict, args.sampling_rate)

    ctx.buffer.init_buffer(in_graph.num_nodes(), ratio, send_size, recv_size, layer_size[:args.n_layers - args.n_linear],
                           use_pp=args.use_pp, backend=args.backend)

    node_dict['out_deg'] = collect_out_degree(node_dict, boundary)
    if args.use_pp:
        node_dict['feat'] = precompute(in_graph, graph, node_dict, boundary, args.model, gpb, pos)

    best_model, best_acc = None, 0

    result_file_name = 'results/%s_n%d_p%.2f.txt' % (args.dataset, args.n_partitions, args.sampling_rate)

    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_dur, comm_dur, reduce_dur, barr_dur = [], [], [], []
    recv_shape = [torch.Size([s]) for s in recv_size]
    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    print(f'Process {rank} start training')

    feat = node_dict['feat']
    train_mask = node_dict['train_mask']
    if args.model == 'gcn':
        in_norm = torch.sqrt(node_dict['in_deg'])
        out_norm = torch.sqrt(node_dict['out_deg'])
    elif args.model == 'graphsage':
        in_norm = node_dict['in_deg']

    del graph
    del node_dict

    for epoch in range(args.n_epochs):

        t0 = time.time()
        selected = select_node(boundary, send_size)
        one_hops = data_transfer(selected, recv_shape, tag=TransferTag.NODE, dtype=torch.long)
        ctx.buffer.set_selected(selected)

        g = construct_graph(in_graph, out_graph, pos, one_hops)

        model.train()

        if args.model == 'gcn':
            # print(out_norm[:g.num_nodes('_V')].shape, out_norm[one_hops].shape)
            out_norm_ = construct_out_norm(g.num_nodes('_V'), out_norm, pos, one_hops)
            logits = model(g, feat, in_norm, out_norm_)
        elif args.model == 'graphsage':
            logits = model(g, feat, in_norm)
        elif args.model == 'gat':
            logits = model(g, construct_feat(g.num_nodes('_V'), feat, pos, one_hops))
        else:
            raise NotImplementedError

        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        pre_reduce = time.time()
        ctx.reducer.synchronize()
        reduce_time = time.time() - pre_reduce
        optimizer.step()

        if epoch >= 5:
            train_dur.append(time.time() - t0)
            comm_dur.append(comm_timer.tot_time())
            reduce_dur.append(reduce_time)

        if (epoch + 1) % args.log_every == 0:
            print(
                "Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                    rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), loss.item() / part_train))

        comm_timer.clear()

        if args.eval and rank == 0 and (epoch + 1) % args.log_every == 0:
            torch.save(model.state_dict(), 'checkpoint/%s_p%.2f_%d.pth.tar' % (args.graph_name, args.sampling_rate, epoch))
            if thread is not None:
                model_copy, val_acc = thread.get()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy
            model.eval()
            model.cpu()
            if not args.inductive:
                thread = pool.apply_async(evaluate_trans, args=('Epoch %05d' % epoch, copy.deepcopy(model),
                                                                val_g, result_file_name))
            else:
                thread = pool.apply_async(evaluate_induc, args=('Epoch %05d' % epoch, copy.deepcopy(model),
                                                                val_g, 'val', result_file_name))
            model.cuda(rank)

    print_memory("memory stats")

    if args.eval and rank == 0:
        if thread is not None:
            model_copy, val_acc = thread.get()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
        torch.save(best_model.state_dict(), 'checkpoint/' + args.graph_name + '_final.pth.tar')
        print('model saved')
        print("Max Validation Accuracy {:.2%}".format(best_acc))
        best_model.cpu()
        _, acc = evaluate_induc('Test Result', best_model, test_g, 'test')


def init_processes(rank, size, args):
    """ Initialize the distributed environment. """
    if args.backend == 'mpi':
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        torch.cuda.set_device('cuda:%d' % local_rank)
    else:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = '%d' % args.port
    dist.init_process_group(args.backend, rank=rank, world_size=size)
    g, node_dict, gpb = load_partition(args, rank)
    run(g, node_dict, gpb, args)


if __name__ == '__main__':
    args = create_parser()
    init_processes(0, 0, args)
