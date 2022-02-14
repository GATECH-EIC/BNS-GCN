import torch
from helper.timer.timer import *
import queue


class Buffer(object):

    def __init__(self):
        super(Buffer, self).__init__()
        self._num_in = None
        self._selected = []
        self._n_layers = 0
        self._layer_size = []
        self._feat_cpu, self._grad_cpu = None, None
        self._ratio = []
        self._f_recv, self._b_recv = None, None
        self._f_recv_cpu, self._b_recv_cpu = None, None
        self._recv_shape = []
        self._backend = None
        self._pl, self._pr = [], []
        self._send_que = queue.Queue()

    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = self._num_in
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)

    def init_buffer(self, num_in, ratio, f_send_shape, f_recv_shape, layer_size, use_pp=False, backend='gloo'):
        if use_pp is False:
            raise NotImplementedError
        rank, size = dist.get_rank(), dist.get_world_size()
        self._num_in = num_in
        self._n_layers = len(layer_size)
        self._layer_size = layer_size
        self._recv_shape = f_recv_shape
        self._ratio = ratio

        if backend == 'gloo':
            tmp1, tmp2, tmp3, tmp4 = [], [], [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                    tmp3.append(None)
                    tmp4.append(None)
                else:
                    s1 = torch.Size([f_send_shape[j], self._layer_size[1]])
                    s2 = torch.Size([f_recv_shape[j], self._layer_size[1]])
                    tmp1.append(torch.zeros(s1, pin_memory=True))
                    tmp2.append(torch.zeros(s2, pin_memory=True))
                    tmp3.append(torch.zeros(s2, pin_memory=True))
                    tmp4.append(torch.zeros(s1, pin_memory=True))
            self._feat_cpu = tmp1
            self._f_recv_cpu = tmp3
            self._grad_cpu = tmp2
            self._b_recv_cpu = tmp4

        self._backend = backend

        tmp1, tmp2 = [], []
        for j in range(size):
            if j == rank:
                tmp1.append(None)
                tmp2.append(None)
            else:
                s1 = torch.Size([f_recv_shape[j], self._layer_size[1]])
                s2 = torch.Size([f_send_shape[j], self._layer_size[1]])
                tmp1.append(torch.zeros(s1, device='cuda'))
                tmp2.append(torch.zeros(s2, device='cuda'))
        self._f_recv = tmp1
        if self._backend == 'mpi':
            self._b_recv = tmp2
        self.__init_pl_pr()

    def set_selected(self, selected):
        self._selected = selected

    def __feat_concat(self, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                tmp.append(self._f_recv[i])
        return torch.cat(tmp)

    def update(self, layer, feat):
        with comm_timer.timer(f'forward_{layer}'):
            self.__feat_transfer(feat)
        res = self.__feat_concat(feat)
        if res.requires_grad:
            res.register_hook(self.__grad_hook(layer))
        return res

    @torch.no_grad()
    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, forward=True):

        rank, size = dist.get_rank(), dist.get_world_size()
        req = queue.Queue()

        while not self._send_que.empty():
            r = self._send_que.get()
            r.wait()

        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size
            r2 = dist.irecv(recv_cpu[left], src=left)
            req.put((r2, left))
            if forward:
                send_cpu[right].copy_(send_gpu[self._selected[right]] / self._ratio[right])
            else:
                send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]])
            r1 = dist.isend(send_cpu[right], dst=right)
            self._send_que.put(r1)
        while not req.empty():
            r, idx = req.get()
            # TODO: if not r.is_completed() run next r first (see issue #30723 of PyTorch)
            r.wait()
            if forward:
                recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
            else:
                send_gpu[self._selected[idx]] += recv_cpu[idx].cuda(non_blocking=True) / self._ratio[idx]

    @torch.no_grad()
    def __mpi_all_to_all(self, send, recv, forward=True):
        rank, size = dist.get_rank(), dist.get_world_size()

        send_list, recv_list = [], []

        for i in range(size):
            if i == rank:
                send_list.append(torch.empty(0))
                recv_list.append(torch.empty(0))
                continue
            if forward:
                send_list.append(send[self._selected[i]] / self._ratio[i])
            else:
                send_list.append(send[self._pl[i]:self._pr[i]])
            recv_list.append(recv[i])

        dist.all_to_all(recv_list, send_list)

        if not forward:
            for i in range(size):
                if i != rank:
                    send[self._selected[i]] += recv[i]

    def __feat_transfer(self, feat):
        if self._backend == 'gloo':
            self.__gloo_all_to_all(feat, self._feat_cpu, self._f_recv_cpu, self._f_recv, forward=True)
        elif self._backend == 'mpi':
            self.__mpi_all_to_all(feat, self._f_recv, forward=True)
        else:
            raise NotImplementedError

    def __update_grad(self, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i != rank:
                grad[self._selected[i]] += self._b_recv[i] / self._ratio[i]

    def __grad_hook(self, layer):
        def fn(grad):
            with comm_timer.timer(f'backward_{layer}'):
                self.__grad_transfer(grad)
            return grad
        return fn

    def __grad_transfer(self, grad):
        if self._backend == 'gloo':
            self.__gloo_all_to_all(grad, self._grad_cpu, self._b_recv_cpu, None, forward=False)
        elif self._backend == 'mpi':
            self.__mpi_all_to_all(grad, self._b_recv, forward=False)
        else:
            raise NotImplementedError
