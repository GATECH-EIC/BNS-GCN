import torch
import torch.distributed as dist
from multiprocessing.pool import ThreadPool
import time


class Reducer(object):

    def __init__(self):
        super(Reducer, self).__init__()
        self._data_cpu = {}
        self._group = {}
        self._pool = None
        self._handles = []
        self._stream = None

    def init(self, model):
        cnt = 0
        for i, (name, param) in enumerate(model.named_parameters()):
            cnt += 1
            if dist.get_backend() == 'gloo':
                self._data_cpu[name] = torch.zeros_like(param.data, pin_memory=True, device='cpu')
            self._group[name] = dist.new_group()
        if dist.get_backend() == 'gloo':
            self._pool = ThreadPool(processes=cnt)
        self._stream = torch.cuda.Stream(device='cuda')

    def reduce(self, param, name, data, n_train):
        def create_stream():
            if dist.get_backend() == 'mpi':
                torch.cuda.set_device('cuda:%d' % dist.get_rank())
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                data.div_(n_train)
                data_cpu, group = self._data_cpu[name], self._group[name]
                data_cpu.copy_(data)
                dist.all_reduce(data_cpu, op=dist.ReduceOp.SUM, group=group)
                param.grad.copy_(data_cpu, non_blocking=True)

        if dist.get_backend() == 'gloo':
            self._handles.append(self._pool.apply_async(create_stream))
        elif dist.get_backend() == 'mpi':
            group = self._group[name]
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                dist.all_reduce(data, op=dist.ReduceOp.SUM, group=group)
                param.grad = data
        else:
            raise NotImplementedError

    def synchronize(self):
        for handle in self._handles:
            handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)
