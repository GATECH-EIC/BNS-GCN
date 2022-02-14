import time
import torch.distributed as dist
from contextlib import contextmanager


class CommTimer(object):

    def __init__(self):
        super(CommTimer, self).__init__()
        self._time = {}

    @contextmanager
    def timer(self, name):
        if name in self._time:
            raise Exception(name + " already exists")
        t0 = time.time()
        yield
        t1 = time.time()
        self._time[name] = (t0, t1)

    def tot_time(self):
        tot = 0
        for (t0, t1) in self._time.values():
            tot += t1 - t0
        return tot

    def print_time(self):
        rank, size = dist.get_rank(), dist.get_world_size()
        for (k, (t0, t1)) in self._time.items():
            print(f'(rank {rank}) Communication time of {k}: {t1 - t0} seconds.')

    def clear(self):
        self._time = {}
