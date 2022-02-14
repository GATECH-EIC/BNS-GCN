from torch.autograd import Function
from torch import nn
import torch
import torch.distributed as dist


class SyncBatchNormFunc(Function):

    @staticmethod
    def forward(ctx, x, weight, bias, whole_size, running_mean, running_var, training, momentum, eps):
        if not training:
            mean = running_mean
            var = running_var
        else:
            sum_x = x.sum(axis=0)
            sum_x2 = (x ** 2).sum(axis=0)
            dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_x2, op=dist.ReduceOp.SUM)
            mean = sum_x / whole_size
            var = (sum_x2 - mean * sum_x) / whole_size
            running_mean.mul_(1 - momentum).add_(mean * momentum)
            running_var.mul_(1 - momentum).add_(var * momentum)
        std = torch.sqrt(var + eps)
        x_hat = (x - mean) / std
        if training:
            ctx.save_for_backward(x_hat, weight, std)
            ctx.whole_size = whole_size
        return x_hat * weight + bias

    @staticmethod
    def backward(ctx, grad):
        x_hat, weight, std = ctx.saved_tensors
        dbias = grad.sum(axis=0)
        dweight = (grad * x_hat).sum(axis=0)
        dist.all_reduce(dbias, op=dist.ReduceOp.SUM)
        dist.all_reduce(dweight, op=dist.ReduceOp.SUM)
        n = ctx.whole_size
        dx = (weight / n) / std * (n * grad - dbias - x_hat * dweight)
        return dx, dweight, dbias, None, None, None, None, None, None


class SyncBatchNorm(nn.Module):
    
    def __init__(self, num_features, whole_size, eps=1e-5, momentum=0.1):
        super(SyncBatchNorm, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.whole_size = whole_size
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return SyncBatchNormFunc.apply(x, self.weight, self.bias, self.whole_size, self.running_mean, self.running_var,
                                       self.training, self.momentum, self.eps)
