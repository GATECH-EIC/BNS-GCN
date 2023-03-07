from torch import nn
import torch
import math

import dgl.function as fn


class GCNLayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 use_pp=False):
        super(GCNLayer, self).__init__()
        self.use_pp = use_pp
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_norm, out_norm):
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                else:
                    in_norm = in_norm.unsqueeze(1)
                    out_norm = out_norm.unsqueeze(1)
                    graph.nodes['_U'].data['h'] = feat / out_norm
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    feat = self.linear(graph.nodes['_V'].data['h'] / in_norm)
            else:
                in_norm = torch.sqrt(graph.in_degrees()).unsqueeze(1)
                out_norm = torch.sqrt(graph.out_degrees()).unsqueeze(1)
                graph.ndata['h'] = feat / out_norm
                graph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                feat = self.linear(graph.ndata.pop('h') / in_norm)
        return feat


class GraphSAGELayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 use_pp=False):
        super(GraphSAGELayer, self).__init__()
        self.use_pp = use_pp
        if self.use_pp:
            self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        else:
            self.linear1 = nn.Linear(in_feats, out_feats, bias=bias)
            self.linear2 = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_pp:
            stdv = 1. / math.sqrt(self.linear.weight.size(1))
            self.linear.weight.data.uniform_(-stdv, stdv)
            if self.linear.bias is not None:
                self.linear.bias.data.uniform_(-stdv, stdv)
        else:
            stdv = 1. / math.sqrt(self.linear1.weight.size(1))
            self.linear1.weight.data.uniform_(-stdv, stdv)
            self.linear2.weight.data.uniform_(-stdv, stdv)
            if self.linear1.bias is not None:
                self.linear1.bias.data.uniform_(-stdv, stdv)
                self.linear2.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_norm):
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                else:
                    degs = in_norm.unsqueeze(1)
                    num_dst = graph.num_nodes('_V')
                    graph.nodes['_U'].data['h'] = feat
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    ah = graph.nodes['_V'].data['h'] / degs
                    feat = self.linear1(feat[0:num_dst]) + self.linear2(ah)
            else:
                degs = graph.in_degrees().unsqueeze(1)
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                ah = graph.ndata.pop('h') / degs
                if self.use_pp:
                    feat = self.linear(torch.cat((feat, ah), dim=1))
                else:
                    feat = self.linear1(feat) + self.linear2(ah)
        return feat
