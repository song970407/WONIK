import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.base import DGLError
from dgl.nn.pytorch.conv import GraphConv as DGLGraphConv
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn import init

from src.nn.ReparameterizedLinear import ReparameterizedLinear


class GraphConv(DGLGraphConv):
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 norm: str = 'both',
                 weight: bool = True,
                 bias: bool = True,
                 is_increasing: bool = True,
                 reparam_method: str = 'ReLU'):
        super(GraphConv, self).__init__(in_feats=in_feats,
                                        out_feats=out_feats,
                                        norm=norm,
                                        weight=weight,
                                        bias=bias)
        self.is_increasing = is_increasing
        self.reparam_method = reparam_method

        delattr(self, 'weight')
        self._weight = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
        init.xavier_uniform_(self._weight)

    @property
    def weight(self):
        if self.reparam_method == 'ReLU':
            if self.is_increasing:
                ret = F.relu(self._weight)
            else:
                ret = -1 * F.relu(self._weight)
        elif self.reparam_method is None:
            ret = self._weight
        else:
            raise NotImplementedError('{} is not implemented'.format(self.reparam_method))
        return ret


class GATConv(nn.Module):
    def __init__(self,
                 in_feats: int,
                 attn_feats: int,
                 out_feats: int,
                 num_heads: int,
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 allow_zero_in_degree: bool = False,
                 is_increasing: bool = True,
                 reparam_method: str = 'ReLU'):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        self.fc = ReparameterizedLinear(in_features=self._in_src_feats,
                                        out_features=out_feats * num_heads,
                                        bias=False,
                                        is_increasing=is_increasing,
                                        reparam_method=reparam_method)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.fc_attn = nn.Linear(in_features=attn_feats,
                                 out_features=out_feats * num_heads,
                                 bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_uniform_(self.fc.weight)
        else:
            nn.init.xavier_uniform_(self.fc_src.weight)
            nn.init.xavier_uniform_(self.fc_dst.weight)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, attn_feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            h_src = self.feat_drop(feat)
            attn_h_src = self.feat_drop(attn_feat)
            feat_src = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
            attn_feat_src = attn_feat_dst = self.fc_attn(attn_h_src).view(
                -1, self._num_heads, self._out_feats)
            if graph.is_block:
                attn_feat_dst = attn_feat_src[:graph.number_of_dst_nodes()]
            el = (attn_feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (attn_feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(dgl.function.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(dgl.function.u_mul_e('ft', 'a', 'm'),
                             dgl.function.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            return rst
