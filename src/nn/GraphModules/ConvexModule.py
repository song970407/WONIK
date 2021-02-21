from typing import Union, List

import torch
import torch.nn as nn

import src.nn.ConvexActivation as ConvexActivation
from src.nn.GraphModules.GraphConv import GraphConv, GATConv
from src.nn.ReparameterizedLinear import ReparameterizedLinear


def get_nn_activation(activation: str,
                      is_convex: bool,
                      **kwargs):  # NEED TO FIX
    return getattr(ConvexActivation, activation)(is_convex=is_convex, **kwargs)


class ConvexLinear(nn.Module):
    def __init__(self,
                 in_features: Union[int, List[int]],
                 out_features: int,
                 biases: Union[bool, List[bool]] = True,
                 is_increasings: Union[bool, List[bool]] = True,
                 reparam_methods: Union[str, List[str]] = 'ReLU',
                 is_convex: bool = True,
                 activation: str = 'LeakyReLU',
                 negative_slope: int = None):

        super(ConvexLinear, self).__init__()
        if isinstance(in_features, int):
            self.in_features = [in_features]
        else:
            self.in_features = in_features

        self.out_features = out_features

        if isinstance(biases, bool):
            self.biases = [biases] * len(self.in_features)
        else:
            self.biases = biases
            assert len(self.in_features) == len(self.biases), "Lengths of in_features and biases are different"

        if isinstance(is_increasings, bool):
            self.is_increasings = [is_increasings] * len(self.in_features)
        else:
            self.is_increasings = is_increasings
            assert len(self.in_features) == len(
                self.is_increasings), "Lengths of in_features and is_increasings are different"

        if isinstance(reparam_methods, str):
            self.reparam_methods = [reparam_methods] * len(self.in_features)
        else:
            self.reparam_methods = reparam_methods
            assert len(self.in_features) == len(
                self.reparam_methods), "Lengths of in_features and reparameterization_methods are different"

        self.activation = get_nn_activation(activation, is_convex, negative_slope=negative_slope)

        self.layers = nn.ModuleList()
        iter = zip(self.in_features, self.biases, self.is_increasings, self.reparam_methods)
        for in_feature, bias, is_increasing, reparam_method in iter:
            l = ReparameterizedLinear(in_feature, out_features, bias, is_increasing, reparam_method)
            self.layers.append(l)

    def forward(self, inputs):
        """
        :param inputs: List[torch.Tensor]
        :return: B x out_features
        """
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        assert len(self.layers) == len(inputs), "Lengths of in_features and inputs are different!"
        res = 0
        for input, layer in zip(inputs, self.layers):
            res += layer(input)
        return self.activation(res)


class PartialConvexLinear(nn.Module):
    def __init__(self,
                 in_nic_features: int,
                 in_ic_features: int,
                 out_nic_features: int,
                 out_ic_features: int,
                 bias: bool = True,
                 is_increasing: bool = True,
                 reparam_method: str = 'ReLU',
                 is_end: bool = False):
        """
        Args:
            in_nic_features:
            in_ic_features:
            out_nic_features:
            out_ic_features:
            bias:
            is_increasing:
            reparam_method:
            is_end:
        """

        super(PartialConvexLinear, self).__init__()
        self.in_nic_features = in_nic_features
        self.in_ic_features = in_ic_features
        self.out_nic_features = out_nic_features
        self.out_ic_features = out_ic_features
        if not is_end:
            self.nic_nic_layer = nn.Linear(in_nic_features, out_nic_features)
        self.nic_ic_layer = nn.Linear(in_nic_features, out_ic_features)
        self.ic_ic_layer = ReparameterizedLinear(in_features=in_ic_features,
                                                 out_features=out_ic_features,
                                                 bias=bias,
                                                 is_increasing=is_increasing,
                                                 reparam_method=reparam_method)
        self.is_end = is_end

    def forward(self, input):
        """
        :param inputs: torch.Tensor, B x (in_features[0]+in_features[1])
        :return: B x (out_features[0]+out_features[1])
        """
        nic_input, ic_input = torch.split(input, [self.in_nic_features, self.in_ic_features], dim=1)
        if self.is_end:
            return self.nic_ic_layer(nic_input) + self.ic_ic_layer(ic_input)
        else:
            return torch.cat([self.nic_nic_layer(nic_input), self.nic_ic_layer(nic_input) + self.ic_ic_layer(ic_input)],
                             dim=1)


class ConvexGraphConv(nn.Module):
    def __init__(self,
                 in_feats: Union[int, List[int]],
                 out_feats: int,
                 norms: Union[str, List[str]] = 'both',
                 weights: Union[bool, List[bool]] = True,
                 biases: Union[bool, List[bool]] = True,
                 allow_zero_in_degrees: Union[bool, List[bool]] = False,
                 is_increasings: Union[bool, List[bool]] = True,
                 reparam_methods: Union[str, List[str]] = 'ReLU',
                 is_convex: bool = True,
                 activation: str = 'LeakyReLU',
                 negative_slope: int = None):
        super(ConvexGraphConv, self).__init__()

        if isinstance(in_feats, int):
            self.in_feats = [in_feats]
        else:
            self.in_feats = in_feats

        self.out_feats = out_feats

        if isinstance(norms, str):
            self.norms = [norms] * len(self.in_feats)
        else:
            self.norms = norms
            assert len(self.in_feats) == len(self.norms), "Lengths of in_feats and norms are different"

        if isinstance(weights, bool):
            self.weights = [weights] * len(self.in_feats)
        else:
            self.weights = weights
            assert len(self.in_feats) == len(self.weights), "Lengths of in_feats and weights are different"

        if isinstance(biases, bool):
            self.biases = [biases] * len(self.in_feats)
        else:
            self.biases = biases
            assert len(self.in_feats) == len(self.biases), "Lengths of in_feats and biases are different"

        if isinstance(allow_zero_in_degrees, bool):
            self.allow_zero_in_degrees = [allow_zero_in_degrees] * len(self.in_feats)
        else:
            self.allow_zero_in_degrees = allow_zero_in_degrees
            assert len(self.in_feats) == len(
                self.allow_zero_in_degrees), "Lengths of in_feats and allow_zero_in_degrees are different"

        if isinstance(is_increasings, bool):
            self.is_increasings = [is_increasings] * len(self.in_feats)
        else:
            self.is_increasings = is_increasings
            assert len(self.in_feats) == len(
                self.is_increasings), "Lengths of in_feats and is_increasings are different"

        if isinstance(reparam_methods, str):
            self.reparam_methods = [reparam_methods] * len(self.in_feats)
        else:
            self.reparam_methods = reparam_methods
            assert len(self.in_feats) == len(
                self.reparam_methods), "Lengths of in_feats and reparameterization_methods are different"

        self.activation = get_nn_activation(activation, is_convex, negative_slope)

        self.layers = nn.ModuleList()
        iter = zip(self.in_feats, self.norms, self.weights, self.biases,
                   self.allow_zero_in_degrees, self.is_increasings, self.reparam_methods)
        for in_feat, norm, weight, bias, allow_zero_in_degree, is_increasing, reparam_method in iter:
            l = GraphConv(in_feat, out_feats, norm, weight, bias,
                          allow_zero_in_degree, is_increasing, reparam_method)
            self.layers.append(l)

    def forward(self, graph, feats):
        """
        :param graph: dgl.graph
        :param feats: Union[torch.Tensor, List[torch.Tensor]]
        :return: torch.Tensor, num_nodes x out_feats
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        assert len(self.layers) == len(feats), "Lengths of in_feats and feats are different!"
        res = 0
        for feat, layer in zip(feats, self.layers):
            res += layer(graph, feat)
        return self.activation(res)


class ConvexGATConv(nn.Module):
    def __init__(self,
                 in_feats: Union[int, List[int]],
                 attn_feats: Union[int, List[int]],
                 out_feats: int,
                 num_heads: Union[int, List[int]],
                 feat_drops: Union[float, List[float]] = 0.,
                 attn_drops: Union[float, List[float]] = 0.,
                 attn_negative_slopes: Union[float, List[float]] = 0.2,
                 allow_zero_in_degrees: Union[bool, List[bool]] = False,
                 is_increasings: Union[bool, List[bool]] = True,
                 reparam_methods: Union[str, List[str]] = 'ReLU',
                 is_convex: bool = True,
                 activation: str = 'LeakyReLU',
                 negative_slope: int = None):
        super(ConvexGATConv, self).__init__()

        if isinstance(in_feats, int):
            self.in_feats = [in_feats]
        else:
            self.in_feats = in_feats

        self.out_feats = out_feats

        if isinstance(attn_feats, int):
            self.attn_feats = [attn_feats] * len(self.in_feats)
        else:
            self.attn_feats = attn_feats
            assert len(self.in_feats) == len(self.attn_feats), "Lengths of in_feats and attn_feats are different"

        if isinstance(num_heads, int):
            self.num_heads = [num_heads] * len(self.in_feats)
        else:
            self.num_heads = num_heads
            assert len(self.in_feats) == len(self.num_heads), "Lengths of in_feats and num_heads are different"

        if isinstance(feat_drops, float):
            self.feat_drops = [feat_drops] * len(self.in_feats)
        else:
            self.feat_drops = feat_drops
            assert len(self.in_feats) == len(self.feat_drops), "Lengths of in_feats and feat_drops are different"

        if isinstance(attn_drops, float):
            self.attn_drops = [attn_drops] * len(self.in_feats)
        else:
            self.attn_drops = attn_drops
            assert len(self.in_feats) == len(self.attn_drops), "Lengths of in_feats and attn_drops are different"

        if isinstance(attn_negative_slopes, float):
            self.attn_negative_slopes = [attn_negative_slopes] * len(self.in_feats)
        else:
            self.attn_negative_slopes = attn_negative_slopes
            assert len(self.in_feats) == len(
                self.attn_negative_slopes), "Lengths of in_feats and attn_negative_slopes are different"

        if isinstance(allow_zero_in_degrees, bool):
            self.allow_zero_in_degrees = [allow_zero_in_degrees] * len(self.in_feats)
        else:
            self.allow_zero_in_degrees = allow_zero_in_degrees
            assert len(self.in_feats) == len(
                self.allow_zero_in_degrees), "Lengths of in_feats and allow_zero_in_degrees are different"

        if isinstance(is_increasings, bool):
            self.is_increasings = [is_increasings] * len(self.in_feats)
        else:
            self.is_increasings = is_increasings
            assert len(self.in_feats) == len(
                self.is_increasings), "Lengths of in_feats and is_increasings are different"

        if isinstance(reparam_methods, str):
            self.reparam_methods = [reparam_methods] * len(self.in_feats)
        else:
            self.reparam_methods = reparam_methods
            assert len(self.in_feats) == len(
                self.reparam_methods), "Lengths of in_feats and reparameterization_methods are different"

        self.activation = get_nn_activation(activation, is_convex, negative_slope=negative_slope)

        self.layers = nn.ModuleList()
        iter = zip(self.in_feats, self.attn_feats, self.num_heads,
                   self.feat_drops, self.attn_drops, self.attn_negative_slopes,
                   self.allow_zero_in_degrees, self.is_increasings, self.reparam_methods)
        for in_feat, attn_feat, num_head, feat_drop, attn_drop, attn_negative_slope, \
            allow_zero_in_degree, is_increasing, reparameterization_method in iter:
            self.layers.append(
                GATConv(in_feats=in_feat, attn_feats=attn_feat, out_feats=out_feats, num_heads=num_head,
                        feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=attn_negative_slope,
                        allow_zero_in_degree=allow_zero_in_degree,
                        is_increasing=is_increasing,
                        reparam_method=reparameterization_method))

    def forward(self, graph, feats, attn_feats):
        """
        :param graph: dgl.graph
        :param feats: Union[torch.Tensor, List[torch.Tensor]]
        :param attn_feats: Union[torch.Tensor, List[torch.Tensor]]
        :return: torch.Tensor, num_nodes x num_heads x out_feats
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        assert len(self.layers) == len(feats), "Lengths of in_feats and feats are different!"
        if isinstance(attn_feats, torch.Tensor):
            attn_feats = [attn_feats] * len(self.layers)
        assert len(self.layers) == len(attn_feats), "Lengths of in_feats and feats are different!"
        res = 0
        for feat, attn_feat, layer in zip(feats, attn_feats, self.layers):
            attn_feat = attn_feat.to(feat.device)
            res += layer(graph, feat, attn_feat)
        return self.activation(res)
