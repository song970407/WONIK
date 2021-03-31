import dgl.function as fn
import torch
import torch.nn as nn

from src.graph_config import u2t, t2t
from src.nn.GraphModules.ConvexModule import PartialConvexLinear
from src.nn.ReparameterizedLinear import ReparameterizedLinear


class Predictor(nn.Module):

    def __init__(self, u_dim, h_dim, mlp_h_dim=32):
        super(Predictor, self).__init__()

        self.u2h_enc = nn.Sequential(
            nn.Linear(u_dim + 6, mlp_h_dim),
            nn.ReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )

        self.h2h_enc = nn.Sequential(
            nn.Linear(h_dim + 6, mlp_h_dim),
            nn.ReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )

        self.h_updater = nn.Sequential(
            nn.Linear(h_dim * 3, mlp_h_dim),
            nn.ReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )

    def forward(self, g, h, u):
        with g.local_scope():
            g.nodes['control'].data['u'] = u
            g.update_all(self.u2t_msg, fn.sum('u2h_msg', 'q'), etype=u2t)

            g.nodes['tc'].data['h'] = h
            g.update_all(self.t2t_msg, fn.sum('h2h_msg', 'sum_h'), etype=t2t)

            inp = torch.cat([g.nodes['tc'].data['h'],
                             g.nodes['tc'].data['q'],
                             g.nodes['tc'].data['sum_h']], dim=-1)
            return self.h_updater(inp)

    def u2t_msg(self, edges):
        src_pos = edges.src['position']
        dst_pos = edges.dst['position']
        u = edges.src['u']
        inp = torch.cat([src_pos, dst_pos, u], dim=-1)
        return {'u2h_msg': self.u2h_enc(inp)}

    def t2t_msg(self, edges):
        src_pos = edges.src['position']
        dst_pos = edges.dst['position']
        h = edges.src['h']
        inp = torch.cat([src_pos, dst_pos, h], dim=-1)
        return {'h2h_msg': self.h2h_enc(inp)}


class ConvexPredictor(Predictor):

    def __init__(self, u_dim, h_dim, mlp_h_dim=32):
        super(ConvexPredictor, self).__init__(u_dim, h_dim, mlp_h_dim)

        self.u2h_enc = nn.Sequential(
            PartialConvexLinear(3 * 2, u_dim, mlp_h_dim, mlp_h_dim, is_increasing=False),
            nn.LeakyReLU(negative_slope=0.2),
            PartialConvexLinear(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.h2h_enc = nn.Sequential(
            PartialConvexLinear(3 * 2, h_dim, mlp_h_dim, mlp_h_dim, is_increasing=False),
            nn.LeakyReLU(negative_slope=0.2),
            PartialConvexLinear(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.h_updater = nn.Sequential(
            ReparameterizedLinear(h_dim * 3, mlp_h_dim),
            nn.LeakyReLU(negative_slope=0.2),
            ReparameterizedLinear(mlp_h_dim, h_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )


class LinearPredictor(Predictor):

    def __init__(self, u_dim, h_dim, mlp_h_dim=32):
        super(LinearPredictor, self).__init__(u_dim, h_dim, mlp_h_dim)

        self.u2h_enc = nn.Sequential(
            ReparameterizedLinear(u_dim + 6, mlp_h_dim, reparam_method='Softmax'),
            ReparameterizedLinear(mlp_h_dim, h_dim, reparam_method='Softmax'),
        )

        self.h2h_enc = nn.Sequential(
            ReparameterizedLinear(h_dim + 6, mlp_h_dim, reparam_method='Softmax'),
            ReparameterizedLinear(mlp_h_dim, h_dim, reparam_method='Softmax'),
        )

        self.h_updater = nn.Sequential(
            nn.Linear(h_dim * 3, mlp_h_dim),
            nn.Linear(mlp_h_dim, h_dim),
        )


class LinearPreCOPredictor(Predictor):

    def __init__(self, u_dim, h_dim):
        super(LinearPreCOPredictor, self).__init__(u_dim, h_dim)
        self.u2h_enc = nn.Linear(u_dim + 6, h_dim)
        self.h2h_enc = nn.Linear(h_dim + 6, h_dim)
        self.h_updater = nn.Linear(h_dim * 3, h_dim)
