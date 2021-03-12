import torch
import torch.nn as nn


class Corrector(nn.Module):

    def __init__(self, x_dim, h_dim, mlp_h_dim=32):
        super(Corrector, self).__init__()
        self.h2h = nn.Sequential(
            nn.Linear(h_dim + x_dim + 3, mlp_h_dim),
            nn.ReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )

    def forward(self, g, h, x):
        with g.local_scope():
            inp = torch.cat([h, x, g.nodes['tc'].data['position']], dim=-1)
            return self.h2h(inp)


class LinearPreCOCorrector(Corrector):

    def __init__(self, x_dim, h_dim):
        super(LinearPreCOCorrector, self).__init__(x_dim, h_dim)
        self.h2h = nn.Linear(h_dim + x_dim + 3, h_dim)
