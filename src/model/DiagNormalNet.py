import torch
import torch.nn as nn

from src.nn.ReparameterizedLinear import ReparameterizedLinear


class DiagNormalNet(nn.Module):

    def __init__(self,
                 base_net: nn.Module,
                 last_hidden_dim: int,
                 output_dim: int):
        super(DiagNormalNet, self).__init__()
        self.net = base_net
        self.fc_mu = nn.Linear(last_hidden_dim, output_dim)
        self.fc_var = nn.Linear(last_hidden_dim, output_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return torch.cat([mu, log_var], dim=-1)


class ConvexDiagNormalNet(nn.Module):

    def __init__(self,
                 base_net: nn.Module,
                 last_hidden_dim: int,
                 output_dim: int):
        super(ConvexDiagNormalNet, self).__init__()
        self.net = base_net
        self.fc_mu = ReparameterizedLinear(last_hidden_dim, output_dim)
        self.fc_var = nn.Linear(last_hidden_dim, output_dim)
        self.concave_act = nn.LeakyReLU(negative_slope=1.5)

    def forward(self, x):
        x = self.net(x)
        mu = self.concave_act(self.fc_mu(x))
        log_var = self.fc_var(x)
        return torch.cat([mu, log_var], dim=-1)
