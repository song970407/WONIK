import torch
import torch.nn as nn
from src.nn.LinearModules.MatrixMultiplication import MatrixMultiplication


class LocalLinearModel(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 state_diff_order: int,
                 state_ma_order: int,
                 action_diff_order: int,
                 action_ma_order: int):
        super(LocalLinearModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_diff_order = state_diff_order
        self.state_ma_order = state_ma_order
        self.action_diff_order = action_diff_order
        self.action_ma_order = action_ma_order

        self.A = MatrixMultiplication(state_diff_order * state_dim, state_dim)
        self.B = MatrixMultiplication(state_ma_order * state_dim, state_dim)
        self.C = MatrixMultiplication(action_diff_order * action_dim, state_dim)
        self.D = MatrixMultiplication(action_ma_order * action_dim, state_dim)
        self.b = nn.Parameter(torch.ones(state_dim))

    def forward(self, state_diff, state_ma, action_diff, action_ma):
        ret = self.A(state_diff) + self.B(state_ma) + self.C(action_diff) + self.D(action_ma) + self.b
        return ret
