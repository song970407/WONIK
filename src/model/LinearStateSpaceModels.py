import torch
import torch.nn as nn

from src.nn.LinearModules.Hypernet import HyperMatrix, ContextHyperMatrix
from src.nn.LinearModules.MatrixMultiplication import MatrixMultiplication
from src.nn.ReparameterizedLinear import ReparameterizedLinear


def rollout(x0, us, model, state_dim: int, action_dim: int):
    """
    :param x0: [batch_size x num states]
    :param us: [batch_size x time x num actions]
    :return:
    """
    assert x0.shape[0] == us.shape[0], "x0 and us batch size must be equal!"
    assert x0.shape[1] == state_dim
    assert us.shape[2] == action_dim

    xs = []  # predicted states
    x = x0
    for u in us.transpose(1, 0):  # looping over time
        x = model(x, u)
        xs.append(x)

    xs = torch.stack(xs)  # [time x batch_size x num states]
    xs = xs.transpose(0, 1)  # [batch x time x num states]
    return xs


class LinearSSM(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 adj_xx: torch.Tensor = None,
                 adj_xu: torch.Tensor = None):
        super(LinearSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.A = MatrixMultiplication(state_dim, state_dim, adj_xx)
        self.B = MatrixMultiplication(action_dim, state_dim, adj_xu)

    def forward(self, x, u):
        return self.A(x) + self.B(u)

    def rollout(self, x0, us):
        xs = rollout(x0, us, self, self.state_dim, self.action_dim)
        return xs


class HyperLinearSSM(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 a_net: nn.Module,
                 b_net: nn.Module,
                 adj_xx: torch.Tensor = None,
                 adj_xu: torch.Tensor = None):
        super(HyperLinearSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.A = HyperMatrix(self.state_dim, a_net, adj_xx)
        self.B = HyperMatrix(self.state_dim, b_net, adj_xu)

    def forward(self, x, u, zx=None, zu=None):
        if zx is None:
            zx = x
        if zu is None:
            zu = u
        return self.A(x, zx) + self.B(u, zu)

    def rollout(self, x0, us):
        xs = rollout(x0, us, self, self.state_dim, self.action_dim)
        return xs


class TimeVariantHyperLinearSSM(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 z_max: int,
                 a_net: nn.Module,
                 b_net: nn.Module,
                 adj_xx: torch.Tensor = None,
                 adj_xu: torch.Tensor = None):
        super(TimeVariantHyperLinearSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_max = z_max
        self.A = HyperMatrix(self.state_dim, a_net, adj_xx)
        self.B = HyperMatrix(self.state_dim, b_net, adj_xu)

    def forward(self, x, u, context):
        return self.A(x, context) + self.B(u, context)

    def rollout(self, x0, context, us):
        xs = []
        x = x0
        num_rollout = 0
        for u in us.transpose(1, 0):  # looping over time
            x = self(x, u, context + num_rollout / self.z_max)
            xs.append(x)
            num_rollout += 1

        xs = torch.stack(xs)  # [time x batch_size x num states]
        xs = xs.transpose(0, 1)  # [batch x time x num states]

        return xs


class ContextHyperLinearSSM(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_context: int,
                 context_map: torch.Tensor,
                 adj_xx: torch.Tensor = None,
                 adj_xu: torch.Tensor = None
                 ):
        super(ContextHyperLinearSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_context = num_context
        self.context_map = context_map
        self.A = ContextHyperMatrix(self.state_dim, self.state_dim, self.num_context, adj_xx)
        self.B = ContextHyperMatrix(self.action_dim, self.state_dim, self.num_context, adj_xu)

    def forward(self, x, u, context):
        """
        :param x: [batch_size * num_states]
        :param u: [batch_size * num_actions]
        :param context: [batch_size * 1]
        :return: [batch_size * num_states]
        """
        return self.A(x, context) + self.B(u, context)

    def rollout(self, x0, context, us):
        xs = []
        x = x0
        num_rollout = 0
        for u in us.transpose(1, 0):  # looping over time
            x = self(x, u, self.context_map[context + num_rollout])
            xs.append(x)
            num_rollout += 1

        xs = torch.stack(xs)  # [time x batch_size x num states]
        xs = xs.transpose(0, 1)  # [batch x time x num states]

        return xs


class MultiLinearSSM(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 state_order: int,
                 action_order: int):
        super(MultiLinearSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_order = state_order
        self.action_order = action_order
        self.A_list = nn.ModuleList([nn.Linear(state_dim, state_dim) for _ in range(state_order)])
        self.B_list = nn.ModuleList([nn.Linear(action_dim, state_dim) for _ in range(action_order)])

    def forward(self, x, u):
        """
        :param x: B x state_order x state_dim
        :param u: B x action_order x action_dim
        :return:
        """
        res = 0
        for i, l in enumerate(self.A_list):
            res += l(x[:, i, :])
        for j, l in enumerate(self.B_list):
            res += l(u[:, j, :])
        return res

    def rollout(self, x0, u0, us):
        """
        :param x0: B x state_order x state_dim
        :param u0: B x (action_order-1) x action_dim
        :param us: B x H x action_dim
        :return:
        """
        xs = []
        if u0 is not None:
            u_cat = torch.cat([u0, us], dim=1)
        else:
            u_cat = us
        for i in range(us.shape[1]):
            x = self.forward(x0, u_cat[:, i:i + self.action_order]).unsqueeze(dim=1)
            xs.append(x)
            x0 = torch.cat([x0[:, 1:, :], x], dim=1)
        return torch.cat(xs, dim=1)

    def multi_step_prediction(self, x0, u0, us):
        """
        :param x0: state_order x state_dim
        :param u0: (action_order-1) x action_dim
        :param us: H x action_dim
        :return:
        """
        x0 = x0.unsqueeze(dim=0)
        u0 = u0.unsqueeze(dim=0)
        us = us.unsqueeze(dim=0)
        xs = self.rollout(x0, u0, us)
        return xs.squeeze(dim=0)


class ReparamMultiLinearSSM(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 state_order: int,
                 action_order: int):
        super(ReparamMultiLinearSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_order = state_order
        self.action_order = action_order
        self.A_list = nn.ModuleList(
            [ReparameterizedLinear(state_dim, state_dim, reparam_method='Expo') for _ in range(state_order)])
        self.B_list = nn.ModuleList(
            [ReparameterizedLinear(action_dim, state_dim, reparam_method='Expo') for _ in range(action_order)])

    def forward(self, x, u):
        """
        :param x: B x state_order x state_dim
        :param u: B x action_order x action_dim
        :return:
        """
        res = 0
        for i, l in enumerate(self.A_list):
            res += l(x[:, i, :])
        for j, l in enumerate(self.B_list):
            res += l(u[:, j, :])
        return res

    def rollout(self, x0, u0, us):
        """
        :param x0: B x state_order x state_dim
        :param u0: B x (action_order-1) x action_dim
        :param us: B x H x action_dim
        :return:
        """
        xs = []
        if u0 is not None:
            u_cat = torch.cat([u0, us], dim=1)
        else:
            u_cat = us
        for i in range(us.shape[1]):
            x = self.forward(x0, u_cat[:, i:i + self.action_order]).unsqueeze(dim=1)
            xs.append(x)
            x0 = torch.cat([x0[:, 1:, :], x], dim=1)
        return torch.cat(xs, dim=1)

    def multi_step_prediction(self, x0, u0, us):
        """
        :param x0: state_order x state_dim
        :param u0: (action_order-1) x action_dim
        :param us: H x action_dim
        :return:
        """
        x0 = x0.unsqueeze(dim=0)
        u0 = u0.unsqueeze(dim=0)
        us = us.unsqueeze(dim=0)
        xs = self.rollout(x0, u0, us)
        return xs.squeeze(dim=0)
