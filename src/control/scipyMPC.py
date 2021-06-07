from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize


class MPC(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 state_dim: int,
                 action_dim: int,
                 num_states: int,
                 num_actions: int,
                 H: int,
                 state_ref: Union[np.array, torch.tensor] = None,
                 action_min: Union[float, List[float]] = None,
                 action_max: Union[float, List[float]] = None,
                 Q: Union[np.array, torch.tensor] = None,
                 R: Union[np.array, torch.tensor] = None,
                 r: Union[np.array, torch.tensor] = None,
                 is_convex: bool = False,
                 device='cpu'):
        """
        :param model: an instance of pytorch nn.module.
        model must have rollout function
        :param state_dim: dimension of state
        :param action_dim: dimension of action
        :param H: receding horizon
        :param state_ref: trajectory of goal state, torch.tensor with dimension [num_states x H x state_dim]
        :param action_min: minimum value of action
        :param action_max: maximum value of action
        :param Q: weighting matrix for (state-x_ref)^2, torch.tensor with dimension [state_dim x state_dim]
        :param R: weighting matrix for (action)^2, torch.tensor with dimension [action_dim x action_dim]
        :param r: weighting matrix for (del_action)^2, torch.tensor with dimension [action_dim x action_dim]
        """
        super(MPC, self).__init__()
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_states = num_states
        self.num_actions = num_actions
        self.H = H

        if action_min is None or action_max is None:  # assuming the actions are not constrained
            self._constraint = False
        else:
            self._constraint = True

        # inferring the action constraints
        self.action_min = action_min
        self.action_max = action_max

        assert action_min < action_max
        self.action_bnds = [(action_min, action_max) for _ in range(self.H * self.num_actions * self.action_dim)]
        self.action_bnds = tuple(self.action_bnds)

        if state_ref is None:  # infer the ground state as reference
            state_ref = torch.zeros(num_states, H, state_dim)

        self.history = None
        # TODO: Asserting / correcting the given 'state_ref' is in valid specification
        self.x_ref = state_ref
        self.u_prev = None

        # state deviation penalty matrix
        if Q is None:
            Q = torch.eye(state_dim * num_states)
        if isinstance(Q, np.ndarray):
            Q = torch.tensor(Q).float()
        self.Q = Q.to(device)

        # action exertion penalty matrix
        if R is None:
            R = torch.zeros(self.action_dim * num_actions, self.action_dim * num_actions)
        if isinstance(R, np.ndarray):
            R = torch.tensor(R).float()
        self.R = R.to(device)

        # delta action penalty matrix
        if r is None:
            r = torch.zeros(self.action_dim, self.action_dim)
        if isinstance(r, np.ndarray):
            r = torch.tensor(r).float()
        self.r = r.to(device)

        self.is_convex = is_convex
        self.device = device

    def roll_out(self, history, us):

        """
        :param history: history.'torch.tensor' with dimension of [#. nodes x (hist length x state_dim)]
        :param us: action. 'torch.tensor' with dimension of [#. control nodes x rollout length x control dim]
        :return: rolled out sequence of states
        """
        xs, info = self.model.rollout(history, us)
        return xs

    @staticmethod
    def _compute_loss(deltas, weight_mat):
        """
        :param deltas:  # [num_steps x variable_dim]
        :param weight_mat: # [variable_dim x variable_dim]
        :return:
        """

        steps = deltas.shape[0]
        weight_mat = weight_mat.unsqueeze(dim=0)  # [1 x variable_dim x variable_dim]
        weight_mat = weight_mat.repeat_interleave(steps, dim=0)  # [num_steps x variable_dim x variable_dim]
        deltas_transposed = deltas.unsqueeze(dim=1)  # [num_steps x 1 x variable_dim]
        deltas = deltas.unsqueeze(dim=-1)  # [num_steps x variable_dim x 1]
        loss = deltas_transposed.bmm(weight_mat).bmm(deltas)  # [num_steps x 1 x 1]
        loss = loss.mean()  # sum()
        return loss

    def compute_objective(self, history, us, x_ref=None, u_prev=None):
        """
        :param history: history.'torch.tensor' with dimension of [#. nodes x (hist length x state_dim)]
        :param us: action sequences assuming the first dimension is for time stamps.
        expected to get 'torch.tensor' with dimension [time stamps *  action_dim]
        :param x_ref: state targets
        """
        assert self.H == us.shape[1], \
            "The length of given action sequences doesn't match with receeding horizon length H."

        # Compute state deviation loss
        x_preds = self.roll_out(history, us)  # [#.nodes x  time stamps x state_dim]

        # [#. glass TC x time x state_dim]
        x_preds = x_preds[(~self.model.g.ndata['is_control'].bool()).squeeze(), :, :]
        x_preds = x_preds.transpose(0, 1).squeeze(-1)  # [time stamps x num_states]
        if x_ref is None:
            x_ref = self.x_ref  # [#. glass TC x time]
        x_deltas = x_preds - x_ref  # [time stamps x state_dim]
        if self.is_convex:
            state_loss = self._compute_loss(F.relu(-x_deltas), self.Q)
        else:
            state_loss = self._compute_loss(x_deltas, self.Q)

        # Compute action exertion loss
        if self.is_convex:
            us_norm = us - self.action_min
            action_loss = self._compute_loss(us_norm.squeeze().T, self.R)
        else:
            action_loss = self._compute_loss(us.squeeze().T, self.R)

        if u_prev is None:
            u_prev = self.u_prev
        us = torch.cat([u_prev, torch.squeeze(us)], dim=1)
        delta_actions = us[:, 1:] - us[:, :-1]  # [num_actions x time stamps]
        delta_action_loss = self._compute_loss(delta_actions.T, self.r)

        # construct MPC loss
        loss = state_loss + action_loss + delta_action_loss
        return loss

    def set_mpc_params(self, history, x_ref=None, u_prev=None, action_bnds=None):
        """
        :param history: torch.Tensor with dimensions of [# nodes x (hist length x state_dim)]
        :param x_ref: torch.Tensor with dimensions of [num_states x H x state_dim]
        :param u_prev: torch.Tensor with dimensions of [num_actions x action_dim]
        :return:
        """
        self.history = history.to(self.device)

        if x_ref is not None:
            self.x_ref = x_ref.to(self.device)
        if u_prev is not None:
            self.u_prev = u_prev.to(self.device)
        if action_bnds is not None:
            self.action_bnds = action_bnds

    def _obj(self, us: np.array):
        us = torch.from_numpy(us).float()
        us = us.view(self.num_actions, self.H, self.action_dim).to(self.device)
        with torch.no_grad():
            obj = self.compute_objective(history=self.history, us=us, x_ref=self.x_ref).cpu().numpy()
        return obj.item()

    def _hes(self, us):
        return self.compute_objective(history=self.history, us=us, x_ref=self.x_ref)

    def _obj_jac(self, us: np.array):
        us = torch.from_numpy(us).float().to(self.device)
        us = us.view(self.num_actions, self.H, self.action_dim)
        jac = torch.autograd.functional.jacobian(self._hes, us)
        jac = jac.reshape(-1, self.action_dim).double().cpu().numpy()
        # jac = torch.clamp(jac, -10., 10.).detach().numpy()
        return jac

    def solve(self, u0=None):
        """
        :param u0:  np.array [#. control nodes x H x 1]
        expected to get 'torch.tensor' with dimension [time stamps *  action_dim]
        :return:
        """
        if u0 is None:
            u0 = np.random.uniform(low=self.action_min, high=self.action_max,
                                   size=(self.num_actions * self.H * self.action_dim))
            u0 = np.clip(u0, a_min=self.action_bnds[0], a_max=self.action_bnds[1])
        opt_result = minimize(self._obj, u0,
                              method='L-BFGS-B',
                              bounds=self.action_bnds,
                              jac=self._obj_jac)

        opt_action = torch.tensor(opt_result.x).view(self.num_actions,
                                                     self.H,
                                                     self.action_dim).float().detach()  # optimized action sequences
        with torch.no_grad():
            pred_states = self.roll_out(self.history, opt_action)
        return opt_action, pred_states, opt_result
