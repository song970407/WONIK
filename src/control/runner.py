import numpy as np
import torch

from src.control.mpc import solve_mpc
from src.control.scipyMPC import MPC
from src.utils.rescale_data import rescale_data


class Runner_linear:
    """
    The experiment runner for WONIK-furnace control project
    Disclaimers: This class is designed for "Proof-of-Concept" purpose. Major refactoring
    would be done in the future including the input-output structures.
    """

    def __init__(self,
                 A: np.array,
                 B: np.array,
                 scaler: tuple,
                 consider_prev_action: bool = True):
        # check the input A matrix is a square matrix
        assert len(A.shape) == 2 and (A.shape[0] == A.shape[1]), "A must be a square matrix"
        self.A = A
        self.state_dim = A.shape[0]

        # check the input B matrix has the same dimension of A.
        assert self.state_dim == B.shape[0], "B must be a matrix of size ['#. states' x '#. actions']"
        self.B = B
        self.action_dim = B.shape[1]

        self.scaler = scaler
        self.consider_prev_action = consider_prev_action
        self._action_executed = None  # the list of executed actions

    def scale_states(self, states):
        min, max = self.scaler
        return (states - min) / (max - min)

    def unscale_states(self, scaled_states):
        min, max = self.scaler
        return (max - min) * scaled_states + min

    def array_like_to_numpy(self, x, size: tuple = None):
        size = (1, -1) if size is None else size
        return np.reshape(np.array(x), newshape=size)

    def step(self, state, target_states, **mpc_params):
        """ The function for finding "good" action
        :param state:
        :param target_states:
        :return: action
        """

        # predicted states / target_states
        # scale states / target_states
        state = self.array_like_to_numpy(state, size=(-1))
        target_states = self.array_like_to_numpy(target_states, size=(-1, self.state_dim))

        _mpc_params = {'A': self.A, 'B': self.B, 'x0': self.scale_states(state),
                       'xref': self.scale_states(target_states)}
        if self.consider_prev_action:
            if self._action_executed is None:
                prev_action = np.zeros(self.action_dim, )
            else:
                prev_action = self._action_executed
            _mpc_params['u_prev'] = prev_action
        _mpc_params.update(mpc_params)
        action = solve_mpc(**_mpc_params)
        self._action_executed = action
        return self.unscale_states(action)


class Runner(torch.nn.Module):

    def __init__(self,
                 config,
                 model,
                 state_dim,
                 action_dim,
                 num_states,
                 num_actions,
                 H,
                 action_min,
                 action_max,
                 Q,
                 R,
                 r,
                 is_convex,
                 device):
        super(Runner, self).__init__()
        self.config = config
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_states = num_states
        self.num_actions = num_actions
        self.H = H
        self.action_min = action_min
        self.action_max = action_max
        self.Q = Q
        self.R = R
        self.r = r
        self.is_convex = is_convex
        self.solver = MPC(model=model,
                          state_dim=state_dim,
                          action_dim=action_dim,
                          num_states=num_states,
                          num_actions=num_actions,
                          H=H,
                          action_min=action_min,
                          action_max=action_max,
                          Q=Q,
                          R=R,
                          r=r,
                          is_convex=is_convex,
                          device=device)

    def array_like_to_numpy(self, x, size: tuple = None):
        size = (1, -1) if size is None else size
        return np.reshape(np.array(x), newshape=size)

    def step(self, history, x_ref, u_prev):
        """
        :param history: numpy ndarray with [num_states+num_actions x HIST_WINDOW]
        :param x_ref: numpy ndarray with [num_states x H]
        :param u_prev: numpy ndarray with [num_actions x 1]
        :return: numpy ndarray with [num_actions x 1]
        """
        history = torch.from_numpy(history).float()
        x_ref = torch.from_numpy(x_ref).float()
        u_prev = torch.from_numpy(u_prev).float()
        if self.config.data.value.SCALING:
            maximum_temperature_dif = 0.8
            norm_dif = maximum_temperature_dif / (self.config.data.value.scaler[1] - self.config.data.value.scaler[0])
            x_ref = (x_ref - self.config.data.value.scaler[0]) / (
                        self.config.data.value.scaler[1] - self.config.data.value.scaler[0])
            history = (history - self.config.data.value.scaler[0]) / (
                        self.config.data.value.scaler[1] - self.config.data.value.scaler[0])
            u_prev = (u_prev - self.config.data.value.scaler[0]) / (
                        self.config.data.value.scaler[1] - self.config.data.value.scaler[0])
            action_bnds = [(max(0.0, u_prev[int(i / self.H)].item() - (1 + i % self.H) * norm_dif),
                            min(1.0, u_prev[int(i / self.H)].item() + (1 + i % self.H) * norm_dif)) for i in
                           range(self.H * self.num_actions)]
            action_bnds = tuple(action_bnds)

            self.solver.set_mpc_params(history=history,
                                       x_ref=x_ref,
                                       u_prev=u_prev,
                                       action_bnds=action_bnds)
        else:
            self.solver.set_mpc_params(history=history,
                                       x_ref=x_ref,
                                       u_prev=u_prev)
        u0 = np.stack([u_prev for _ in range(self.H)]).reshape(-1)
        opt_actions, pred_states, _ = self.solver.solve(u0)  # [num_actions x 1 x 1]
        if self.config.data.value.SCALING:
            opt_actions = rescale_data(opt_actions, self.config.data.value.scaler)
        return opt_actions[:, 0, :].detach().numpy()  # [num_actions x 1]
