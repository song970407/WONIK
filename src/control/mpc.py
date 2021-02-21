from typing import Union, List

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pyMPC.mpc import MPCController
from scipy.optimize import minimize


def solve_mpc(A: np.array,
              B: np.array,
              x0: np.array,
              xref: np.array,
              u_prev: np.array = None,
              Qx: np.array = None,
              QDu: np.array = None):
    """
    :param A: state-state transition matrix of linear model; expected size is [num states, num states]
    :param B: state-action transition matrix of linear model; expected size is [num states, num actions]
    :param x0: current states values; expected size is [num states, ]
    :param xref: the target states (set-points); expected size is [prediction horizon H, num states]
    High prediction horizon H requires more computational time.
    It is recommended to use H in [30, 60]
    :param u_prev: previously exerted actions
    :param Qx: penalty matrix for state deviation; expected size is [num_states x num_states]
    :param QDu: penalty matrix for action changing; expected size is [num_actions x num_actions]

    :return: u: optimized action for current time.
    """

    state_dim, action_dim = A.shape[0], B.shape[1]
    # check the integrity of the x0
    assert len(x0.shape) == 1 and x0.size == state_dim, "x0 must be a vector of size ['#. states']"

    # check the integrity of the xref
    assert len(xref.shape) == 2 and xref.shape[1] == state_dim, "xref must a vector of size [H+1 x '#. states']"
    H = xref.shape[0]

    if Qx is None:
        Qx = np.eye(state_dim)

    if isinstance(Qx, float) or isinstance(Qx, int):
        Qx = np.eye(state_dim) * Qx

    if QDu is None:
        QDu = np.eye(action_dim) * 1
    if isinstance(QDu, float) or isinstance(QDu, int):
        QDu = np.eye(action_dim) * QDu
    if u_prev is not None:
        K = MPCController(A,
                          B,
                          Np=H - 1,
                          x0=x0,
                          xref=xref,
                          uminus1=u_prev,
                          Qx=Qx,
                          QDu=QDu,
                          umin=np.zeros(action_dim),
                          umax=np.ones(action_dim))
    else:
        K = MPCController(A,
                          B,
                          Np=H - 1,
                          x0=x0,
                          xref=xref,
                          Qx=Qx,
                          QDu=QDu,
                          umin=np.zeros(action_dim),
                          umax=np.ones(action_dim))
    K.setup()
    _, ret_dict = K.output(return_x_seq=True, return_u_seq=True)
    opt_action = ret_dict['u_seq'][0, :]  # numpy array with size of [num_action]
    return opt_action




class MPC_ICGRNN:

    def __init__(self,
                 model: nn.Module,
                 g: dgl.graph,
                 state_dim: int,
                 action_dim: int,
                 H: int,
                 history: torch.tensor = None,
                 state_ref: Union[np.array, torch.tensor] = None,
                 action_min: Union[float, List[float]] = None,
                 action_max: Union[float, List[float]] = None,
                 Q: Union[np.array, torch.tensor] = None,
                 R: Union[np.array, torch.tensor] = None,
                 r: Union[np.array, torch.tensor] = None,
                 device: str = 'cpu'):
        """
        :param model: an instance of pytorch nn.module. Assume that this model is ICGRNN model
        the input of model expected to be [1 x state_dim] and [1 x action_dim]
        the output of model expected to be [1 x state_dim]
        :param state_dim: dimension of state
        :param action_dim: dimension of action
        :param H: receeding horizon
        :param state_ref: trajectory of goal state, torch.tensor with dimension [H x state_dim]
        :param action_min: minimum value of action
        :param action_max: maximum value of action
        :param Q: weighting matrix for (state-x_ref)^2, torch.tensor with dimension [state_dim x state_dim]
        :param R: weighting matrix for (action)^2, torch.tensor with dimension [action_dim x action_dim]
        :param r: weighting matrix for (del_action)^2, torch.tensor with dimension [action_dim x action_dim]
        """
        self.model = model
        self.g = g
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.H = H
        self.history = history
        self.device = device

        if action_min is None or action_max is None:  # assuming the actions are not constrained
            self._constraint = False
        else:
            self._constraint = True

        # inferring the action constraints
        if isinstance(action_min, float):
            self.action_min = [action_min] * self.action_dim * self.H
        else:
            self.action_min = action_min

        if isinstance(action_max, float):
            self.action_max = [action_max] * self.action_dim * self.H
        else:
            self.action_max = action_max

        self.action_bnds = []
        for a_min, a_max in zip(self.action_min, self.action_max):
            assert a_min < a_max, "Action min is larger or equal to the action max"
            self.action_bnds.append((a_min, a_max))
        self.action_bnds = tuple(self.action_bnds)

        if state_ref is None:  # infer the ground state as reference
            state_ref = torch.zeros(H, state_dim).to(self.device)

        self.x0 = None
        # TODO: Asserting / correcting the given 'state_ref' is in valid specification
        self.x_ref = state_ref
        self.u_prev = None

        # state deviation penalty matrix
        if Q is None:
            Q = torch.eye(state_dim).to(self.device)
        if isinstance(Q, np.ndarray):
            Q = torch.tensor(Q).float().to(self.device)
        self.Q = Q

        # action exertion penalty matrix
        if R is None:
            R = torch.zeros(self.action_dim, self.action_dim).to(self.device)
        if isinstance(R, np.ndarray):
            R = torch.tensor(R).float().to(self.device)
        self.R = R

        # delta action penalty matrix
        if r is None:
            r = torch.zeros(self.action_dim, self.action_dim).to(self.device)
        if isinstance(r, np.ndarray):
            r = torch.tensor(r).float().to(self.device)
        self.r = r

    def roll_out(self, x0, us):

        """
        :param x0: initial state. expected to get 'torch.tensor' with dimension of [1 x state_dim x 1]
        :param us: action sequences assuming the first dimension is for time stamps.
        expected to get 'torch.tensor' with dimension [1 x time stamps x  action_dim x 1]
        :return: rolled out sequence of states
        """
        return torch.squeeze(self.model.rollout_mpc(self.g, self.history, x0, us))  # [time stamps x state_dim]

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

    def compute_objective(self, x0, us, x_ref=None, u_prev=None):
        """
        :param x0: initial state. expected to get 'torch.tensor' with dimension of [1 x state_dim x 1]
        :param us: action sequences assuming the first dimension is for time stamps.
        expected to get 'torch.tensor' with dimension [1 x time stamps x  action_dim x 1]
        :param x_ref: state targets
        """
        assert self.H == us.shape[1], \
            "The length of given action sequences doesn't match with receeding horizon length H."

        # Compute state deviation loss
        x_preds = self.roll_out(x0, us)  # [time stamps x state_dim]
        if x_ref is None:
            x_ref = self.x_ref  # [time stamps x state_dim]
        x_deltas = x_preds - x_ref  # [time stamps x state_dim]
        state_loss = self._compute_loss(x_deltas, self.Q)

        # Compute action exertion loss
        action_loss = self._compute_loss(torch.squeeze(us), self.R)

        # Compute delta action loss
        if u_prev is None:
            u_prev = torch.zeros((1, self.action_dim)).to(self.device)
        us = torch.cat([u_prev, torch.squeeze(us)], dim=0)
        delta_actions = us[:1, :] - us[:-1, :]
        delta_action_loss = self._compute_loss(delta_actions, self.r)

        # construct MPC loss
        loss = state_loss + action_loss + delta_action_loss
        return loss

    def set_mpc_params(self, history, x0, x_ref=None, u_prev=None):
        self.history = history
        self.x0 = x0

        if x_ref is not None:
            self.x_ref = x_ref
        self.u_prev = u_prev

    def _obj(self, us: np.array):
        us = torch.from_numpy(us).float().to(self.device)
        us = us.view(1, self.H, self.action_dim, 1)
        with torch.no_grad():
            obj = self.compute_objective(x0=self.x0, us=us, x_ref=self.x_ref, u_prev=self.u_prev).cpu().detach().numpy()
        return obj

    def _obj_jac(self, us: np.array):
        us = torch.from_numpy(us).float().to(self.device)
        us = us.view(1, self.H, self.action_dim, 1)

        def _hes(us): return self.compute_objective(x0=self.x0, us=us, x_ref=self.x_ref, u_prev=self.u_prev)

        jac = torch.autograd.functional.jacobian(_hes, us)
        jac = torch.clamp(jac, -10, 10).cpu().detach().numpy()
        return np.reshape(jac, (-1))

    def solve(self, u0=None):
        """
        :param u0:  initial action sequences, only 1D.
        expected to get 'torch.tensor' with dimension [time stamps *  action_dim]
        :return:
        """
        if u0 is None:
            u0 = np.stack([self.action_max, self.action_min]).mean(axis=0)

        opt_result = minimize(self._obj, u0, method='SLSQP', bounds=self.action_bnds, jac=self._obj_jac)

        opt_action = torch.tensor(opt_result.x).view(self.H, self.action_dim).float().detach().to(
            self.device)  # optimized action sequences
        pred_states = self.roll_out(self.x0, opt_action.view(1, self.H, self.action_dim, 1))
        return opt_action, pred_states, opt_result


class MPC_ICGATRNN:

    def __init__(self,
                 model: nn.Module,
                 g: dgl.graph,
                 state_dim: int,
                 action_dim: int,
                 history_dim: int,
                 H: int,
                 action_min: Union[float, List[float]] = None,
                 action_max: Union[float, List[float]] = None,
                 Q: Union[np.array, torch.tensor] = None,
                 R: Union[np.array, torch.tensor] = None,
                 r: Union[np.array, torch.tensor] = None,
                 device: str = 'cpu'):
        """
        :param model: an instance of pytorch nn.module. Assume that this model is ICGATRNN model
        :param g : dgl.graph
        :param state_dim: dimension of state
        :param action_dim: dimension of action
        :param H: receeding horizon
        :param action_min: minimum value of action
        :param action_max: maximum value of action
        :param Q: weighting matrix for (state-x_ref)^2, torch.tensor with dimension [state_dim x state_dim]
        :param R: weighting matrix for (action)^2, torch.tensor with dimension [action_dim x action_dim]
        :param r: weighting matrix for (del_action)^2, torch.tensor with dimension [action_dim x action_dim]
        """
        self.model = model
        self.g = g
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_dim = history_dim
        self.H = H
        self.device = device

        if action_min is None or action_max is None:  # assuming the actions are not constrained
            self._constraint = False
        else:
            self._constraint = True

        # inferring the action constraints
        if isinstance(action_min, float):
            self.action_min = [action_min] * self.action_dim * self.H
        else:
            self.action_min = action_min

        if isinstance(action_max, float):
            self.action_max = [action_max] * self.action_dim * self.H
        else:
            self.action_max = action_max

        self.action_min = np.reshape(np.array(self.action_min), newshape=(self.H, self.action_dim))
        self.action_max = np.reshape(np.array(self.action_max), newshape=(self.H, self.action_dim))

        self.xs = None
        self.x_ref = None
        self.u_prev = None

        # state deviation penalty matrix
        if Q is None:
            Q = torch.eye(state_dim).to(self.device)
        if isinstance(Q, np.ndarray):
            Q = torch.tensor(Q).float().to(self.device)
        self.Q = Q

        # action exertion penalty matrix
        if R is None:
            R = torch.zeros(self.action_dim, self.action_dim).to(self.device)
        if isinstance(R, np.ndarray):
            R = torch.tensor(R).float().to(self.device)
        self.R = R

        # delta action penalty matrix
        if r is None:
            r = torch.zeros(self.action_dim, self.action_dim).to(self.device)
        if isinstance(r, np.ndarray):
            r = torch.tensor(r).float().to(self.device)
        self.r = r

    def set_mpc_params(self, xs=None, x_ref=None, u_prev=None):
        """
        :param xs: torch.tensor with dimension [1 x history_dim x (state_dim+action_dim)]
        :param x_ref: torch.tensor with dimension [H x state_dim]
        :param u_prev: torch.tensor with dimension [1 x action_dim]
        :return:
        """
        self.xs = xs
        self.x_ref = x_ref
        self.u_prev = u_prev

    def roll_out(self, xs, us):
        """
        :param xs: initial state. expected to get 'torch.tensor' with dimension of [1 x (state_dim+action_dim) x 1]
        :param us: action sequences assuming the first dimension is for time stamps.
        expected to get 'torch.tensor' with dimension [1 x time stamps x  action_dim x 1]
        :return: rolled out sequence of states
        """
        return torch.squeeze(self.model.rollout_mpc(self.g, xs, us), dim=-1)  # [time stamps x state_dim]

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

    def compute_objective(self, xs, us, x_ref=None, u_prev=None):
        """
        :param xs: initial state. expected to get 'torch.tensor' with dimension of [1 x self.history_dim x self.state_dim]
        :param us: action sequences assuming the first dimension is for time stamps.
        expected to get 'torch.tensor' with dimension [1 x time stamps x  action_dim x 1]
        :param x_ref: state targets
        """
        assert self.H == us.shape[1], \
            "The length of given action sequences doesn't match with receeding horizon length H."

        # Compute state deviation loss
        x_preds = torch.squeeze(self.model.rollout_mpc(self.g, xs, us), dim=-1)  # [time stamps x state_dim]
        if x_ref is None:
            x_ref = self.x_ref  # [time stamps x state_dim]
        x_deltas = x_preds - x_ref  # [time stamps x state_dim]

        state_loss = self._compute_loss(F.relu(-x_deltas), self.Q)
        print('Loss print')
        print(state_loss.item())
        # Compute action exertion loss
        action_loss = self._compute_loss(torch.squeeze(us), self.R)
        print(action_loss.item())
        # Compute delta action loss
        if u_prev is None:
            u_prev = torch.zeros((1, self.action_dim)).to(self.device)
        us = torch.cat([u_prev, torch.squeeze(us)], dim=0)
        delta_actions = us[:1, :] - us[:-1, :]
        delta_action_loss = self._compute_loss(delta_actions, self.r)
        print(delta_action_loss.item())
        # construct MPC loss
        loss = state_loss + action_loss + delta_action_loss
        return loss

    def _obj(self, us: torch.tensor):
        us = us.view(1, self.H, self.action_dim, 1)
        with torch.no_grad():
            obj = self.compute_objective(xs=self.xs, us=us, x_ref=self.x_ref, u_prev=self.u_prev)
        return obj

    def _obj_jac(self, us: torch.tensor):
        us = us.view(1, self.H, self.action_dim, 1)

        def _hes(us): return self.compute_objective(xs=self.xs, us=us, x_ref=self.x_ref, u_prev=self.u_prev)

        jac = torch.autograd.functional.jacobian(_hes, us)
        jac = torch.clamp(jac, -10, 10)
        return jac.view(self.H, self.action_dim)

    def _bound_u(self, us, u_bound):
        if isinstance(us, torch.Tensor):
            return torch.from_numpy(np.clip(us.cpu().detach().numpy(), a_min=u_bound[0], a_max=u_bound[1])).float()
        else:
            return torch.from_numpy(np.clip(us, a_min=u_bound[0], a_max=u_bound[1])).float()

    def solve(self, u0=None, u_bound=None, step_size=1e-3, step_number=500):
        """
        :param u0:  initial action sequences, torch.tensor with dimension [H x action_dim]
        expected to get 'torch.tensor' with dimension [time stamps *  action_dim]
        :param u_bound: boundary of u, optional. tuple of np.array with dimension [H x action_dim]
        :param step_size: step size of GD. float. Default: 1e-4
        :param step_number: number of gradient update steps. Default: 100
        :return:
        """
        if u_bound is None:
            u_bound = (self.action_min, self.action_max)
        if u0 is None:
            u0 = self.u_prev
        us = self._bound_u(u0, u_bound)
        u_dif = []
        grad_norms = []
        for step in range(step_number):
            grad = self._obj_jac(us)
            # print(grad[0])
            grad_norms.append(torch.norm(grad).item())
            if torch.norm(grad).item() < 1e-6:
                print('Gradient is so small. Finish at ' + str(step + 1))
                break
            grad = grad / torch.norm(grad).item()
            next_us = us - step_size * grad / (step + 1) ** 2
            next_us = self._bound_u(next_us, u_bound)
            u_dif.append(torch.norm(next_us - us).item())
            us = next_us
        pred_states = self.roll_out(self.xs, us.view(1, self.H, self.action_dim, 1))
        return us, pred_states, u_dif, grad_norms
