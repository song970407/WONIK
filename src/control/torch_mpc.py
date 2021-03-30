import numpy as np
import stopit
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Actions(nn.Module):
    def __init__(self,
                 ref: torch.tensor,  # [H] torch.tensor
                 num_actions: int,
                 action_dim: int,
                 u_min: float,
                 u_max: float):
        super(Actions, self).__init__()
        us = np.reshape(ref.cpu().detach().numpy(), (1, -1, 1))
        us = np.repeat(us, num_actions, axis=0)
        us = np.repeat(us, action_dim, axis=2)
        us = np.clip(us, u_min, u_max)
        self.us = torch.nn.Parameter(torch.from_numpy(us).float())

    def forward(self):
        return self.us


class Linear_Actions(nn.Module):
    def __init__(self,
                 ref: torch.tensor,  # [H] torch.tensor
                 num_actions: int,
                 u_min: float,
                 u_max: float):
        super(Linear_Actions, self).__init__()
        # us = np.random.uniform(low=u_min, high=u_max, size=(H, num_actions))
        us = np.reshape(ref.cpu().detach().numpy(), (-1, 1))
        us = np.repeat(us, num_actions, axis=1)
        us = np.clip(us, u_min, u_max)
        self.us = torch.nn.Parameter(torch.from_numpy(us).float())

    def forward(self):
        return self.us


def get_discount_factor(horizon_length, gamma):
    g = 1.0
    gs = [g]
    for i in range(1, horizon_length):
        g *= gamma
        gs.append(g)
    return torch.tensor(gs)


def loss_function(prediction, target, crit, weight=None):
    if weight is None:
        loss = crit()
    dif = prediction - target
    return nn.functional.relu(dif) * weight[:, :, 1] - nn.functional.relu(-dif) * weight[:, :, 0]


class TorchMPC(nn.Module):

    def __init__(self, model,
                 time_aggregator: str = 'sum',
                 optimizer_mode: str = 'Adam',
                 max_iter: int = None,
                 u_min: float = 0.0,
                 u_max: float = 1.0,
                 alpha: float = 1.0,
                 timeout: float = 300,
                 device: str = None,
                 opt_config: dict = {}):
        super(TorchMPC, self).__init__()

        if device is None:
            print("Running device is not given. Infer the running device from the system configuration ...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Initiating solver with {}".format(device))
        self.device = device

        self.model = model.to(device)
        self.time_aggregator = time_aggregator
        self.optimizer_mode = optimizer_mode
        if max_iter is None:
            self.max_iter = 20 if self.optimizer_mode == 'Adam' else 5
        self.u_min = u_min
        self.u_max = u_max
        self.alpha = alpha
        self.timeout = timeout
        self.opt_config = opt_config

    def solve_mpc_adam(self, graph, history, target, weight=None):
        """
        Args:
            graph: DGL graph
            history: tuple (tc history, control history)
                - tc history [#.tc sensors x history length]
                - control history [#. control sensors x history length]
            target: torch.tensor [#.tc tensors x history length]
            weight: torch.tensor [#.tc tensors x history length x 2]
        Returns:
        """
        crit = torch.nn.MSELoss(reduction='none')
        gamma_mask = get_discount_factor(target.shape[1], 1.0).view(1, -1).to(self.device)
        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            us = Actions(target[0, :], history[1].shape[0], 1, self.u_min, self.u_max).to(self.device)
            opt = torch.optim.Adam(us.parameters(), lr=1e-3)
            with torch.no_grad():
                h0 = self.model.filter_history(graph,
                                               history[0],
                                               history[1])
            for i in range(self.max_iter):
                opt.zero_grad()
                prediction = self.predict_future(graph, h0, us())
                loss = F.relu(prediction - target) * weight[:, :, 1] - F.relu(target - prediction) * weight[:, :, 0]
                loss = ((loss ** 2) * gamma_mask).sum()
                loss.backward()
                opt.step()
                us.us.data = us.us.data.clamp(min=0.0, max=1.0)

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.us.data = us.us.data.clamp(min=0.0, max=1.0)
        return us

    def solve_mpc_LBFGS(self, graph, history, target, weight=None):
        crit = torch.nn.MSELoss(reduction='none')
        gamma_mask = get_discount_factor(target.shape[1], 1.0).view(1, -1).to(self.device)

        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            us = Actions(target[0, :], history[1].shape[0], 1, self.u_min, self.u_max).to(self.device)
            opt = torch.optim.LBFGS(us.parameters(),
                                    history_size=15,
                                    max_iter=self.max_iter,
                                    line_search_fn='strong_wolfe',
                                    **self.opt_config)
            with torch.no_grad():
                h0 = self.model.filter_history(graph,
                                               history[0],
                                               history[1])
            for i in range(self.max_iter):
                def closure():
                    opt.zero_grad()
                    prediction = self.predict_future(graph, h0, us())
                    loss = F.relu(prediction - target) * weight[:, :, 1] - F.relu(target - prediction) * weight[:, :, 0]
                    loss = ((loss ** 2) * gamma_mask).sum()
                    init_concat_us = torch.cat([history[1][:, -1:, :], us.us], dim=1)
                    delta_u_loss = (init_concat_us[:, 1:, :] - init_concat_us[:, :-1, :]).pow(2).sum()
                    loss = loss + self.alpha * delta_u_loss
                    loss.backward()
                    return loss
                opt.step(closure)
                us.us.data = us.us.data.clamp(min=0.0, max=1.0)

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.us.data = us.us.data.clamp(min=0.0, max=1.0)
        return us

    def solve_mpc(self, graph, history, target, weight=None):
        if self.optimizer_mode == 'Adam':
            opt_actions = self.solve_mpc_adam(graph, history, target, weight)
        elif self.optimizer_mode == 'LBFGS':
            opt_actions = self.solve_mpc_LBFGS(graph, history, target, weight)
        return opt_actions

    def predict_future(self, g, h0, us):
        ## TODO: Design a module functions for coping w/ different model outputs.
        prediction_mu_logvar = self.model.multi_step_prediction(g, h0, us)
        prediction = prediction_mu_logvar[:, :, 0]
        return prediction

    def solve_max_entropy(self, graph, history):
        pass


class LinearTorchMPC(nn.Module):

    def __init__(self, model,
                 time_aggregator: str = 'sum',
                 optimizer_mode: str = 'Adam',
                 max_iter: int = None,
                 u_min: float = 0.0,
                 u_max: float = 1.0,
                 alpha: float = 1.0,
                 timeout: float = 300,
                 device: str = None,
                 opt_config: dict = {}):
        super(LinearTorchMPC, self).__init__()

        if device is None:
            print("Running device is not given. Infer the running device from the system configuration ...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Initiating solver with {}".format(device))
        self.device = device

        self.model = model.to(device)
        self.time_aggregator = time_aggregator
        self.optimizer_mode = optimizer_mode
        if max_iter is None:
            self.max_iter = 20 if self.optimizer_mode == 'Adam' else 5
        self.u_min = u_min
        self.u_max = u_max
        self.alpha = alpha
        self.timeout = timeout
        self.opt_config = opt_config

    def solve_mpc_adam(self, history_tc, history_ws, target, initial_ws=None):
        crit = torch.nn.MSELoss(reduction='sum')
        start = time.time()
        trajectory_us_value = []
        trajectory_us_gradient = []
        trajectory_loss_objective = []
        trajectory_loss_delta_u = []
        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            if initial_ws is None:
                us = torch.nn.Parameter(target[:, :history_ws.shape[1]]).to(self.device)
            else:
                us = torch.nn.Parameter(initial_ws).to(self.device)
            opt = torch.optim.Adam([us], lr=1e-4)
            for i in range(self.max_iter):
                opt.zero_grad()
                prediction = self.predict_future(history_tc, history_ws, us)
                loss_objective = crit(prediction, target)
                init_concat_us = torch.cat([history_ws[-1:, :], us], dim=0)
                loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                loss_delta_u = self.alpha * loss_delta_u
                loss = loss_objective + loss_delta_u
                loss.backward()
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
                trajectory_us_value.append(us.data.cpu().detach())
                trajectory_us_gradient.append(us.grad.data.cpu().detach())
                trajectory_loss_objective.append(loss_objective.cpu().detach())
                trajectory_loss_delta_u.append(loss_delta_u.cpu().detach())
                opt.step()
        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.data = us.data.clamp(min=self.u_min, max=self.u_max)
        with torch.no_grad():
            prediction = self.predict_future(history_tc, history_ws, us)
            loss_objective = crit(prediction, target)
            init_concat_us = torch.cat([history_ws[-1:, :], us], dim=0)
            loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
            loss_delta_u = self.alpha * loss_delta_u
        trajectory_us_value.append(us.data.cpu().detach())
        trajectory_us_gradient.append(us.grad.data.cpu().detach())
        trajectory_loss_objective.append(loss_objective.cpu().detach())
        trajectory_loss_delta_u.append(loss_delta_u.cpu().detach())
        end = time.time()
        if len(trajectory_us_value) > 0:
            trajectory_us_value = torch.stack(trajectory_us_value)
        if len(trajectory_us_gradient) > 0:
            trajectory_us_gradient = torch.stack(trajectory_us_gradient)
        if len(trajectory_loss_objective) > 0:
            trajectory_loss_objective = torch.stack(trajectory_loss_objective)
        if len(trajectory_loss_delta_u) > 0:
            trajectory_loss_delta_u = torch.stack(trajectory_loss_delta_u)
        log = {}
        log['trajectory_us_value'] = trajectory_us_value
        log['trajectory_us_gradient'] = trajectory_us_gradient
        log['trajectory_loss_objective'] = trajectory_loss_objective
        log['trajectory_loss_delta_u'] = trajectory_loss_delta_u
        log['total_time'] = end-start
        return us, log

    def solve_mpc_LBFGS(self, history_tc, history_ws, target, initial_ws=None):
        """
        :param history_tc: [history_length x num_state]
        :param history_ws: [history_length-1 x num_action]
        :param target: [H x num_state]
        :param initial_ws: [H x num_action]
        :return:
        """
        crit = torch.nn.MSELoss(reduction='sum')
        start = time.time()
        trajectory_us_value = []
        trajectory_us_gradient = []
        trajectory_loss_objective = []
        trajectory_loss_delta_u = []

        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            if initial_ws is None:
                us = torch.nn.Parameter(target[:, :history_ws.shape[1]]).to(self.device)
            else:
                us = torch.nn.Parameter(initial_ws).to(self.device)
            opt = torch.optim.LBFGS(params=[us],
                                    history_size=15,
                                    max_iter=self.max_iter,
                                    line_search_fn='strong_wolfe',
                                    **self.opt_config)

            for i in range(self.max_iter):
                def closure():
                    opt.zero_grad()
                    prediction = self.predict_future(history_tc, history_ws, us)
                    loss_objective = crit(prediction, target)
                    init_concat_us = torch.cat([history_ws[-1:, :], us], dim=0)
                    loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                    loss_delta_u = self.alpha * loss_delta_u
                    loss = loss_objective + loss_delta_u
                    loss.backward()
                    return loss
                with torch.no_grad():
                    prediction = self.predict_future(history_tc, history_ws, us)
                    loss_objective = crit(prediction, target)
                    init_concat_us = torch.cat([history_ws[-1:, :], us], dim=0)
                    loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                    loss_delta_u = self.alpha * loss_delta_u
                trajectory_us_value.append(us.data.cpu().detach())
                opt.step(closure)
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
                trajectory_us_gradient.append(us.grad.data.cpu().detach())
                trajectory_loss_objective.append(loss_objective.cpu().detach())
                trajectory_loss_delta_u.append(loss_delta_u.cpu().detach())

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.data = us.data.clamp(min=self.u_min, max=self.u_max)
        trajectory_us_value.append(us.data.cpu().detach())
        end = time.time()
        log = {}
        if len(trajectory_us_value) > 0:
            trajectory_us_value = torch.stack(trajectory_us_value)
        if len(trajectory_us_gradient) > 0:
            trajectory_us_gradient = torch.stack(trajectory_us_gradient)
        if len(trajectory_loss_objective) > 0:
            trajectory_loss_objective = torch.stack(trajectory_loss_objective)
        if len(trajectory_loss_delta_u) > 0:
            trajectory_loss_delta_u = torch.stack(trajectory_loss_delta_u)
        log['trajectory_us_value'] = trajectory_us_value
        log['trajectory_us_gradient'] = trajectory_us_gradient
        log['trajectory_loss_objective'] = trajectory_loss_objective
        log['trajectory_loss_delta_u'] = trajectory_loss_delta_u
        log['total_time'] = end - start
        return us, log

    def solve_mpc(self, history_tc, history_ws, target, initial_ws=None):
        opt_actions = None
        log = {}
        if self.optimizer_mode == 'Adam':
            opt_actions, log = self.solve_mpc_adam(history_tc, history_ws, target, initial_ws)
        elif self.optimizer_mode == 'LBFGS':
            opt_actions, log = self.solve_mpc_LBFGS(history_tc, history_ws, target, initial_ws)
        return opt_actions, log

    def predict_future(self, history_tc, history_ws, us):
        ## TODO: Design a module functions for coping w/ different model outputs.
        prediction = self.model.multi_step_prediction(history_tc, history_ws, us)
        return prediction

    def solve_max_entropy(self, graph, history):
        pass
