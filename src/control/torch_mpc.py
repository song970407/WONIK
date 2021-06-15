import time

import numpy as np
import stopit
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 alpha: float = 1.0,
                 is_del_u: bool = False,
                 from_target: bool = False,
                 u_min: float = 0.0,
                 u_max: float = 1.0,
                 optimizer_mode: str = 'Adam',
                 max_iter: int = None,
                 timeout: float = 300,
                 is_logging: bool = True,
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
        self.alpha = alpha
        self.is_del_u = is_del_u
        self.from_target = from_target
        self.u_min = u_min
        self.u_max = u_max
        self.optimizer_mode = optimizer_mode
        if max_iter is None:
            # self.max_iter = 100000
            self.max_iter = 200 if self.optimizer_mode == 'Adam' else 200
        else:
            self.max_iter = max_iter
        self.timeout = timeout
        self.is_logging = is_logging
        self.opt_config = opt_config

    def compute_us_from_del_u(self, initial_u, del_u, weights):
        dus = torch.repeat_interleave(del_u.unsqueeze(dim=0), repeats=del_u.shape[0], dim=0)
        us = initial_u + torch.sum(weights * dus, dim=1)
        return us

    def solve_mpc_adam(self, history_tc, history_ws, target, initial_ws):
        """
        :param history_tc: [state_order x num_of_stateTCs]
        :param history_ws: [action_order-1 x action_dim]
        :param target: [H x num_of_glassTCs]
        :param initial_ws: [H x action_dim]
        :return:
        """
        H = target.shape[0]
        state_dim = target.shape[1]
        action_dim = history_ws.shape[1]
        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            crit = torch.nn.MSELoss(reduction='sum')
            log = {}
            if self.is_logging:
                log['history_tc'] = history_tc
                log['history_ws'] = history_ws
                log['target'] = target
                log['initial_ws'] = initial_ws
                start = time.time()
                trajectory_us_value = []
                # trajectory_us_gradient = []
                trajectory_loss_objective = []
                trajectory_loss_delta_u = []
                trajectory_loss = []
                trajectory_time = []
            data = initial_ws.tolist()
            us = torch.nn.Parameter(torch.tensor(data, device=self.device)).to(self.device)
            us.data = us.data.clamp(min=self.u_min, max=self.u_max)
            opt = torch.optim.Adam([us], lr=1e-3)
            if self.is_del_u:
                if not self.from_target:
                    weights = torch.repeat_interleave(torch.tril(torch.ones((H, H)).to(self.device)).unsqueeze(dim=-1),
                                                      repeats=action_dim, dim=-1)
                    initial_u = torch.repeat_interleave(history_ws[-1:, :], repeats=H, dim=0)
            for i in range(self.max_iter):
                opt.zero_grad()
                if self.is_del_u:
                    if self.from_target:
                        computed_us = us + target[:, :action_dim]
                        prediction = self.predict_future(history_tc, history_ws, computed_us)
                    else:
                        computed_us = self.compute_us_from_del_u(initial_u, us, weights)
                        prediction = self.predict_future(history_tc, history_ws, computed_us)
                else:
                    computed_us = us
                    prediction = self.predict_future(history_tc, history_ws, computed_us)
                loss_objective = crit(prediction[:, :state_dim], target)
                init_concat_us = torch.cat([history_ws[-1:, :], computed_us], dim=0)
                loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                loss_delta_u = self.alpha * loss_delta_u
                loss = loss_objective + loss_delta_u
                loss.backward()
                if self.is_logging:
                    with torch.no_grad():
                        trajectory_us_value.append(us.tolist())
                        # trajectory_us_gradient.append(us.grad.data.cpu().detach())
                        trajectory_loss_objective.append(loss_objective.item())
                        trajectory_loss_delta_u.append(loss_delta_u.item())
                        trajectory_loss.append(loss.item())
                opt.step()
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
                if self.is_logging:
                    end = time.time()
                    trajectory_time.append(end - start)

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.data = us.data.clamp(min=self.u_min, max=self.u_max)
        if self.is_logging:
            with torch.no_grad():
                if self.is_del_u:
                    if self.from_target:
                        computed_us = us + target[:, :action_dim]
                        prediction = self.predict_future(history_tc, history_ws, computed_us)
                    else:
                        computed_us = self.compute_us_from_del_u(initial_u, us, weights)
                        prediction = self.predict_future(history_tc, history_ws, computed_us)
                else:
                    computed_us = us
                    prediction = self.predict_future(history_tc, history_ws, computed_us)
                loss_objective = crit(prediction[:, :state_dim], target)
                init_concat_us = torch.cat([history_ws[-1:, :], computed_us], dim=0)
                loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                loss_delta_u = self.alpha * loss_delta_u
                loss = loss_objective + loss_delta_u

                trajectory_us_value.append(us.tolist())
                # trajectory_us_gradient.append(us.grad.data)
                trajectory_loss_objective.append(loss_objective.item())
                trajectory_loss_delta_u.append(loss_delta_u.item())
                trajectory_loss.append(loss.item())
                end = time.time()
                trajectory_time.append(end - start)
            # log['trajectory_us_value'] = trajectory_us_value
            # log['trajectory_us_gradient'] = trajectory_us_gradient
            log['trajectory_loss_objective'] = trajectory_loss_objective
            log['trajectory_loss_delta_u'] = trajectory_loss_delta_u
            log['trajectory_loss'] = trajectory_loss
            log['trajectory_time'] = trajectory_time
        # Return the best u
        idx = np.argmin(trajectory_loss)
        optimal_u = torch.tensor(trajectory_us_value[idx], device=self.device)
        if self.is_logging:
            log['optimal_us_value'] = trajectory_us_value[idx]
            log['idx_optimal_us_value'] = idx
        return optimal_u, log

    # HAVE TO FIX

    def solve_mpc_LBFGS(self, history_tc, history_ws, target, initial_ws=None):
        """
        :param history_tc: [history_length x num_state]
        :param history_ws: [history_length-1 x num_action]
        :param target: [H x num_state]
        :param initial_ws: [H x num_action]
        :return:
        """
        crit = torch.nn.MSELoss(reduction='sum')
        log = {}
        if self.is_logging:
            start = time.time()
            trajectory_us_value = []
            # trajectory_us_gradient = []
            trajectory_loss_objective = []
            trajectory_loss_delta_u = []
            trajectory_time = []

        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            if initial_ws is None:
                data = target[:, :history_ws.shape[1]].tolist()
                us = torch.tensor(data, device=self.device)
                us = torch.nn.Parameter(us).to(self.device)
            else:
                us = torch.nn.Parameter(initial_ws).to(self.device)
            """opt = torch.optim.LBFGS(params=[us],
                                    lr=1e-4,
                                    max_iter=2,
                                    line_search_fn='strong_wolfe',
                                    **self.opt_config)"""
            opt = torch.optim.LBFGS(params=[us],
                                    lr=1e-2,
                                    max_iter=2,
                                    **self.opt_config)
            if self.is_del_u:
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
            else:
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

            # for i in range(self.max_iter):
            for i in range(self.max_iter):
                if self.is_logging:
                    with torch.no_grad():
                        prediction = self.predict_future(history_tc, history_ws, us)
                        loss_objective = crit(prediction, target)
                        init_concat_us = torch.cat([history_ws[-1:, :], us], dim=0)
                        loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                        loss_delta_u = self.alpha * loss_delta_u
                        trajectory_us_value.append(us.data.tolist())
                        trajectory_loss_objective.append(loss_objective.item())
                        trajectory_loss_delta_u.append(loss_delta_u.item())
                opt.step(closure)
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
                if self.is_logging:
                    end = time.time()
                    trajectory_time.append(end - start)
                else:
                    log[i] = opt.state_dict()
        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.data = us.data.clamp(min=self.u_min, max=self.u_max)
        if self.is_logging:
            with torch.no_grad():
                prediction = self.predict_future(history_tc, history_ws, us)
                loss_objective = crit(prediction, target)
                init_concat_us = torch.cat([history_ws[-1:, :], us], dim=0)
                loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                loss_delta_u = self.alpha * loss_delta_u
                trajectory_us_value.append(us.data.tolist())
                trajectory_loss_objective.append(loss_objective.item())
                trajectory_loss_delta_u.append(loss_delta_u.item())
                end = time.time()
                trajectory_time.append(end - start)
            # log['trajectory_us_value'] = trajectory_us_value
            log['trajectory_loss_objective'] = trajectory_loss_objective
            log['trajectory_loss_delta_u'] = trajectory_loss_delta_u
            log['trajectory_time'] = trajectory_time
        return us, log

    def solve_mpc(self, history_tc, history_ws, target, initial_ws):
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


class PreCoTorchMPC(nn.Module):

    def __init__(self, model,
                 alpha: float = 1.0,
                 is_del_u: bool = False,
                 from_target: bool = False,
                 u_min: float = 0.0,
                 u_max: float = 1.0,
                 optimizer_mode: str = 'Adam',
                 max_iter: int = None,
                 timeout: float = 300,
                 is_logging: bool = True,
                 device: str = None,
                 opt_config: dict = {}):
        super(PreCoTorchMPC, self).__init__()
        if device is None:
            print("Running device is not given. Infer the running device from the system configuration ...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Initiating solver with {}".format(device))
        self.device = device
        self.model = model.to(device)
        self.alpha = alpha
        self.is_del_u = is_del_u
        self.from_target = from_target
        self.u_min = u_min
        self.u_max = u_max
        self.optimizer_mode = optimizer_mode
        if max_iter is None:
            # self.max_iter = 100000
            self.max_iter = 200 if self.optimizer_mode == 'Adam' else 200
        else:
            self.max_iter = max_iter
        self.timeout = timeout
        self.is_logging = is_logging
        self.opt_config = opt_config

        self.tol = 1e-5

    def compute_us_from_del_u(self, initial_u, del_u, weights):
        dus = torch.repeat_interleave(del_u.unsqueeze(dim=0), repeats=del_u.shape[0], dim=0)
        us = initial_u + torch.sum(weights * dus, dim=1)
        return us

    def solve_mpc_adam(self, history_tc, history_ws, target, initial_ws):
        """
        :param history_tc: [receding_history x state_dim]
        :param history_ws: [receding_history-1 x action_dim]
        :param target: [H x state_dim]
        :param initial_ws: [H x action_dim]
        :return:
        """
        history_tc = history_tc.unsqueeze(dim=0)  # [1 x receding_history x state_dim]
        history_ws = history_ws.unsqueeze(dim=0)  # [1 x receding_history-1 x action_dim]
        target = target.unsqueeze(dim=0)  # [1 x H x state_dim]
        initial_ws = initial_ws.unsqueeze(dim=0)  # [1 x H x action_dim]
        state_dim = target.shape[2]
        action_dim = history_ws.shape[2]
        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            start = time.time()
            crit = torch.nn.MSELoss(reduction='mean')
            log = {}
            solve = False
            if self.is_logging:
                log['history_tc'] = history_tc
                log['history_ws'] = history_ws
                log['target'] = target
                log['initial_ws'] = initial_ws
                start = time.time()
                trajectory_us_value = []
                trajectory_loss_objective = []
                trajectory_loss_delta_u = []
                trajectory_loss = []
                trajectory_time = []
            data = initial_ws.tolist()
            us = torch.nn.Parameter(torch.tensor(data, device=self.device)).to(self.device)
            us.data = us.data.clamp(min=self.u_min, max=self.u_max)
            with torch.no_grad():
                h0 = self.filter_history(history_tc, history_ws)
            # opt = torch.optim.Adam([us], lr=1e-3)
            opt = torch.optim.Adam([us], lr=0.5 * 1e-2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

            for i in range(self.max_iter):
                opt.zero_grad()
                if self.is_del_u:
                    if self.from_target:
                        computed_us = us + target[:, :, :action_dim]
                        prediction = self.predict_future(h0, computed_us)
                else:
                    computed_us = us
                    prediction = self.predict_future(h0, computed_us)
                loss_objective = crit(prediction[:, :, :state_dim], target)
                init_concat_us = torch.cat([history_ws[:, -1:, :], computed_us], dim=1)
                loss_delta_u = (init_concat_us[:, 1:, :] - init_concat_us[:, :-1, :]).pow(2)
                loss_delta_u = loss_delta_u.mean()  # it was 'sum()'
                loss_delta_u = self.alpha * loss_delta_u
                loss = loss_objective + loss_delta_u
                loss.backward()
                scheduler.step(loss)
                if self.is_logging:
                    with torch.no_grad():
                        trajectory_us_value.append(us.tolist())
                        trajectory_loss_objective.append(loss_objective.item())
                        trajectory_loss_delta_u.append(loss_delta_u.item())
                        trajectory_loss.append(loss.item())
                opt.step()
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
                if self.is_logging:
                    end = time.time()
                    trajectory_time.append(end - start)

                if loss <= self.tol:
                    solve = True
                    break

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.data = us.data.clamp(min=self.u_min, max=self.u_max)
        if self.is_logging:
            with torch.no_grad():
                if self.is_del_u:
                    if self.from_target:
                        computed_us = us + target[:, :, :action_dim]
                        prediction = self.predict_future(h0, computed_us)
                else:
                    computed_us = us
                    prediction = self.predict_future(h0, computed_us)
                loss_objective = crit(prediction[:, :, :state_dim], target)
                init_concat_us = torch.cat([history_ws[:, -1:, :], computed_us], dim=1)
                loss_delta_u = (init_concat_us[:, 1:, :] - init_concat_us[:, :-1, :]).pow(2).mean()
                loss_delta_u = self.alpha * loss_delta_u
                loss = loss_objective + loss_delta_u

                trajectory_us_value.append(us.tolist())
                trajectory_loss_objective.append(loss_objective.item())
                trajectory_loss_delta_u.append(loss_delta_u.item())
                trajectory_loss.append(loss.item())
                end = time.time()
                trajectory_time.append(end - start)
            # log['trajectory_us_value'] = trajectory_us_value
            log['trajectory_loss_objective'] = trajectory_loss_objective
            log['trajectory_loss_delta_u'] = trajectory_loss_delta_u
            log['trajectory_loss'] = trajectory_loss
            log['trajectory_time'] = trajectory_time
        # Return the best u
        idx = np.argmin(trajectory_loss)
        optimal_u = torch.tensor(trajectory_us_value[idx], device=self.device)
        if self.is_logging:
            log['optimal_us_value'] = trajectory_us_value[idx]
            log['idx_optimal_us_value'] = idx
            log['total_time'] = time.time() - start
        log['solve'] = solve
        return optimal_u[0], log

    # HAVE TO FIX

    def solve_mpc_LBFGS(self, history_tc, history_ws, target, initial_ws=None):
        """
        :param history_tc: [history_length x num_state]
        :param history_ws: [history_length-1 x num_action]
        :param target: [H x num_state]
        :param initial_ws: [H x num_action]
        :return:
        """
        crit = torch.nn.MSELoss(reduction='sum')
        log = {}
        if self.is_logging:
            start = time.time()
            trajectory_us_value = []
            # trajectory_us_gradient = []
            trajectory_loss_objective = []
            trajectory_loss_delta_u = []
            trajectory_time = []

        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            if initial_ws is None:
                data = target[:, :history_ws.shape[1]].tolist()
                us = torch.tensor(data, device=self.device)
                us = torch.nn.Parameter(us).to(self.device)
            else:
                us = torch.nn.Parameter(initial_ws).to(self.device)
            """opt = torch.optim.LBFGS(params=[us],
                                    lr=1e-4,
                                    max_iter=2,
                                    line_search_fn='strong_wolfe',
                                    **self.opt_config)"""
            opt = torch.optim.LBFGS(params=[us],
                                    lr=1e-2,
                                    max_iter=2,
                                    **self.opt_config)
            if self.is_del_u:
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
            else:
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

            # for i in range(self.max_iter):
            for i in range(self.max_iter):
                if self.is_logging:
                    with torch.no_grad():
                        prediction = self.predict_future(history_tc, history_ws, us)
                        loss_objective = crit(prediction, target)
                        init_concat_us = torch.cat([history_ws[-1:, :], us], dim=0)
                        loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                        loss_delta_u = self.alpha * loss_delta_u
                        trajectory_us_value.append(us.data.tolist())
                        trajectory_loss_objective.append(loss_objective.item())
                        trajectory_loss_delta_u.append(loss_delta_u.item())
                opt.step(closure)
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
                if self.is_logging:
                    end = time.time()
                    trajectory_time.append(end - start)
                else:
                    log[i] = opt.state_dict()
        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.data = us.data.clamp(min=self.u_min, max=self.u_max)
        if self.is_logging:
            with torch.no_grad():
                prediction = self.predict_future(history_tc, history_ws, us)
                loss_objective = crit(prediction, target)
                init_concat_us = torch.cat([history_ws[-1:, :], us], dim=0)
                loss_delta_u = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                loss_delta_u = self.alpha * loss_delta_u
                trajectory_us_value.append(us.data.tolist())
                trajectory_loss_objective.append(loss_objective.item())
                trajectory_loss_delta_u.append(loss_delta_u.item())
                end = time.time()
                trajectory_time.append(end - start)
            # log['trajectory_us_value'] = trajectory_us_value
            log['trajectory_loss_objective'] = trajectory_loss_objective
            log['trajectory_loss_delta_u'] = trajectory_loss_delta_u
            log['trajectory_time'] = trajectory_time
        return us, log

    def solve_mpc(self, history_tc, history_ws, target, initial_ws):
        opt_actions = None
        log = {}
        if self.optimizer_mode == 'Adam':
            opt_actions, log = self.solve_mpc_adam(history_tc, history_ws, target, initial_ws)
        elif self.optimizer_mode == 'LBFGS':
            opt_actions, log = self.solve_mpc_LBFGS(history_tc, history_ws, target, initial_ws)
        return opt_actions, log

    def filter_history(self, history_tc, history_ws):
        return self.model.filter_history(history_tc, history_ws)

    def predict_future(self, h, us):
        ## TODO: Design a module functions for coping w/ different model outputs.
        prediction = self.model.multi_step_prediction(h, us)
        return prediction

    def solve_max_entropy(self, graph, history):
        pass
