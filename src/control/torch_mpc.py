import numpy as np
import stopit
import torch
import torch.nn as nn


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

    def solve_mpc_adam(self, graph, history, target):
        """
        Args:
            graph: DGL graph
            history: tuple (tc history, control history)
                - tc history [#.tc sensors x history length]
                - control history [#. control sensors x history length]
            target:
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
                loss = crit(prediction, target)
                loss = (loss * gamma_mask).sum()
                loss.backward()
                opt.step()
                us.us.data = us.us.data.clamp(min=0.0, max=1.0)

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.us.data = us.us.data.clamp(min=0.0, max=1.0)
        return us

    def solve_mpc_LBFGS(self, graph, history, target):
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
                    loss = crit(prediction, target)
                    loss = (loss * gamma_mask).sum()
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

    def solve_mpc(self, graph, history, target):
        if self.optimizer_mode == 'Adam':
            opt_actions = self.solve_mpc_adam(graph, history, target)
        if self.optimizer_mode == 'LBFGS':
            opt_actions = self.solve_mpc_LBFGS(graph, history, target)
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

    def solve_mpc_adam(self, history_tc, history_ws, target):
        crit = torch.nn.MSELoss()
        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            us = Linear_Actions(target[:, 0], history_ws.shape[1], self.u_min, self.u_max).to(self.device)
            opt = torch.optim.Adam(us.parameters(), lr=1e-3)
            for i in range(self.max_iter):
                opt.zero_grad()
                prediction = self.predict_future(history_tc, history_ws, us())
                loss = crit(prediction, target)
                init_concat_us = torch.cat([history_ws[-1:, :], us()], dim=0)
                delta_u_loss = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                loss = loss + self.alpha * delta_u_loss
                loss.backward()
                opt.step()
                us.us.data = us.us.data.clamp(min=self.u_min, max=self.u_max)

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.us.data = us.us.data.clamp(min=self.u_min, max=self.u_max)
        return us

    def solve_mpc_LBFGS(self, history_tc, history_ws, target):
        """
        :param history_tc: [history_length x num_state]
        :param history_ws: [history_length-1 x num_action]
        :param target: [H x num_action]
        :return:
        """
        crit = torch.nn.MSELoss()
        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            us = Linear_Actions(target[:, 0], history_ws.shape[1], self.u_min, self.u_max).to(self.device)
            opt = torch.optim.LBFGS(us.parameters(),
                                    history_size=15,
                                    max_iter=self.max_iter,
                                    line_search_fn='strong_wolfe',
                                    **self.opt_config)
            for i in range(self.max_iter):
                def closure():
                    opt.zero_grad()
                    prediction = self.predict_future(history_tc, history_ws, us())
                    loss = crit(prediction, target)
                    init_concat_us = torch.cat([history_ws[-1:, :], us()], dim=0)
                    delta_u_loss = (init_concat_us[1:, :] - init_concat_us[:-1, :]).pow(2).sum()
                    loss = loss + self.alpha * delta_u_loss
                    loss.backward()
                    return loss

                opt.step(closure)
                us.us.data = us.us.data.clamp(min=self.u_min, max=self.u_max)

        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            us.us.data = us.us.data.clamp(min=self.u_min, max=self.u_max)
        return us

    def solve_mpc(self, history_tc, history_ws, target):
        if self.optimizer_mode == 'Adam':
            opt_actions = self.solve_mpc_adam(history_tc, history_ws, target)
        if self.optimizer_mode == 'LBFGS':
            opt_actions = self.solve_mpc_LBFGS(history_tc, history_ws, target)
        return opt_actions

    def predict_future(self, history_tc, history_ws, us):
        ## TODO: Design a module functions for coping w/ different model outputs.
        prediction = self.model.multi_step_prediction(history_tc, history_ws, us)
        return prediction

    def solve_max_entropy(self, graph, history):
        pass
