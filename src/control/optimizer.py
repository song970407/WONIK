import torch
import torch.nn as nn
import torch.nn.init as init


class MPCController(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 action_dim: int,
                 H: int,
                 max_iters: int):
        super(MPCController, self).__init__()

        self.model = model
        self.us = nn.Parameter(torch.Tensor(action_dim, H))
        self.H = H
        self.max_iters = max_iters
        self.opt = torch.optim.SGD(self.us, lr=1e-3)  # consider to solve convex opt problems.
        # TODO: check whether adam works well or BTLS + GD works well.

    def post_process_us(self):
        self.us.clamp(min=0.0, max=1.0)

    def reset_parameters(self):
        init.uniform_(self._weight)

    def step(self, history,
             state_reference,
             prev_action=None):
        for i in range(self.max_iters):
            state_prediction = self.model(self.us, history)
            # TODO 1: implement loss function
            # TODO 2: Backtracking line search?
            loss = ''
            loss.backward()
            self.opt.step()
            self.post_process_us()

        opt_us = self.us.item()
        return opt_us
