import torch
import torch.nn as nn


class RBFKernel(nn.Module):
    def __init__(self, dim=3):
        super(RBFKernel, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.FloatTensor(size=(1, dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)

    def forward(self, x, y):
        weight = nn.functional.softplus(self.weight)
        weighted_dist = ((x - y).pow(2) * weight).sum(dim=1, keepdim=True)
        dist = torch.exp(-1 * weighted_dist)
        return dist
