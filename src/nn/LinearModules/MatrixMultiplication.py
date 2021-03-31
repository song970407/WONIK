import math

import torch
import torch.nn as nn


class MatrixMultiplication(nn.Module):
    """
        batch operation supporting matrix multiplication layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 adjacent_mat: torch.Tensor = None):
        super(MatrixMultiplication, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

        if adjacent_mat is None:
            adjacent_mat = torch.ones(in_features, out_features)
        self.register_buffer('adjacent_mat', adjacent_mat)

        if adjacent_mat is not None:
            assert self.weight.shape == self.adjacent_mat.shape

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def reset_parameters_new(self, val):
        nn.init.constant_(self.weight, val)

    def clip_params(self, min, max):
        self.weight.data = torch.clamp(self.weight.data, min, max)

    def forward(self, input):
        if self.adjacent_mat is None:
            weight = self.weight
        else:
            weight = self.adjacent_mat * self.weight

        return torch.einsum('bx, xy -> by', input, weight)


class DiagonalMatrix(nn.Module):

    def __init__(self, n: int, clipping_val):
        super(DiagonalMatrix, self).__init__()

        self.weight = nn.Parameter(torch.ones(n, ))
        self.clipping_val = clipping_val

    def clip_params(self, min=1e-10, max=None):
        if max is None:
            max = self.clipping_val
        self.weight.data = self.weight.clamp(min, max)

    def forward(self, x):
        diag_mat = torch.diag(self.weight)
        return x.matmul(diag_mat)
