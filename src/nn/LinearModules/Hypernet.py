import math

import torch
import torch.nn as nn


class HyperMatrix(nn.Module):

    def __init__(self,
                 output_dim: int,
                 net: nn.Module,
                 output_matrix: torch.Tensor = None):
        super(HyperMatrix, self).__init__()

        if output_matrix is not None:
            _use_matrix_for_output_shaping = True
            self._nonzero_idx = torch.nonzero(output_matrix, as_tuple=True)
        else:
            _use_matrix_for_output_shaping = False
            self._nonzero_idx = None

        self.output_dim = output_dim
        self.net = net

        self.register_buffer('output_matrix', output_matrix)
        self._use_matrix_for_output_shaping = _use_matrix_for_output_shaping

    def forward(self, x, z):
        device = x.device
        batch_size, input_dim = x.shape[0], x.shape[1]
        mat_entity = self.net(z)  # [batch x #. entity]

        # reshaping 'mat_entity'
        if self._use_matrix_for_output_shaping:
            mat = torch.zeros(batch_size, self.output_dim, input_dim).float().to(device)
            mat[:, self._nonzero_idx[1], self._nonzero_idx[0]] = mat_entity
        else:
            mat = mat_entity.view(batch_size, self.output_dim, input_dim)

        ret = torch.einsum('boi, bi -> bo', mat, x)
        return ret


class ContextHyperMatrix(nn.Module):
    """
        batch operation supporting matrix multiplication layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_context: int,
                 adjacent_mat: torch.Tensor = None):
        super(ContextHyperMatrix, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_context = num_context
        self.weight = nn.Parameter(torch.Tensor(num_context, in_features, out_features))
        self.reset_parameters()

        self.register_buffer('adjacent_mat', adjacent_mat)

        if adjacent_mat is not None:
            assert self.weight.shape == self.adjacent_mat.shape

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def clip_params(self, min, max):
        self.weight.data = torch.clamp(self.weight.data, min, max)

    def forward(self, input, context):
        if self.adjacent_mat is None:
            weight = self.weight
        else:
            weight = self.adjacent_mat * self.weight
        return torch.einsum('bx, bxy -> by', input, weight[context[:,0],:,:])