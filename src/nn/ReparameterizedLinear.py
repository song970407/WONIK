import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.parameter import Parameter


class ReparameterizedLinear(nn.Linear):
    """
    To make every weight parameter non-negative (or non-positive) on torch.nn.linear by using reparameterization trick
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 is_increasing: bool = True,
                 reparam_method: str = 'ReLU'):
        super(ReparameterizedLinear, self).__init__(in_features=in_features,
                                                    out_features=out_features,
                                                    bias=bias)
        self._in_dim = in_features
        self._out_dim = out_features
        self._use_bais = bias
        self.reparam_method = reparam_method
        self.is_increasing = is_increasing
        delattr(self, 'weight')
        self._weight = Parameter(torch.Tensor(out_features, in_features))
        if reparam_method == 'Expo':
            init.constant_(self._weight, math.log(1/(in_features + out_features)))
        else:
            init.uniform_(self._weight, b=2 / (in_features + out_features))

    @property
    def weight(self):
        if self.reparam_method == 'ReLU':
            if self.is_increasing:
                ret = F.relu(self._weight)
            else:
                ret = -1 * F.relu(self._weight)
        elif self.reparam_method == 'Softmax':
            if self.is_increasing:
                ret = F.softmax(self._weight, dim=1)
            else:
                ret = -F.softmax(self._weight, dim=1)
        elif self.reparam_method == 'ReLUnorm':
            if self.is_increasing:
                ret = F.relu(self._weight) / torch.unsqueeze(torch.sum(F.relu(self._weight), dim=1) + 1e-8, dim=1)
            else:
                ret = - F.relu(self._weight) / torch.unsqueeze(torch.sum(F.relu(self._weight), dim=1) + 1e-8, dim=1)
        elif self.reparam_method == 'Expo':
            if self.is_increasing:
                ret = torch.exp(self._weight)
            else:
                ret = - torch.exp(self._weight)
        elif self.reparam_method is None:
            ret = self._weight
        else:
            raise NotImplementedError('{} is not implemented'.format(self.reparam_method))
        return ret

    def __repr__(self):
        msg = '\n'
        msg += "Reparameterized Linear \n"
        msg += "Input dim : {} \n".format(self._in_dim)
        msg += "Output dim : {} \n".format(self._out_dim)
        msg += "Use bias : {} \n".format(self._use_bais)
        msg += "is increasing : {} \n".format(self.is_increasing)
        msg += "Reparametrization method : {}".format(self.reparam_method)
        return msg

if __name__ == '__main__':
    l = ReparameterizedLinear(2, 3)
    print(l)
