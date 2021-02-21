import torch
import torch.nn as nn
import torch.nn.functional as F

TORCH_ACTIVATION_LIST = ['ReLU',
                         'Sigmoid',
                         'SELU',
                         'LeakyReLU',
                         'Softplus',
                         'Tanh']

ACTIVATION_LIST = ['Mish', 'Swish', 'Clip1', None]


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class Clip1(nn.Module):

    def forward(self, x):
        return x.clamp(0, 1)


def get_nn_activation(activation: str):
    if not activation in TORCH_ACTIVATION_LIST + ACTIVATION_LIST:
        raise RuntimeError("Not implemented activation function!")

    if activation in TORCH_ACTIVATION_LIST:
        act = getattr(torch.nn, activation)()

    if activation in ACTIVATION_LIST:
        if activation == 'Mish':
            act = Mish()
        elif activation == 'Swish':
            act = Swish()
        elif activation == 'Clip1':
            act = Clip1()
        elif activation is None:
            act = nn.Identity()

    return act


class LinearModule(nn.Module):

    def __init__(self,
                 activation: str,
                 norm: str = None,
                 dropout_p: float = 0.0,
                 weight_init: str = None,
                 use_residual: bool = False,
                 **linear_kwargs):
        super(LinearModule, self).__init__()

        if linear_kwargs['in_features'] == linear_kwargs['out_features'] and use_residual:
            self.use_residual = True
        else:
            self.use_residual = False

        # layers

        linear_layer = torch.nn.Linear(**linear_kwargs)
        self.linear_layer = linear_layer
        if dropout_p > 0.0:
            self.dropout_layer = torch.nn.Dropout(dropout_p)
        else:
            self.dropout_layer = torch.nn.Identity()
        self.activation_layer = get_nn_activation(activation)

        self.weight_init = weight_init
        self.activation = activation
        self.norm = norm

        # apply weight initialization methods
        self.apply_weight_init(self.linear_layer, self.weight_init)

        if norm == 'batch':
            self.norm_layer = torch.nn.BatchNorm1d(self.linear_layer.out_features)
        elif norm == 'layer':
            self.norm_layer = torch.nn.LayerNorm(self.linear_layer.out_features)
        elif norm == 'spectral':
            self.linear_layer = torch.nn.utils.spectral_norm(self.linear_layer)
            self.norm_layer = torch.nn.Identity()
        elif norm is None:
            self.norm_layer = torch.nn.Identity()
        else:
            raise RuntimeError("Not implemented normalization function!")

    def apply_weight_init(self, tensor, weight_init=None):
        if weight_init is None:
            pass  # do not apply weight init
        elif weight_init == "normal":
            torch.nn.init.normal_(tensor.weight, std=0.3)
            torch.nn.init.constant_(tensor.bias, 0.0)
        elif weight_init == "kaiming_normal":
            if self.activation in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
                torch.nn.init.kaiming_normal_(tensor.weight, nonlinearity=self.activation)
                torch.nn.init.constant_(tensor.bias, 0.0)
            else:
                pass
        elif weight_init == "xavier":
            torch.nn.init.xavier_uniform_(tensor.weight)
            torch.nn.init.constant_(tensor.bias, 0.0)
        else:
            raise NotImplementedError("MLP initializer {} is not supported".format(weight_init))

    def forward(self, x):
        if self.use_residual:
            input_x = x

        x = self.linear_layer(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        x = self.dropout_layer(x)

        if self.use_residual:
            x = input_x + x
        return x
