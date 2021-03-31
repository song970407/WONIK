import torch.nn as nn

from src.model.DiagNormalNet import ConvexDiagNormalNet, DiagNormalNet
from src.nn.MLP import MultiLayerPerceptron as MLP
from src.nn.ReparameterizedLinear import ReparameterizedLinear


def get_convex_decoder(is_deterministic, preco_hidden_dim, output_dim=1):
    if is_deterministic:
        decoder = nn.Sequential(
            ReparameterizedLinear(preco_hidden_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            ReparameterizedLinear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            ReparameterizedLinear(32, 16),
            nn.LeakyReLU(negative_slope=0.2),
            ReparameterizedLinear(16, output_dim, is_increasing=False),
            nn.LeakyReLU(negative_slope=1.5)
        )
    else:
        decoder_net = nn.Sequential(
            ReparameterizedLinear(preco_hidden_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            ReparameterizedLinear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            ReparameterizedLinear(32, 16, is_increasing=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        decoder = ConvexDiagNormalNet(decoder_net, 16, output_dim)
    return decoder


def get_decoder(is_deterministic, preco_hidden_dim, output_dim=1):
    if is_deterministic:
        decoder = MLP(preco_hidden_dim,
                      output_dim,
                      num_neurons=[64, 32, 16],
                      activation='ReLU',
                      out_activation='ReLU',
                      normalization=None)
    else:
        decoder_net = MLP(preco_hidden_dim,
                          output_dimension=16,
                          activation='ReLU',
                          num_neurons=[64, 32],
                          normalization=None,
                          out_activation=None)
        decoder = DiagNormalNet(decoder_net, 16, output_dim)
    return decoder


def get_linear_decoder(is_deterministic, preco_hidden_dim, output_dim=1):
    if is_deterministic:
        decoder = MLP(preco_hidden_dim,
                      output_dim,
                      num_neurons=[64, 32, 16],
                      activation=None,
                      out_activation=None,
                      normalization=None)
    else:
        decoder_net = MLP(preco_hidden_dim,
                          output_dimension=16,
                          activation=None,
                          out_activation=None,
                          num_neurons=[64, 32],
                          normalization=None)
        decoder = DiagNormalNet(decoder_net, 16, output_dim)
    return decoder
