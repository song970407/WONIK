import torch
import torch.nn as nn
from typing import Union, List
from src.model.GraphStateSpaceModels import GraphSSM_GAT
from src.model.HeteroGraphSSM import HeteroGraphSSM
from src.model.LinearStateSpaceModels import MultiLinearSSM, ReparamMultiLinearSSM, MultiLinearResSSM
from src.model.PreCo import PreCo
from src.model.distance_kernel import RBFKernel
from src.nn.GraphModules.ConvexModule import ConvexLinear, ConvexGATConv
from src.nn.MLP import MultiLayerPerceptron as MLP


class DummyEnc(nn.Module):

    def __init__(self, hidden_dim):
        super(DummyEnc, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, xs):
        return torch.ones(xs.shape[0], self.hidden_dim).to(xs.device)


class TCN(nn.Module):

    def __init__(self,
                 initial_channel=1,
                 hidden_channel=8,
                 out_channel=64):
        super(TCN, self).__init__()
        self.layers = nn.Sequential(nn.Conv1d(initial_channel, hidden_channel, 19, stride=9),
                                    nn.ReLU(),
                                    nn.Conv1d(hidden_channel, out_channel, 10),
                                    nn.ReLU())

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.layers(x).squeeze(dim=-1)
        return -x


def get_model(enc_tcn: bool = False,
              hidden_reparam_method: Union[str, List[str]] = 'Softmax',
              dec_reparam_method: Union[str, List[str]] = 'ReLU',
              is_residual: bool = False,
              history_len: int = 10,
              use_dummy_enc: bool = False):
    history_dim = 1
    history_len = history_len
    transition_hidden_dim = 64

    # encoder

    if use_dummy_enc:
        enc = DummyEnc(transition_hidden_dim)
    elif enc_tcn:
        enc_hidden_dim = 8
        enc = TCN(1, enc_hidden_dim, transition_hidden_dim)
    else:
        enc_hidden_dims = [128, 128]
        enc = MLP(input_dimension=history_dim * history_len,
                  output_dimension=transition_hidden_dim,
                  num_neurons=enc_hidden_dims,
                  activation='LeakyReLU',
                  out_activation='LeakyReLU')

    # transition model
    control_dim = 1
    pos_dim = 3  # will be fixed
    trans = ConvexGATConv(in_feats=[transition_hidden_dim + control_dim],
                          attn_feats=[pos_dim],
                          out_feats=transition_hidden_dim,
                          num_heads=[1],
                          is_increasings=[True],
                          reparam_methods=hidden_reparam_method,
                          is_convex=True)

    # decoder
    dec_hidden_dims = [32, 32]
    dec_out_dim = 1

    dec_layers = []
    dec_layers.append(ConvexLinear(in_features=transition_hidden_dim,
                                   out_features=dec_hidden_dims[0],
                                   reparam_methods=dec_reparam_method))
    for i in range(len(dec_hidden_dims) - 1):
        l = ConvexLinear(in_features=dec_hidden_dims[i],
                         out_features=dec_hidden_dims[i + 1],
                         reparam_methods=dec_reparam_method)
        dec_layers.append(l)
    dec_layers.append(
        ConvexLinear(in_features=dec_hidden_dims[-1],
                     out_features=dec_out_dim,
                     reparam_methods=dec_reparam_method,
                     is_increasings=False, is_convex=False))

    dec = nn.Sequential(*tuple(dec_layers))

    m = GraphSSM_GAT(enc, trans, dec, is_residual=is_residual)
    return m


def get_hetero_model(control_reparam_method: Union[str, List[str]] = 'ReLU',
                     dec_reparam_method: Union[str, List[str]] = 'ReLU',
                     history_x_len: int = 10,
                     history_u_len: int = 10):
    history_dim = 1
    u_dim = 1
    transition_hidden_dim = 32
    dec_out_dim = 1

    alpha_enc = RBFKernel()
    beta_enc = RBFKernel()

    # hist_enc
    enc_hidden_dims = [64, 64]
    hist_enc = MLP(input_dimension=history_dim * history_x_len,
                   output_dimension=transition_hidden_dim,
                   num_neurons=enc_hidden_dims,
                   activation='LeakyReLU',
                   out_activation='LeakyReLU')

    # control_to_hidden
    c2h_hidden_dims = [64, 64]
    c2h_layers = [ConvexLinear(in_features=u_dim * history_u_len,
                               out_features=c2h_hidden_dims[0],
                               reparam_methods=control_reparam_method)]
    for i in range(len(c2h_hidden_dims) - 1):
        c2h_layers.append(ConvexLinear(in_features=c2h_hidden_dims[i],
                                       out_features=c2h_hidden_dims[i + 1],
                                       reparam_methods=control_reparam_method))
    c2h_layers.append(ConvexLinear(in_features=c2h_hidden_dims[-1],
                                   out_features=transition_hidden_dim,
                                   reparam_methods=control_reparam_method,
                                   is_increasings=False))
    control_to_hidden = nn.Sequential(*tuple(c2h_layers))

    # decoder
    h2x_hidden_dims = [16, 16]
    h2x_layers = [ConvexLinear(in_features=transition_hidden_dim,
                               out_features=h2x_hidden_dims[0],
                               reparam_methods=dec_reparam_method)]
    for i in range(len(h2x_hidden_dims) - 1):
        h2x_layers.append(ConvexLinear(in_features=h2x_hidden_dims[i],
                                       out_features=h2x_hidden_dims[i + 1],
                                       reparam_methods=dec_reparam_method))
    h2x_layers.append(ConvexLinear(in_features=h2x_hidden_dims[-1],
                                   out_features=dec_out_dim,
                                   reparam_methods=dec_reparam_method,
                                   is_increasings=False, is_convex=False))

    hidden_to_tc = nn.Sequential(*tuple(h2x_layers))
    m = HeteroGraphSSM(alpha_enc, beta_enc, hist_enc, control_to_hidden, hidden_to_tc)
    return m


def get_multi_linear_model(state_dim, action_dim, state_order, action_order):
    m = MultiLinearSSM(state_dim, action_dim, state_order, action_order)
    return m


def get_multi_linear_residual_model(state_dim, action_dim, state_order, action_order):
    m = MultiLinearResSSM(state_dim, action_dim, state_order, action_order)
    return m


def get_reparam_multi_linear_model(state_dim, action_dim, state_order, action_order):
    m = ReparamMultiLinearSSM(state_dim, action_dim, state_order, action_order)
    return m


def get_preco_model1(state_dim, action_dim, hidden_dim=256, hidden_mlp_dim=256):
    pred = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_mlp_dim),
                         nn.Tanh(),
                         nn.Linear(hidden_mlp_dim, hidden_dim),
                         nn.Tanh())
    corr = nn.Sequential(nn.Linear(hidden_dim + state_dim, hidden_mlp_dim),
                         nn.Tanh(),
                         nn.Linear(hidden_mlp_dim, hidden_dim),
                         nn.Tanh())
    dec = nn.Sequential(nn.Linear(hidden_dim, hidden_mlp_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_mlp_dim, state_dim))
    m = PreCo(pred, corr, dec, state_dim, action_dim, hidden_dim)
    return m

name_model_dict={
    'PreCo1': get_preco_model1
}


def get_preco_model(model_name, load_saved, state_dim, action_dim, hidden_dim, hidden_mlp_dim=256):
    m = name_model_dict[model_name](state_dim, action_dim, hidden_dim, hidden_mlp_dim)
    if load_saved:
        m.load_state_dict(torch.load('model/{}.pt'.format(model_name), map_location='cpu'))
    return m


if __name__ == '__main__':
    m = get_model()
    print(m)
