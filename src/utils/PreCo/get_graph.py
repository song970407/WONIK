import dgl
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from src.graph_config import t2t, u2t


def get_hetero_graph(glass_tc_pos_path: str,
                     control_tc_pos_path: str,
                     t2t_threshold: float = 850.0,
                     u2t_threshold: float = np.inf,
                     weight: float = (1.0, 1.0, 1.0)):
    dist_func = lambda u, v: np.sqrt(((u - v) ** 2 * weight).sum())
    POS_COLS = ['Position_x', 'Position_y', 'Position_z']

    df_glass_tc = pd.read_csv(glass_tc_pos_path)
    g_tc_pos = df_glass_tc[POS_COLS].to_numpy()

    df_control_tc = pd.read_csv(control_tc_pos_path)
    c_tc_pos = df_control_tc[POS_COLS].to_numpy()

    tc_pos = np.concatenate([g_tc_pos, c_tc_pos], axis=0)

    graph_data = dict()
    # construct 'tc' to 'tc' edges
    t2t_dist_mat = cdist(tc_pos, tc_pos, dist_func)
    u, v = torch.nonzero(torch.tensor(t2t_dist_mat <= t2t_threshold).bool(),
                         as_tuple=True)
    graph_data[t2t] = (u, v)

    # construct 'control' to 'tc' edges
    c2t_dist_mat = cdist(c_tc_pos, tc_pos, dist_func)
    u, v = torch.nonzero(torch.tensor(c2t_dist_mat <= u2t_threshold).bool(),
                         as_tuple=True)
    graph_data[u2t] = (u, v)

    g = dgl.heterograph(graph_data)

    # standardize positions
    scaler = MinMaxScaler()
    pos = np.concatenate([tc_pos, c_tc_pos], axis=0)
    pos_std = scaler.fit_transform(pos)
    g.nodes['tc'].data['position'] = torch.from_numpy(pos_std[:tc_pos.shape[0], :]).float()
    g.nodes['control'].data['position'] = torch.from_numpy(pos_std[tc_pos.shape[0]:, :]).float()

    # add binary indicator for noticing the node is glass tc or not.
    is_glass_tc = torch.ones(tc_pos.shape[0], 1)
    is_glass_tc[:g_tc_pos.shape[0], :] = 0
    g.nodes['tc'].data['is-glass-tc'] = is_glass_tc
    return g
