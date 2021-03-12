import dgl
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from src.graph_config import g2g, g2c, c2g, c2c, u2g, u2c
from src.utils.adjacent_mat_utils import get_adj_matrix



def get_graph(glass_tc_pos_path,
              control_tc_pos_path,
              threshold,
              weight,
              only_control: bool = True):
    adj_matrix, pos_glass, pos_control = get_adj_matrix(glass_tc_pos_path,
                                                        control_tc_pos_path,
                                                        threshold,
                                                        weight,
                                                        only_control)
    num_state_node = pos_glass.shape[0]
    total_node = adj_matrix.shape[0]

    u = torch.nonzero(adj_matrix)[:, 0]
    v = torch.nonzero(adj_matrix)[:, 1]

    # 0th node ~ 'state dim-1'th node : state node (glass TC)
    # Others : action node (control TC)
    g = dgl.graph((u, v), num_nodes=total_node)
    g = dgl.add_self_loop(g)

    # Add control node flags
    is_control = torch.zeros(total_node, 1)
    is_control[num_state_node:, 0] = 1
    g.ndata['is_control'] = is_control

    # handling positional information.
    scaler = MinMaxScaler()
    pos = np.concatenate([pos_glass, pos_control], axis=0)
    pos_std = scaler.fit_transform(pos)
    g.ndata['position'] = torch.from_numpy(pos_std).float()
    return g


def get_hetero_graph(glass_tc_pos_path: str,
                     control_tc_pos_path: str,
                     threshold: float = 850.0,
                     weight: float = (1.0, 1.0, 1.0)):

    dist_func = lambda u, v: np.sqrt(((u - v) ** 2 * weight).sum())
    POS_COLS = ['Position_x', 'Position_y', 'Position_z']

    df_glass_tc = pd.read_csv(glass_tc_pos_path)
    g_tc_pos = df_glass_tc[POS_COLS].to_numpy()

    df_control_tc = pd.read_csv(control_tc_pos_path)
    c_tc_pos = df_control_tc[POS_COLS].to_numpy()

    graph_data = dict()
    # construct 'glass-tc' to 'glass-tc' edges
    g2g_dist_mat = cdist(g_tc_pos, g_tc_pos, dist_func)
    u, v = torch.nonzero(torch.tensor(g2g_dist_mat <= threshold).bool(),
                         as_tuple=True)
    graph_data[g2g] = (u, v)

    # construct 'glass-tc' to 'control-tc' edges
    g2c_dist_mat = cdist(g_tc_pos, c_tc_pos, dist_func)
    u, v = torch.nonzero(torch.tensor(g2c_dist_mat <= threshold).bool(),
                         as_tuple=True)
    graph_data[g2c] = (u, v)

    # construct 'control-tc' to 'glass-tc' edges
    c2g_dist_mat = cdist(c_tc_pos, g_tc_pos, dist_func)
    u, v = torch.nonzero(torch.tensor(c2g_dist_mat <= threshold).bool(),
                         as_tuple=True)
    graph_data[c2g] = (u, v)

    # construct 'control-tc' to 'control-tc' edges
    c2c_dist_mat = cdist(c_tc_pos, c_tc_pos, dist_func)
    u, v = torch.nonzero(torch.tensor(c2c_dist_mat <= threshold).bool(),
                         as_tuple=True)
    graph_data[c2c] = (u, v)

    # construct 'control' to 'glass-tc' edges
    cntl2g_dist_mat = cdist(c_tc_pos, g_tc_pos, dist_func)
    u, v = torch.nonzero(torch.tensor(cntl2g_dist_mat <= threshold).bool(),
                         as_tuple=True)

    graph_data[u2g] = (u, v)

    # construct 'control' to 'control-tc' edges
    cntl2c_dist_mat = cdist(c_tc_pos, c_tc_pos, dist_func)
    u, v = torch.nonzero(torch.tensor(cntl2c_dist_mat <= threshold).bool(),
                         as_tuple=True)
    graph_data[u2c] = (u, v)

    g = dgl.heterograph(graph_data)
    g.nodes['glass-tc'].data['position'] = torch.tensor(g_tc_pos).float()
    g.nodes['control-tc'].data['position'] = torch.tensor(c_tc_pos).float()
    g.nodes['control'].data['position'] = torch.tensor(c_tc_pos).float()
    return g
