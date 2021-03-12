import numpy as np
import pandas as pd
import torch


def _measure_distance(w, x1, x2):
    dis = w * (x1 - x2)
    dis = np.sqrt(np.sum(np.square(dis)))
    return dis


def get_adj_matrix(glass_tc_pos_path: str,
                   control_tc_pos_path: str,
                   threshold: float,
                   weight: tuple = (1, 1, 1),
                   only_control: bool = True):
    pos_cols = ['Position_x', 'Position_y', 'Position_z']
    df_glass = pd.read_csv(glass_tc_pos_path)
    pos_glass = df_glass[pos_cols].to_numpy()
    df_control = pd.read_csv(control_tc_pos_path)
    pos_control = df_control[pos_cols].to_numpy()
    weight = np.array(weight)
    pos = np.concatenate([pos_glass, pos_control], axis=0)
    num_glass = df_glass.shape[0]
    num_control = df_control.shape[0]
    A = torch.zeros((num_glass + num_control, num_glass + num_control))

    for i in range(num_glass + num_control):
        if only_control and i < num_glass:
            continue
        for j in range(num_glass + num_control):
            dist = _measure_distance(weight, pos[i], pos[j])
            if dist <= threshold:
                A[i, j] = 1
            # Old graph if you add this constraint
            """if i != j and j >= num_glass:
                A[i, j] = 0"""

    return A, pos_glass, pos_control


def get_context_adj_matrix(glass_tc_pos_path: str,
                           control_tc_pos_path: str,
                           num_context: int,
                           threshold: float,
                           weight: tuple = (1, 1, 1)):
    pos_cols = ['Position_x', 'Position_y', 'Position_z']
    df_glass = pd.read_csv(glass_tc_pos_path)
    pos_glass = df_glass[pos_cols].to_numpy()
    df_control = pd.read_csv(control_tc_pos_path)
    pos_control = df_control[pos_cols].to_numpy()
    weight = np.array(weight)

    num_glass = df_glass.shape[0]
    num_control = df_control.shape[0]

    A = torch.zeros((num_context, num_control, num_glass))
    for z in range(num_context):
        for i in range(num_control):
            for j in range(num_glass):
                dist = _measure_distance(weight, pos_control[i], pos_glass[j])
                if dist <= threshold:
                    A[z, i, j] = 1
    return A
