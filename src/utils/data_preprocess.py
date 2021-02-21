from typing import Union, List

import pandas as pd
import torch


def get_data(states: Union[torch.Tensor, List[torch.Tensor]],
             actions: Union[torch.Tensor, List[torch.Tensor]],
             rollout_window: int,
             history_x_window: int,
             history_u_window: int,
             glass_tc_pos_path: str,
             control_tc_pos_path: str,
             num_glass_tc: int,
             num_control_tc: int,
             device: str = 'cpu'):
    if isinstance(states, torch.Tensor):
        states = [states]
        actions = [actions]

    pos_cols = ['Position_x', 'Position_y', 'Position_z']
    df_glass = pd.read_csv(glass_tc_pos_path)
    pos_glass = df_glass[pos_cols].to_numpy(dtype='float')
    df_control = pd.read_csv(control_tc_pos_path)
    pos_control = df_control[pos_cols].to_numpy(dtype='float')
    pos_tc = torch.cat([torch.tensor(pos_glass).float(), torch.tensor(pos_control).float()], dim=0).to(device)
    for i in range(pos_tc.shape[1]):
        pos_max = pos_tc[:, i].min()
        pos_min = pos_tc[:, i].max()
        pos_tc[:, i] = (pos_tc[:, i] - pos_min) / (pos_max - pos_min)
    history_xs = []
    history_us = []
    us = []
    ys = []
    for state, action in zip(states, actions):
        num_obs = action.shape[0]
        for i in range(num_obs - rollout_window - history_u_window + 2):
            history_xs.append(state[i:i + history_x_window, :])
            history_us.append(action[i:i + history_u_window - 1, :])
            us.append(action[i + history_u_window - 1:i + history_u_window + rollout_window - 1, :])
            ys.append(state[i + history_x_window:i + history_x_window + rollout_window, :])

    history_xs = torch.stack(history_xs).transpose(1, 2).to(device)
    history_xs_glass = history_xs[:, :num_glass_tc, :]
    history_xs_control = history_xs[:, num_glass_tc:, :]
    history_us = torch.stack(history_us).transpose(1, 2).to(device)
    us = torch.stack(us).transpose(1, 2).to(device)
    ys = torch.stack(ys).transpose(1, 2).to(device)
    ys_glass = ys[:, :num_glass_tc, :]
    ys_control = ys[:, num_glass_tc:, :]
    return (history_xs_glass, history_xs_control), history_us, us, (ys_glass, ys_control), pos_tc


def get_xuy(states: Union[torch.Tensor, List[torch.Tensor]],
            actions: Union[torch.Tensor, List[torch.Tensor]],
            window_size: int,
            history_window_size: int,
            glass_tc_pos_path: str,
            control_tc_pos_path: str,
            action_dif: bool = False,
            device: str = 'cpu'):
    if isinstance(states, torch.Tensor):
        states = [states]
        actions = [actions]

    pos_cols = ['Position_x', 'Position_y', 'Position_z']
    df_glass = pd.read_csv(glass_tc_pos_path)
    pos_glass = df_glass[pos_cols].to_numpy(dtype='float')
    df_control = pd.read_csv(control_tc_pos_path)
    pos_control = df_control[pos_cols].to_numpy(dtype='float')
    pos_tc = torch.cat([torch.tensor(pos_glass).float(), torch.tensor(pos_control).float()], dim=0).to(device)
    for i in range(pos_tc.shape[1]):
        pos_max = pos_tc[:, i].min()
        pos_min = pos_tc[:, i].max()
        pos_tc[:, i] = (pos_tc[:, i] - pos_min) / (pos_max - pos_min)
    xs = []
    us = []
    ys = []
    for state, action in zip(states, actions):
        num_obs = action.shape[0]
        for i in range(num_obs - window_size + 1):
            xs.append(state[i:i + history_window_size, :])
            us.append(action[i:i + window_size, :].view(window_size, -1, 1))
            ys.append(state[i + history_window_size:i + history_window_size + window_size, :].view(window_size, -1, 1))
    xs = torch.stack(xs).to(device)
    us = torch.stack(us).to(device)
    ys = torch.stack(ys).to(device)
    if action_dif:
        for idx in range(us.shape[0]):
            with torch.no_grad():
                y = torch.cat([xs[idx, -1:, -us.shape[2]:], ys[idx, :-1, -us.shape[2]:, 0]], dim=0).unsqueeze(dim=-1)
                # us[idx] = torch.nn.functional.relu(us[idx]-y)
                us[idx] = us[idx] - y
    return xs, us, ys, pos_tc


def get_xzuy(states: torch.Tensor,
             actions: torch.Tensor,
             window_size: int,
             device: str = 'cpu'):
    """
    :param states: [time stamps x state dim]
    :param actions: [time stamps x action dim]
    :param window_size:
    :return:
    """
    assert states.shape[0] == actions.shape[0], "states and actions do not have the same time stamps"
    num_obs = states.shape[0]

    xs = []
    zs = []
    us = []
    ys = []
    for i in range(num_obs - window_size):
        xs.append(states[i, :])
        zs.append(torch.tensor([i]))
        us.append(actions[i + 1:i + window_size + 1, :])
        ys.append(states[i + 1:i + window_size + 1, :])

    xs = torch.stack(xs).to(device)
    zs = torch.stack(zs).to(device).long()
    us = torch.stack(us).to(device)
    ys = torch.stack(ys).to(device)
    return xs, zs, us, ys


def get_xzuy_context(states: torch.Tensor,
                     actions: torch.Tensor,
                     window_size: int,
                     z_max: int,
                     device: str = 'cpu'):
    """
    :param states: [time stamps x state dim]
    :param actions: [time stamps x action dim]
    :param window_size:
    :return:
    """
    assert states.shape[0] == actions.shape[0], "states and actions do not have the same time stamps"
    num_obs = states.shape[0]

    xs = []
    zs = []
    us = []
    ys = []
    for i in range(num_obs - window_size):
        xs.append(states[i, :])
        zs.append(torch.tensor([i / z_max]))
        us.append(actions[i + 1:i + window_size + 1, :])
        ys.append(states[i + 1:i + window_size + 1, :])

    xs = torch.stack(xs).to(device)
    zs = torch.stack(zs).to(device)
    us = torch.stack(us).to(device)
    ys = torch.stack(ys).to(device)

    return xs, zs, us, ys


def get_xuy_gat(states: torch.Tensor,
                actions: torch.Tensor,
                window_size: int,
                glass_tc_pos_path: str,
                control_tc_pos_path: str,
                device: str = 'cpu'):
    num_obs = states.shape[0]
    pos_cols = ['Position_x', 'Position_y', 'Position_z']
    df_glass = pd.read_csv(glass_tc_pos_path)
    pos_glass = df_glass[pos_cols].to_numpy(dtype='float')
    df_control = pd.read_csv(control_tc_pos_path)
    pos_control = df_control[pos_cols].to_numpy(dtype='float')
    pos_tc = torch.cat([torch.tensor(pos_glass).float(), torch.tensor(pos_control).float()], dim=0).to(device)
    for i in range(pos_tc.shape[1]):
        pos_max = pos_tc[:, i].min()
        pos_min = pos_tc[:, i].max()
        pos_tc[:, i] = (pos_tc[:, i] - pos_min) / (pos_max - pos_min)
    xs = []
    us = []
    ys = []
    for i in range(num_obs - window_size):
        xs.append(states[i, :].view((-1, 1)))
        us.append(actions[i + 1:i + window_size + 1, :].view(window_size, -1, 1))
        ys.append(states[i + 1:i + window_size + 1, :].view(window_size, -1, 1))
    xs = torch.stack(xs).to(device)
    us = torch.stack(us).to(device)
    ys = torch.stack(ys).to(device)

    return xs, us, ys, pos_tc


def get_xuy_icgrnn(states: torch.Tensor,
                   actions: torch.Tensor,
                   window_size: int,
                   history_window_size: int,
                   glass_tc_pos_path: str,
                   control_tc_pos_path: str,
                   device: str = 'cpu'):
    num_obs = actions.shape[0]
    pos_cols = ['Position_x', 'Position_y', 'Position_z']
    df_glass = pd.read_csv(glass_tc_pos_path)
    pos_glass = df_glass[pos_cols].to_numpy(dtype='float')
    df_control = pd.read_csv(control_tc_pos_path)
    pos_control = df_control[pos_cols].to_numpy(dtype='float')
    pos_tc = torch.cat([torch.tensor(pos_glass).float(), torch.tensor(pos_control).float()], dim=0).to(device)
    for i in range(pos_tc.shape[1]):
        pos_max = pos_tc[:, i].min()
        pos_min = pos_tc[:, i].max()
        pos_tc[:, i] = (pos_tc[:, i] - pos_min) / (pos_max - pos_min)
    xs = []
    hs = []
    us = []
    ys = []
    for i in range(num_obs - window_size):
        xs.append(states[i + history_window_size, :].view((-1, 1)))
        hs.append(states[i:i + history_window_size, :])
        us.append(actions[i + 1:i + window_size + 1, :].view(window_size, -1, 1))
        ys.append(
            states[i + history_window_size + 1: i + history_window_size + window_size + 1, :].view(window_size, -1, 1))
    xs = torch.stack(xs).to(device)
    hs = torch.stack(hs).to(device)
    us = torch.stack(us).to(device)
    ys = torch.stack(ys).to(device)

    return xs, hs, us, ys, pos_tc


def get_xuy_icgatrnn(states: torch.Tensor,
                     actions: torch.Tensor,
                     window_size: int,
                     history_window_size: int,
                     glass_tc_pos_path: str,
                     control_tc_pos_path: str,
                     device: str = 'cpu'):
    num_obs = actions.shape[0]
    pos_cols = ['Position_x', 'Position_y', 'Position_z']
    df_glass = pd.read_csv(glass_tc_pos_path)
    pos_glass = df_glass[pos_cols].to_numpy(dtype='float')
    df_control = pd.read_csv(control_tc_pos_path)
    pos_control = df_control[pos_cols].to_numpy(dtype='float')
    pos_tc = torch.cat([torch.tensor(pos_glass).float(), torch.tensor(pos_control).float()], dim=0).to(device)
    for i in range(pos_tc.shape[1]):
        pos_max = pos_tc[:, i].min()
        pos_min = pos_tc[:, i].max()
        pos_tc[:, i] = (pos_tc[:, i] - pos_min) / (pos_max - pos_min)
    xs = []
    us = []
    ys = []
    for i in range(num_obs - window_size + 1):
        xs.append(states[i:i + history_window_size, :])
        us.append(actions[i:i + window_size, :].view(window_size, -1, 1))
        ys.append(states[i + history_window_size:i + history_window_size + window_size, :].view(window_size, -1, 1))
    xs = torch.stack(xs).to(device)
    us = torch.stack(us).to(device)
    ys = torch.stack(ys).to(device)

    return xs, us, ys, pos_tc


def get_xuys_icgatrnn(states: Union[torch.Tensor, List[torch.Tensor]],
                      actions: Union[torch.Tensor, List[torch.Tensor]],
                      window_size: int,
                      history_window_size: int,
                      glass_tc_pos_path: str,
                      control_tc_pos_path: str,
                      device: str = 'cpu'):
    if isinstance(states, torch.Tensor):
        states = [states]
        actions = [actions]

    pos_cols = ['Position_x', 'Position_y', 'Position_z']
    df_glass = pd.read_csv(glass_tc_pos_path)
    pos_glass = df_glass[pos_cols].to_numpy(dtype='float')
    df_control = pd.read_csv(control_tc_pos_path)
    pos_control = df_control[pos_cols].to_numpy(dtype='float')
    pos_tc = torch.cat([torch.tensor(pos_glass).float(), torch.tensor(pos_control).float()], dim=0).to(device)
    for i in range(pos_tc.shape[1]):
        pos_max = pos_tc[:, i].min()
        pos_min = pos_tc[:, i].max()
        pos_tc[:, i] = (pos_tc[:, i] - pos_min) / (pos_max - pos_min)
    xs = []
    us = []
    ys = []
    for state, action in zip(states, actions):
        num_obs = action.shape[0]
        for i in range(num_obs - window_size + 1):
            xs.append(state[i:i + history_window_size, :])
            us.append(action[i:i + window_size, :].view(window_size, -1, 1))
            ys.append(state[i + history_window_size:i + history_window_size + window_size, :].view(window_size, -1, 1))
    xs = torch.stack(xs).to(device)
    us = torch.stack(us).to(device)
    ys = torch.stack(ys).to(device)

    return xs, us, ys, pos_tc
