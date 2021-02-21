import re
from functools import reduce
from typing import Union, List, Tuple

import pandas as pd
import torch


def _preprocess_state_and_action(state, action,
                                 history_len, future_len,
                                 hx, fx, hu, fu):
    # prepare state histories and futures
    state_start_i, state_end_i = history_len, state.shape[0] - future_len
    for i in range(state_start_i, state_end_i):
        hx.append(state[i - history_len:i])
        fx.append(state[i: i + future_len])

    for j in range(state_start_i, state_end_i):
        hu.append(action[j - history_len:j - 1])
        fu.append(action[j - 1: j + (future_len - 1)])


def load_data(paths: Union[str, List[str]],
              history_len: int,
              future_len: int,
              scaling: bool = True,
              state_scaler: Tuple[float, float] = None,
              action_scaler: Tuple[float, float] = None,
              action_ws: bool = True,
              step_names: List[str] = ['375H', '375S', '375A'],
              step_names_mode: str = 'or'):
    """
    Args:
        paths: List of data paths
        history_len:
        future_len:
        scaling: whether perform minmax scaling or not
        state_scaler: if given, use the 'state_scaler' values for minmax scaling state values
        action_scaler:  if given, use the 'action_scaler' values for minmax scaling action values
        action_ws: if True, use 'work-set' values as action inputs. if False, use control-tc values as action inputs
        step_names: if given use data only from the given 'step_name'
        step_names_mode: if 'or', consider the input 'step_names' as consecutive steps.

    Returns:

    """

    paths = [paths] if isinstance(paths, str) else paths

    hist_x, future_x = [], []
    hist_u, future_u = [], []

    for path in paths:
        df = pd.read_csv(path)
        l = df.columns.to_list()
        action_rex = 'Z.+_WS$' if action_ws else 'Z.+_PWR$'
        action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
        glass_cols = [string for string in l if re.search(re.compile('TC_Data_.+', flags=re.IGNORECASE), string)]
        control_cols = [string for string in l if re.search(re.compile('Z.+_TC$'), string)]
        state_cols = glass_cols + control_cols
        state_df = df[state_cols]
        state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
        action_df = df[action_cols]
        action_nan_cols = action_df.columns[action_df.isna().any()].tolist()

        assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."
        assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

        if step_names is None:
            state = torch.from_numpy(state_df.to_numpy()).float()  # [time stamp x #. states]
            action = torch.from_numpy(action_df.to_numpy()).float()  # [time_stamp x  #. controls]
            _preprocess_state_and_action(state, action, history_len, future_len,
                                         hist_x, future_x, hist_u, future_u)

        else:
            step_names = [step_names] if isinstance(step_names, str) else step_names
            if step_names_mode == 'or':
                conds = [df['Step_Name'] == step_name for step_name in step_names]
                compute_or = lambda x, y: x | y
                cur_step_idx = reduce(compute_or, conds)
                _state_df = state_df.loc[cur_step_idx]
                _action_df = action_df.loc[cur_step_idx]
                if _state_df.shape[0] > history_len + future_len:
                    # if only the certain step has enough number of data.
                    state = torch.from_numpy(_state_df.to_numpy()).float()  # [time stamp x #. states]
                    action = torch.from_numpy(_action_df.to_numpy()).float()  # [time_stamp x  #. controls]
                    _preprocess_state_and_action(state, action, history_len, future_len,
                                                 hist_x, future_x, hist_u, future_u)
            else:
                for step_name in step_names:
                    cur_step_idx = df['Step_Name'] == step_name
                    _state_df = state_df.loc[cur_step_idx]
                    _action_df = action_df.loc[cur_step_idx]
                    if _state_df.shape[0] > history_len + future_len:
                        # if only the certain step has enough number of data.
                        state = torch.from_numpy(_state_df.to_numpy()).float()  # [time stamp x #. states]
                        action = torch.from_numpy(_action_df.to_numpy()).float()  # [time_stamp x  #. controls]
                        _preprocess_state_and_action(state, action, history_len, future_len,
                                                     hist_x, future_x, hist_u, future_u)

    hist_x = torch.stack(hist_x).transpose(2, 1)
    hist_x = hist_x.unsqueeze(dim=-1)  # [batch x state nodes x history_len x state dim (=1)]
    future_x = torch.stack(future_x).transpose(2, 1)
    future_x = future_x.unsqueeze(dim=-1)  # [batch x state nodes x future_len x state dim (=1)]

    hist_u = torch.stack(hist_u).transpose(2, 1)
    hist_u = hist_u.unsqueeze(dim=-1)  # [batch x state nodes x history_len x control dim (=1)]
    future_u = torch.stack(future_u).transpose(2, 1)
    future_u = future_u.unsqueeze(dim=-1)  # [batch x state nodes x future_len x control dim (=1)]

    info = dict()
    if scaling:
        # scaling states
        if state_scaler is None:
            state_min = min(hist_x.min(), future_x.min())
            state_max = max(hist_x.max(), future_x.max())
        else:
            state_min, state_max = state_scaler[0], state_scaler[1]

        hist_x = (hist_x - state_min) / (state_max - state_min)
        future_x = (future_x - state_min) / (state_max - state_min)

        # scaling controls
        if action_scaler is None:
            action_min = min(hist_u.min(), future_u.min())
            action_max = max(hist_u.max(), future_u.max())
        else:
            action_min, action_max = action_scaler[0], action_scaler[1]

        hist_u = (hist_u - action_min) / (action_max - action_min)
        future_u = (future_u - action_min) / (action_max - action_min)

        info['state_scaler'] = (state_min, state_max)
        info['action_scaler'] = (action_min, action_max)

    return hist_x, future_x, hist_u, future_u, info


def load_validate_data(paths: Union[str, List[str]],
                       history_len: int,
                       future_len: int,
                       scaling: bool = True,
                       state_scaler: Tuple[float, float] = None,
                       action_scaler: Tuple[float, float] = None,
                       action_ws: bool = True,
                       step_names_mode: str = 'or'):
    """
    Args:
        paths: List of data paths
        history_len:
        future_len:
        scaling: whether perform minmax scaling or not
        state_scaler: if given, use the 'state_scaler' values for minmax scaling state values
        action_scaler:  if given, use the 'action_scaler' values for minmax scaling action values
        action_ws: if True, use 'work-set' values as action inputs. if False, use control-tc values as action inputs
        step_names: if given use data only from the given 'step_name'
        step_names_mode: if 'or', consider the input 'step_names' as consecutive steps.

    Returns:

    """

    paths = [paths] if isinstance(paths, str) else paths

    hist_x, future_x = [], []
    hist_u, future_u = [], []

    for path in paths:
        df = pd.read_csv(path)
        l = df.columns.to_list()
        action_rex = 'Z.+_WS$' if action_ws else 'Z.+_PWR$'
        action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
        glass_cols = [string for string in l if re.search(re.compile('TC_Data_.+', flags=re.IGNORECASE), string)]
        control_cols = [string for string in l if re.search(re.compile('Z.+_TC$'), string)]
        state_cols = glass_cols + control_cols
        state_df = df[state_cols]
        state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
        action_df = df[action_cols]
        action_nan_cols = action_df.columns[action_df.isna().any()].tolist()

        assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."
        assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

        step_col = ['Step_Name']
        step_df = df[step_col].values.tolist()
        step_df_length = len(step_df)
        first = 0
        for i in range(step_df_length):
            if step_df[i][0] == '375H':
                first = i
                break
        for i in range(step_df_length):
            if step_df[i][0] == 'SLOW_COOL':
                _state_df = state_df[first - history_len:i]
                _action_df = action_df[first - history_len:i]
                break

        state = torch.from_numpy(_state_df.to_numpy()).float()  # [time stamp x #. states]
        action = torch.from_numpy(_action_df.to_numpy()).float()  # [time_stamp x  #. controls]
        _preprocess_state_and_action(state, action, history_len, future_len,
                                     hist_x, future_x, hist_u, future_u)

    hist_x = torch.stack(hist_x).transpose(2, 1)
    hist_x = hist_x.unsqueeze(dim=-1)  # [batch x state nodes x history_len x state dim (=1)]
    future_x = torch.stack(future_x).transpose(2, 1)
    future_x = future_x.unsqueeze(dim=-1)  # [batch x state nodes x future_len x state dim (=1)]

    hist_u = torch.stack(hist_u).transpose(2, 1)
    hist_u = hist_u.unsqueeze(dim=-1)  # [batch x state nodes x history_len x control dim (=1)]
    future_u = torch.stack(future_u).transpose(2, 1)
    future_u = future_u.unsqueeze(dim=-1)  # [batch x state nodes x future_len x control dim (=1)]

    info = dict()
    if scaling:
        # scaling states
        if state_scaler is None:
            state_min = min(hist_x.min(), future_x.min())
            state_max = max(hist_x.max(), future_x.max())
        else:
            state_min, state_max = state_scaler[0], state_scaler[1]

        hist_x = (hist_x - state_min) / (state_max - state_min)
        future_x = (future_x - state_min) / (state_max - state_min)

        # scaling controls
        if action_scaler is None:
            action_min = min(hist_u.min(), future_u.min())
            action_max = max(hist_u.max(), future_u.max())
        else:
            action_min, action_max = action_scaler[0], action_scaler[1]

        hist_u = (hist_u - action_min) / (action_max - action_min)
        future_u = (future_u - action_min) / (action_max - action_min)

        info['state_scaler'] = (state_min, state_max)
        info['action_scaler'] = (action_min, action_max)

    return hist_x, future_x, hist_u, future_u, info


if __name__ == '__main__':
    path = '~/dev/WONIK/docs/new_data/linear/data_1.csv'
    hist_len, future_len = 10, 600
    hx, fx, hu, fu, info = load_data(path, hist_len, future_len,
                                     step_names=['375H', '375S', '375A'])
