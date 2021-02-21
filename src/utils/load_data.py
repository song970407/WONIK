import re
import pandas as pd
import torch

from typing import Union, List


def load_data(paths: Union[str, List[str]],
              scaling: bool = True,
              scaler: tuple = None,
              action_ws: bool = True,
              preprocess: bool = False,
              history_x: int = 1,
              history_u: int = 1,
              device: str = 'cpu'):
    if isinstance(paths, str):
        paths = [paths]
    states = []
    actions = []
    for path in paths:
        df = pd.read_csv(path)
        l = df.columns.to_list()
        action_rex = 'Z.+_WS$' if action_ws else 'Z.+_TC$'
        action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
        glass_cols = [string for string in l if re.search(re.compile('TC_Data_.+'), string)]
        control_cols = [string for string in l if re.search(re.compile('Z.+_TC$'), string)]
        state_cols = glass_cols + control_cols
        state_df = df[state_cols]
        state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
        action_df = df[action_cols]
        action_nan_cols = action_df.columns[action_df.isna().any()].tolist()

        assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."
        assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

        state = torch.tensor(state_df.to_numpy()).float()
        action = torch.tensor(action_df.to_numpy()).float()

        info = {
            'states_cols': state_cols,
            'action_cols': action_cols,
            'num_glass_tc': len(glass_cols),
            'num_control_tc': len(control_cols),
            'num_control': len(action_cols)
        }

        step_col = ['Step_Name']
        step_df = df[step_col].values.tolist()
        step_df_length = len(step_df)

        if preprocess:
            for i in range(step_df_length):
                if step_df[i][0] == '375H':
                    first = i
                    break
            for i in range(step_df_length):
                if step_df[i][0] == 'SLOW_COOL':
                    state = state[first - history_x + 1:i + 1, :]
                    action = action[first - history_u + 1:i, :]
                    break
        states.append(state)
        actions.append(action)
    sa_min = 10000000000
    sa_max = -10000000000
    if scaling:
        if scaler is None:
            for state, action in zip(states, actions):
                sa_min = min(min(state.min(), action.min()), sa_min)
                sa_max = max(max(state.max(), action.max()), sa_max)
        else:
            sa_min, sa_max = scaler[0], scaler[1]
        for idx in range(len(states)):
            states[idx] = (states[idx] - sa_min) / (sa_max - sa_min)
            actions[idx] = (actions[idx] - sa_min) / (sa_max - sa_min)

        info['scale_min'] = sa_min
        info['scale_max'] = sa_max
    states = states.to(device)
    actions = actions.to(device)
    return states, actions, info


def load_new_data(path: str,
                  scaling: bool = True,
                  scaler: tuple = None,
                  action_ws: bool = False,
                  preprocess: bool = False):
    df = pd.read_csv(path)
    l = df.columns.to_list()
    action_rex = 'Z.+_WS$' if action_ws else 'Z.+_TC$'
    action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
    state_cols = [string for string in l if re.search(re.compile('TC_Data_.+'), string)]
    state_df = df[state_cols]
    state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
    assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."

    action_df = df[action_cols]
    action_nan_cols = action_df.columns[action_df.isna().any()].tolist()
    assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

    state = torch.tensor(state_df.to_numpy()).float()
    action = torch.tensor(action_df.to_numpy()).float()

    info = {
        'states_cols': state_cols,
        'action_cols': action_cols
    }

    step_col = ['Step_Name']
    step_df = df[step_col].values.tolist()
    step_df_length = len(step_df)

    if preprocess:
        for i in range(step_df_length):
            if step_df[i][0] == '375H':
                first = i
                break
        for i in range(step_df_length):
            if step_df[i][0] == 'SLOW_COOL':
                state = state[first:i, :]
                action = action[first:i, :]
                break

    if scaling:
        if scaler is None:
            sa_min = min(state.min(), action.min())
            sa_max = max(state.max(), action.max())
        else:
            sa_min, sa_max = scaler[0], scaler[1]

        state = (state - sa_min) / (sa_max - sa_min)
        action = (action - sa_min) / (sa_max - sa_min)

        info['scale_min'] = sa_min
        info['scale_max'] = sa_max

    return state, action, info


def load_data_icgrnn(path: str,
                     scaling: bool = True,
                     scaler: tuple = None,
                     action_ws: bool = False,
                     preprocess: bool = False,
                     history_window: int = 0):
    df = pd.read_csv(path)
    l = df.columns.to_list()
    action_rex = 'Z.+_WS$' if action_ws else 'Z.+_TC$'
    action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
    glass_cols = [string for string in l if re.search(re.compile('TC_Data_.+'), string)]
    control_cols = [string for string in l if re.search(re.compile('Z.+_TC$'), string)]
    state_cols = glass_cols + control_cols
    state_df = df[state_cols]
    state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
    assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."

    action_df = df[action_cols]
    action_nan_cols = action_df.columns[action_df.isna().any()].tolist()
    assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

    state = torch.tensor(state_df.to_numpy()).float()
    action = torch.tensor(action_df.to_numpy()).float()

    info = {
        'states_cols': state_cols,
        'action_cols': action_cols
    }

    step_col = ['Step_Name']
    step_df = df[step_col].values.tolist()
    step_df_length = len(step_df)

    if preprocess:
        for i in range(step_df_length):
            if step_df[i][0] == '375H':
                first = i
                break
        for i in range(step_df_length):
            if step_df[i][0] == 'SLOW_COOL':
                state = state[first - history_window:i, :]
                action = action[first:i, :]
                break

    if scaling:
        if scaler is None:
            sa_min = min(state.min(), action.min())
            sa_max = max(state.max(), action.max())
        else:
            sa_min, sa_max = scaler[0], scaler[1]

        state = (state - sa_min) / (sa_max - sa_min)
        action = (action - sa_min) / (sa_max - sa_min)

        info['scale_min'] = sa_min
        info['scale_max'] = sa_max

    return state, action, info


def load_data_icgatrnn(path: str,
                       scaling: bool = True,
                       scaler: tuple = None,
                       action_ws: bool = False,
                       preprocess: bool = False,
                       history_window: int = 0):
    df = pd.read_csv(path)
    l = df.columns.to_list()
    action_rex = 'Z.+_WS$' if action_ws else 'Z.+_TC$'
    action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
    glass_cols = [string for string in l if re.search(re.compile('TC_Data_.+'), string)]
    control_cols = [string for string in l if re.search(re.compile('Z.+_TC$'), string)]
    state_cols = glass_cols + control_cols
    state_df = df[state_cols]
    state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
    assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."

    action_df = df[action_cols]
    action_nan_cols = action_df.columns[action_df.isna().any()].tolist()
    assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

    state = torch.tensor(state_df.to_numpy()).float()
    action = torch.tensor(action_df.to_numpy()).float()

    info = {
        'states_cols': state_cols,
        'action_cols': action_cols
    }

    step_col = ['Step_Name']
    step_df = df[step_col].values.tolist()
    step_df_length = len(step_df)

    if preprocess:
        for i in range(step_df_length):
            if step_df[i][0] == '375H':
                first = i
                break
        for i in range(step_df_length):
            if step_df[i][0] == 'SLOW_COOL':
                state = state[first - history_window + 1:i + 1, :]
                action = action[first:i, :]
                break

    if scaling:
        if scaler is None:
            sa_min = min(state.min(), action.min())
            sa_max = max(state.max(), action.max())
        else:
            sa_min, sa_max = scaler[0], scaler[1]

        state = (state - sa_min) / (sa_max - sa_min)
        action = (action - sa_min) / (sa_max - sa_min)

        info['scale_min'] = sa_min
        info['scale_max'] = sa_max

    return state, action, info


def load_datas_icgatrnn(paths: Union[str, List[str]],
                        scaling: bool = True,
                        scaler: tuple = None,
                        action_ws: bool = False,
                        preprocess: bool = False,
                        history_window: int = 1):
    if isinstance(paths, str):
        paths = [paths]
    states = []
    actions = []
    for path in paths:
        df = pd.read_csv(path)
        l = df.columns.to_list()
        action_rex = 'Z.+_WS$' if action_ws else 'Z.+_TC$'
        action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
        glass_cols = [string for string in l if re.search(re.compile('TC_Data_.+'), string)]
        control_cols = [string for string in l if re.search(re.compile('Z.+_TC$'), string)]
        state_cols = glass_cols + control_cols
        state_df = df[state_cols]
        state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
        action_df = df[action_cols]
        action_nan_cols = action_df.columns[action_df.isna().any()].tolist()

        assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."
        assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

        state = torch.tensor(state_df.to_numpy()).float()
        action = torch.tensor(action_df.to_numpy()).float()

        info = {
            'states_cols': state_cols,
            'action_cols': action_cols
        }

        step_col = ['Step_Name']
        step_df = df[step_col].values.tolist()
        step_df_length = len(step_df)

        if preprocess:
            for i in range(step_df_length):
                if step_df[i][0] == '375H':
                    first = i
                    break
            for i in range(step_df_length):
                if step_df[i][0] == 'SLOW_COOL':
                    state = state[first - history_window + 1:i + 1, :]
                    action = action[first:i, :]
                    break
        states.append(state)
        actions.append(action)
    sa_min = 10000000000
    sa_max = -10000000000
    if scaling:
        if scaler is None:
            for state, action in zip(states, actions):
                sa_min = min(min(state.min(), action.min()), sa_min)
                sa_max = max(max(state.max(), action.max()), sa_max)
        else:
            sa_min, sa_max = scaler[0], scaler[1]
        for idx in range(len(states)):
            states[idx] = (states[idx] - sa_min) / (sa_max - sa_min)
            actions[idx] = (actions[idx] - sa_min) / (sa_max - sa_min)

        info['scale_min'] = sa_min
        info['scale_max'] = sa_max

    return states, actions, info


def load_data_pwr(path: str,
                  scaling: bool = True,
                  scaler: tuple = None,
                  preprocess: bool = False):
    df = pd.read_csv(path)
    l = df.columns.to_list()
    action_rex = 'Z.+_PWR$'
    action_cols = [string for string in l if re.search(re.compile(action_rex), string)]
    state_cols = [string for string in l if re.search(re.compile('TC_Data_.+'), string)]

    state_df = df[state_cols]
    state_nan_cols = state_df.columns[state_df.isna().any()].tolist()
    assert len(state_nan_cols) == 0, "Some of state columns contain NaNs."

    action_df = df[action_cols]
    action_nan_cols = action_df.columns[action_df.isna().any()].tolist()
    assert len(action_nan_cols) == 0, "Some of action columns contain NaNs."

    state = torch.tensor(state_df.to_numpy()).float()
    action = torch.tensor(action_df.to_numpy()).float()

    info = {
        'states_cols': state_cols,
        'action_cols': action_cols
    }

    step_col = ['Step_Name']
    step_df = df[step_col].values.tolist()
    step_df_length = len(step_df)

    if preprocess:
        for i in range(step_df_length):
            if step_df[i][0] == '150H':
                first = i
                break
        for i in range(step_df_length):
            if step_df[i][0] == 'PRECOOL1':
                state = state[first:i, :]
                action = action[first:i, :]
                break

    if scaling:
        if scaler is None:
            sa_min = state.min()
            sa_max = state.max()
        else:
            sa_min, sa_max = scaler[0], scaler[1]

        state = (state - sa_min) / (sa_max - sa_min)

        info['scale_min'] = sa_min
        info['scale_max'] = sa_max

    return state, action, info
