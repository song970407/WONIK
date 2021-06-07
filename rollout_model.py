import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model.get_model import get_multi_linear_residual_model
from src.utils.load_data import load_data
from src.utils.data_preprocess import get_data
from src.control.torch_mpc import LinearTorchMPC

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def main(state_order, action_order, H, alpha, is_control_TC, rollout_window, test_data_path):
    state_dim = 140
    action_dim = 40
    if is_control_TC:
        model_filename = 'model/Multistep_linear_with_control_TC/residual_{}_{}/model_{}_{}.pt'.format(H, alpha,
                                                                                                       state_order,
                                                                                                       action_order)
        m = get_multi_linear_residual_model(state_dim + action_dim, action_dim, state_order, action_order).to(device)
    else:
        model_filename = 'model/Multistep_linear_without_control_TC/residual_{}_{}/model_{}_{}.pt'.format(H, alpha,
                                                                                                          state_order,
                                                                                                          action_order)
        m = get_multi_linear_residual_model(state_dim, action_dim, state_order, action_order).to(device)

    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()

    scaler = (20.0, 420.0)

    test_states, test_actions, _ = load_data(paths=test_data_path,
                                             scaling=True,
                                             scaler=scaler,
                                             preprocess=True,
                                             history_x=state_order,
                                             history_u=action_order)

    glass_tc_path = 'docs/new_location/glass_TC.csv'
    control_tc_path = 'docs/new_location/control_TC.csv'

    history_xs, history_us, us, ys, _ = get_data(states=test_states,
                                                 actions=test_actions,
                                                 rollout_window=rollout_window,
                                                 history_x_window=state_order,
                                                 history_u_window=action_order,
                                                 glass_tc_pos_path=glass_tc_path,
                                                 control_tc_pos_path=control_tc_path,
                                                 num_glass_tc=140,
                                                 num_control_tc=40,
                                                 device=device)
    if is_control_TC:
        history_xs = torch.cat([history_xs[0], history_xs[1]], dim=1).transpose(1, 2)
        ys = torch.cat([ys[0], ys[1]], dim=1).transpose(1, 2)
    else:
        history_xs = history_xs[0].transpose(1, 2)
        ys = ys[0].transpose(1, 2)
    history_us = history_us.transpose(1, 2)
    us = us.transpose(1, 2)
    num_of_test = us.shape[0]
    predicted_ys = []

    with torch.no_grad():
        for idx in range(num_of_test):
            if idx > 0:
                break
            predicted_y = m.multi_step_prediction(history_xs[idx], history_us[idx], us[idx])
            predicted_ys.append(predicted_y)
        predicted_ys = torch.stack(predicted_ys, dim=0)
        ys = ys * (scaler[1] - scaler[0]) + scaler[0]
        predicted_ys = predicted_ys * (scaler[1] - scaler[0]) + scaler[0]
        ys = ys.cpu().detach().numpy()
        predicted_ys = predicted_ys.cpu().detach().numpy()

        us = us * (scaler[1] - scaler[0]) + scaler[0]
        us = us.cpu().detach().numpy()

    return ys, predicted_ys, us


if __name__ == '__main__':
    is_control_TCs = [False]
    state_orders = [1, 5, 10, 30, 50]
    action_orders = [50]
    H = 50
    alphas = [0]
    pre_model_filename = 'model/Multistep_linear/residual_model_ridge_' + str(H) + '/model_'
    # pre_model_filename = 'model/Multistep_linear/residual_model_ridge_100/model_'
    test_data_paths = ['docs/new_data/overshoot/data_1.csv', 'docs/new_data/expert/data_3.csv',
                       'docs/new_data/icgrnn/data_4.csv', 'docs/new_data/linear/data_3.csv']
    rollout_windows = [2400, 900, 900, 900]
    yss = []
    for is_control_TC in is_control_TCs:
        for state_order in state_orders:
            for action_order in action_orders:
                for alpha in alphas:
                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10 * 2, 10 * 2))
                    axes_flatten = axes.flatten()
                    for k in range(len(test_data_paths)):
                        ys, predicted_ys, us = main(state_order, action_order, H, alpha, is_control_TC,
                                                    rollout_windows[k], test_data_paths[k])
                        if len(yss) < len(test_data_paths):
                            yss.append(ys[0])
                        axes_flatten[k].plot(predicted_ys[0])
                        axes_flatten[k].set_title('Rollout Test dataset {}'.format(k + 1))
                    fig.show()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10 * 2, 10 * 2))
    axes_flatten = axes.flatten()
    for k in range(len(test_data_paths)):
        axes_flatten[k].plot(yss[k])
        axes_flatten[k].set_title('True Test dataset {}'.format(k + 1))
    fig.show()
