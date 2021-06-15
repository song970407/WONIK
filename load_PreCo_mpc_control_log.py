import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt


def main(is_control_TC,
         state_order,
         action_order,
         training_H,
         training_alpha,
         control_scheme,
         smooth_u_type,
         H,
         alpha,
         u_range,
         optimizer_mode,
         initial_solution,
         max_iter):
    if smooth_u_type == 'penalty':
        u_range = 0

    dir = 'log'
    control_log_path = dir + '/control_log_20210419_151157.txt'
    with open(control_log_path, 'rb') as f:
        control_log = pickle.load(f)
    trajectory_tc = np.load(dir + '/trajectory_tc_20210419_151157.npy')
    trajectory_ws = np.load(dir + '/trajectory_ws_20210419_151157.npy')
    print(trajectory_tc.shape)
    print(trajectory_ws.shape)
    log_idx_optimal = []
    log_loss_objective = []
    log_loss_delta_u = []
    log_loss_variance = []
    log_loss = []
    log_time = []
    print(control_log[0].keys())
    for i in range(len(control_log)):
        log_idx_optimal.append(control_log[i]['idx_optimal_us_value'])
        log_loss_objective.append(control_log[i]['trajectory_loss_objective'])
        log_loss_delta_u.append(control_log[i]['trajectory_loss_delta_u'])
        if 'trajectory_loss_variance' in control_log[0].keys():
            log_loss_variance.append(control_log[i]['trajectory_loss_variance'])
        log_loss.append(control_log[i]['trajectory_loss'])
        log_time.append(control_log[i]['trajectory_time'])

    log_loss_objective = np.array(log_loss_objective).transpose()
    log_loss_delta_u = np.array(log_loss_delta_u).transpose()
    if len(log_loss_variance) > 0:
        log_loss_variance = np.array(log_loss_variance).transpose()
    log_loss = np.array(log_loss).transpose()
    log_time = np.array(log_time).transpose()
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(3 * 10, 3 * 10))
    # Glass TC, Work Set, Loss objective, Loss delta u, total loss, time, best idx
    axes_flatten = axes.flatten()
    axes_flatten[0].plot(trajectory_tc)
    axes_flatten[0].set_title('Glass TC')
    axes_flatten[1].plot(trajectory_ws)
    axes_flatten[1].set_title('Work Set')
    axes_flatten[2].plot(log_idx_optimal)
    axes_flatten[2].set_title('Best Optimization Step')
    axes_flatten[3].plot(log_loss_objective)
    axes_flatten[3].set_title('Objective Loss')
    axes_flatten[4].plot(log_loss_delta_u)
    axes_flatten[4].set_title('Delta u Loss')
    axes_flatten[5].plot(log_loss)
    axes_flatten[5].set_title('Total Loss')
    axes_flatten[6].plot(log_time)
    axes_flatten[6].set_title('Time')
    if 'trajectory_loss_variance' in control_log[0].keys():
        axes_flatten[7].set_title('Variance Loss')
        axes_flatten[7].plot(log_loss_variance)

    if is_control_TC:
        fig_dir = 'fig/control_performance/multistep_linear_residual_with_control_TC/residual_{}_{}_{}_{}'.format(
            state_order, action_order, training_H, training_alpha)
    else:
        fig_dir = 'fig/control_performance/multistep_linear_residual_without_control_TC/residual_{}_{}_{}_{}'.format(
            state_order, action_order, training_H, training_alpha)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.suptitle('Model: {}_{}_{}_{}, Control: {}_{}_{}_{}_{}_{}_{}_{}'.format(state_order, action_order, training_H,
                                                                            training_alpha, control_scheme,
                                                                            smooth_u_type, H, alpha, u_range,
                                                                            optimizer_mode, initial_solution, max_iter))
    fig.savefig(
        fig_dir + '/{}_{}_{}_{}_{}_{}_{}_{}_control_result.png'.format(control_scheme, H, smooth_u_type, alpha, u_range,
                                                                       optimizer_mode, initial_solution, max_iter))
    fig.show()
#    print(np.min(trajectory_tc[-1, :]))
#    print(np.max(trajectory_tc[-1, :]))
    #print(trajectory_tc[0])
    print("--------------------------------")
    #print(trajectory_tc[1])
    print(control_log[0]['history_tc'])
    print(control_log[0]['history_ws'])


def load_log(model_type):
    dir = 'simulation_data/{}/'.format(model_type)
    control_log_path = dir + 'control_log.txt'
    with open(control_log_path, 'rb') as f:
        control_log = pickle.load(f)
    trajectory_tc = np.load(dir + 'trajectory_tc.npy')
    trajectory_ws = np.load(dir + 'trajectory_ws.npy')
    print(trajectory_tc.shape)
    print(trajectory_ws.shape)
    H = len(control_log)
    print(H)
    idx_list = []
    time_list = []
    for h in range(H):
        log = control_log[h]
        # print(log.keys())
        # idx_list.append(log['idx_optimal_us_value'])
        # time_list.append(log['trajectory_time'])
        plt.scatter(range(len(log['trajectory_time'])), log['trajectory_time'])
    plt.show()


if __name__ == '__main__':
    model_type = 'PreCo'
    load_log(model_type)
