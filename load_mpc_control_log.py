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
    '''if is_control_TC:
        dir = 'control_log_simulation/multistep_linear_residual_with_control_TC/residual_{}_{}_{}_{}/{}_{}_{}_{}_{}_{}_{}_{}'.format(
            state_order, action_order, training_H, training_alpha, control_scheme, H, smooth_u_type, alpha, u_range,
            optimizer_mode,
            initial_solution, max_iter)
    else:
        dir = 'control_log_simulation/multistep_linear_residual_without_control_TC/residual_{}_{}_{}_{}/{}_{}_{}_{}_{}_{}_{}_{}'.format(
            state_order, action_order, training_H, training_alpha, control_scheme, H, smooth_u_type, alpha, u_range,
            optimizer_mode,
            initial_solution, max_iter)'''
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
    #print(len(trajectory_ws))
    #print(len(control_log))
'''
if __name__ == '__main__':
    is_control_TCs = [False]
    state_orders = [10]
    action_orders = [50]
    training_Hs = [50]
    training_alphas = [1]

    # Control Type, Assume that this code is only for MPC style
    control_scheme = 'MPC'  # MPC or openloop
    smooth_u_type = 'penalty'  # penalty or constraint or boundary
    u_range = 0.01  # will be ignored if smooth_u_type == penalty
    alphas = [0, 100, 1000]  # will be ignored if smooth_u_type == constraint or boundary

    # Hyper-parameters for MPC optimizer
    Hs = [50, 100]  # Receding horizon
    optimizer_modes = ['Adam']  # Adam or LBFGS
    initial_solutions = ['previous']  # target or previous
    max_iters = [50]  # Maximum number of optimizer iterations
    for is_control_TC in is_control_TCs:
        for state_order in state_orders:
            for action_order in action_orders:
                for training_H in training_Hs:
                    for training_alpha in training_alphas:
                        for H in Hs:
                            for alpha in alphas:
                                for optimizer_mode in optimizer_modes:
                                    for initial_solution in initial_solutions:
                                        for max_iter in max_iters:
                                            main(is_control_TC=is_control_TC,
                                                 state_order=state_order,
                                                 action_order=action_order,
                                                 training_H=training_H,
                                                 training_alpha=training_alpha,
                                                 control_scheme=control_scheme,
                                                 smooth_u_type=smooth_u_type,
                                                 H=H,
                                                 alpha=alpha,
                                                 u_range=u_range,
                                                 optimizer_mode=optimizer_mode,
                                                 initial_solution=initial_solution,
                                                 max_iter=max_iter)
'''

if __name__ == '__main__':
    # Model Type
    is_control_TCs = [False]
    state_orders = [10]
    action_orders = [50]
    training_Hs = [50]
    training_alphas = [1]

    # Control Type, Assume that this code is only for MPC style
    control_scheme = 'MPC'  # MPC or openloop
    smooth_u_type = 'penalty'  # penalty or constraint or boundary, cannot be list

    # Hyper-parameters for MPC optimizer
    Hs = [100]  # Receding horizon
    alphas = [100]  # will be ignored if smooth_u_type == constraint or boundary
    optimizer_modes = ['Adam']  # Adam or LBFGS
    initial_solutions = ['previous']  # target or previous
    max_iters = [50]  # Maximum number of optimizer iterations
    u_range = 0.01  # will be ignored if smooth_u_type == penalty, cannot be list

    for is_control_TC in is_control_TCs:
        for state_order in state_orders:
            for action_order in action_orders:
                for training_H in training_Hs:
                    for training_alpha in training_alphas:
                        for H in Hs:
                            for alpha in alphas:
                                for optimizer_mode in optimizer_modes:
                                    for initial_solution in initial_solutions:
                                        for max_iter in max_iters:
                                            main(is_control_TC=is_control_TC,
                                                 state_order=state_order,
                                                 action_order=action_order,
                                                 training_H=training_H,
                                                 training_alpha=training_alpha,
                                                 control_scheme=control_scheme,
                                                 smooth_u_type=smooth_u_type,
                                                 H=H,
                                                 alpha=alpha,
                                                 u_range=u_range,
                                                 optimizer_mode=optimizer_mode,
                                                 initial_solution=initial_solution,
                                                 max_iter=max_iter)