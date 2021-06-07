import os
import random
import torch
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from src.model.get_model import get_multi_linear_residual_model
from src.utils.load_data import load_data
from src.utils.data_preprocess import get_data
from src.control.torch_mpc import LinearTorchMPC

# Check whether the GPU (CUDA) is available or not
from src.utils.reference_generator import generate_reference, generate_reference2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)

# Fix seed number for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class Runner:
    def __init__(self,
                 model,
                 state_scaler,
                 action_scaler,
                 alpha,
                 is_del_u,
                 from_target,
                 u_min,
                 u_max,
                 optimizer_mode,
                 max_iter,
                 timeout=5,
                 is_logging=True):
        self.solver = LinearTorchMPC(model=model,
                                     time_aggregator='sum',
                                     optimizer_mode=optimizer_mode,
                                     max_iter=max_iter,
                                     alpha=alpha,
                                     u_min=u_min,
                                     u_max=u_max,
                                     is_logging=is_logging,
                                     is_del_u=is_del_u,
                                     from_target=from_target,
                                     timeout=timeout,
                                     device=device)
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler

    def solve(self, history_tc, history_ws, target, initial_ws):
        """
        :param history_tc: state_order x num_state
        :param history_ws: (action_order-1) x num_action
        :param target: H x num_state
        :param initial_ws: H x num_action
        :return:
        """
        history_tc = torch.from_numpy(history_tc).float().to(device)
        history_ws = torch.from_numpy(history_ws).float().to(device)
        history_tc = (history_tc - self.state_scaler[0]) / (self.state_scaler[1] - self.state_scaler[0])
        history_ws = (history_ws - self.action_scaler[0]) / (self.action_scaler[1] - self.action_scaler[0])
        action, log = self.solver.solve_mpc(history_tc, history_ws, target, initial_ws)
        return action, log


def main(is_control_TC,
         state_order,
         action_order,
         training_H,
         training_alpha,
         smooth_u_type,
         alpha,
         u_range,
         optimizer_mode,
         initial_solution,
         max_iter,
         start_point=0,
         end_point=None):
    # Setting
    noise = 0.05
    state_dim = 140
    action_dim = 40
    is_del_u = False
    from_target = False
    use_previous = False
    u_min = 0.0
    u_max = 1.0
    if smooth_u_type == 'penalty':
        u_min = 0.3
        u_max = 1.0
        u_range = 0
    elif smooth_u_type == 'constraint':
        is_del_u = True
        u_min = -1 * u_range
        u_max = u_range
    elif smooth_u_type == 'boundary':
        is_del_u = True
        from_target = True
        u_min = -1 * u_range
        u_max = u_range
    else:
        raise NotImplementedError
    if initial_solution == 'target':
        use_previous = False
    elif initial_solution == 'previous':
        use_previous = True
    else:
        raise NotImplementedError
    is_logging = True
    time_limit = 50000  # seconds

    # Set Model Filename
    if is_control_TC:
        model_filename = 'model/multistep_linear_residual_with_control_TC/model_{}_{}_{}_{}.pt'.format(
            state_order, action_order, training_H, training_alpha)
    else:
        model_filename = 'model/multistep_linear_residual_without_control_TC/model_{}_{}_{}_{}.pt'.format(
            state_order, action_order, training_H, training_alpha)

    # Load Prediction Model
    if is_control_TC:
        m = get_multi_linear_residual_model(state_dim + action_dim, action_dim, state_order, action_order)
    else:
        m = get_multi_linear_residual_model(state_dim, action_dim, state_order, action_order)
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()
    state_scaler = (20.0, 420.0)
    action_scaler = (20.0, 420.0)

    # Create Target Temperature
    heatup150 = 545
    stable150 = 181
    anneal150 = 182

    initial_temp = 150.0
    heatup_times = [heatup150]
    anneal_times = [stable150 + anneal150]
    target_temp = [375.0]
    target = generate_reference(initial_temp=initial_temp,
                                heatup_times=heatup_times,
                                anneal_times=anneal_times,
                                target_temps=target_temp)

    target_temps = [159.0, 375.0, 375.0, 375.0]
    times = [heatup150, stable150, anneal150]
    target = generate_reference2(target_temps=target_temps, times=times)

    target = torch.reshape(torch.tensor(target).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))
    target = (target - state_scaler[0]) / (state_scaler[1] - state_scaler[0])  # Change scale into 0-1, [908+H x 140]
    T = target.shape[0]
    # Load any initial point
    sample_data_path = 'docs/new_data/expert/data_1.csv'
    states, actions, _ = load_data(paths=sample_data_path,
                                   scaling=False,
                                   preprocess=True,
                                   history_x=state_order,
                                   history_u=action_order,
                                   device=device)
    if is_control_TC:
        history_tc = states[0][start_point:start_point + state_order].cpu().detach().numpy()
    else:
        history_tc = states[0][start_point:start_point + state_order, :state_dim].cpu().detach().numpy()
    history_ws = actions[0][start_point:start_point + action_order - 1].cpu().detach().numpy()
    if is_del_u:
        if from_target:
            initial_ws = torch.zeros((end_point - start_point, action_dim), device=device)
        else:
            initial_ws = target[start_point + 1:end_point + 1, :action_dim] - target[start_point:end_point, :action_dim]
    else:
        initial_ws = target[start_point:end_point, :action_dim]  # [H x 40], 0-1 scale
    runner = Runner(model=m,
                    state_scaler=state_scaler,
                    action_scaler=action_scaler,
                    alpha=alpha,
                    is_del_u=is_del_u,
                    from_target=from_target,
                    u_min=u_min,
                    u_max=u_max,
                    optimizer_mode=optimizer_mode,
                    max_iter=max_iter,
                    timeout=time_limit,
                    is_logging=is_logging)

    log_history = []
    print('Simulation starts from timestep ' + str(start_point))
    start = time.time()
    action, log = runner.solve(history_tc, history_ws, target[start_point:end_point], initial_ws)
    end = time.time()
    print('Time computation : {}'.format(end - start))

    if is_del_u:
        if from_target:
            workset = target[start_point:end_point, :action_dim] + action
        else:
            # NEED TO FIX
            prev_workset = (history_ws[-1:, :] - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
            workset = torch.from_numpy(prev_workset).float().to(device) + action[0:1, :]
    else:
        workset = action
    log_history.append(log)
    with torch.no_grad():
        x0 = torch.from_numpy(history_tc).float().to(device)
        u0 = torch.from_numpy(history_ws).float().to(device)
        x0 = (x0 - state_scaler[0]) / (state_scaler[1] - state_scaler[0])
        u0 = (u0 - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
        observed_tc = m.multi_step_prediction(x0, u0, workset).cpu().detach().numpy()  # [H x 140]
        observed_tc = observed_tc * (state_scaler[1] - state_scaler[0]) + state_scaler[0]  # un-scaling
    workset = workset * (state_scaler[1] - state_scaler[0]) + state_scaler[0]
    workset = workset.cpu().detach().numpy()  # [1 x 40] numpy.array


    if is_control_TC:
        dir = 'control_log_simulation/multistep_linear_residual_with_control_TC/residual_{}_{}_{}_{}/openloop_{}_{}_{}_{}_{}_{}'.format(
            state_order, action_order, training_H, training_alpha, smooth_u_type, alpha, u_range, optimizer_mode,
            initial_solution, max_iter)
    else:
        dir = 'control_log_simulation/multistep_linear_residual_without_control_TC/residual_{}_{}_{}_{}/openloop_{}_{}_{}_{}_{}_{}'.format(
            state_order, action_order, training_H, training_alpha, smooth_u_type, alpha, u_range, optimizer_mode,
            initial_solution, max_iter)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + '/control_log.txt', 'wb') as f:
        pickle.dump(log_history, f)
    np.save(dir + '/trajectory_tc.npy', observed_tc)
    np.save(dir + '/trajectory_ws.npy', workset)

if __name__ == '__main__':
    # Model Type
    is_control_TCs = [False]
    state_orders = [10]
    action_orders = [50]
    training_Hs = [50]
    training_alphas = [1]

    # Control Type, Assume that this code is only for MPC style
    smooth_u_type = 'boundary'  # penalty or constraint or boundary, cannot be list

    # Hyper-parameters for MPC optimizer
    alphas = [1000]  # will be ignored if smooth_u_type == constraint or boundary
    optimizer_modes = ['Adam']  # Adam or LBFGS
    initial_solutions = ['previous']  # target or previous
    max_iters = [1000]  # Maximum number of optimizer iterations
    u_range = 0.03  # will be ignored if smooth_u_type == penalty, cannot be list

    # Data clipping for fast result
    start_point = 400
    # end_point = 800
    end_point = 800

    for is_control_TC in is_control_TCs:
        for state_order in state_orders:
            for action_order in action_orders:
                for training_alpha in training_alphas:
                    for training_H in training_Hs:
                            for alpha in alphas:
                                for optimizer_mode in optimizer_modes:
                                    for initial_solution in initial_solutions:
                                        for max_iter in max_iters:
                                            main(is_control_TC=is_control_TC,
                                                 state_order=state_order,
                                                 action_order=action_order,
                                                 training_H=training_H,
                                                 training_alpha=training_alpha,
                                                 smooth_u_type=smooth_u_type,
                                                 alpha=alpha,
                                                 u_range=u_range,
                                                 optimizer_mode=optimizer_mode,
                                                 initial_solution=initial_solution,
                                                 max_iter=max_iter,
                                                 start_point=start_point,
                                                 end_point=end_point)