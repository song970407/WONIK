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

from ips.ControlPlc import ControlPlc
from ips.TraceLog import TraceLog, setFilename

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


def main2(is_control_TC, state_order, action_order, training_H, training_alpha, H, smooth_u_type, alpha, u_range,
          optimizer_mode, initial_solution, max_iter):
    # IPS Code
    plc = ControlPlc()
    plc.setDaemon(True)
    plc.start()

    step_log = TraceLog()

    running = False
    stop = False

    # Setting
    noise = 0
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
    time_limit = 4.5  # seconds

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

    stepTime = 0
    while True:
        heatup_step = '375H'
        stable_step = '375S'
        anneal_step = '375A'

        state_scaler = (20.0, 420.0)
        action_scaler = (20.0, 420.0)

        time.sleep(0.1)

        global runner_state_
        runner_state_ = running

        if not plc.processing or stop:
            if running:
                step = 0
                step_time = 0
                before_time = 0
                running = False
                step_log.write('runner End')

                # Save whole value when runner finished
                cur_time = setFilename()
                with open('log/control_log_{}.txt'.format(cur_time), 'wb') as f:
                    pickle.dump(log_history, f)

                trajectory_tc = np.concatenate(trajectory_tc, axis=0)
                trajectory_ws = np.concatenate(trajectory_ws, axis=0)
                np.save('log/trajectory_tc_{}.npy'.format(cur_time), trajectory_tc)
                np.save('log/trajectory_ws_{}.npy'.format(cur_time), trajectory_ws)

                # print(log_history)
                # print("-------------")
                # print(len(trajectory_ws))
                # print("-------------")
                # print(len(trajectory_tc))
                step_log.write('logging End')

            stop = False
            continue

        stepName = plc.step_name

        if not running and stepName == heatup_step:
            log_history = []
            trajectory_tc = []
            trajectory_ws = []

            # Create Target Temperature
            heatup150 = 545
            stable150 = 181
            anneal150 = 182

            target_temps = [159.0, 363.0, 375.0, 375.0]
            times = [heatup150, stable150, anneal150 + H]

            target = generate_reference2(target_temps=target_temps, times=times)

            target = torch.reshape(torch.tensor(target).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))
            target = (target - state_scaler[0]) / (
                    state_scaler[1] - state_scaler[0])  # Change scale into 0-1, [908+H x 140]

            # Get glass TC, WS & reshape
            glass_tc_value = np.array(plc.glass_tc, dtype='float32')  # 140*1
            glass_tc_value = np.reshape(glass_tc_value, (1, -1))  # 1*140

            heater_sp_value = np.array(plc.heater_sp, dtype='float32')
            heater_sp_value = np.reshape(heater_sp_value[:40], (1, -1))

            history_tc = tc_buff = glass_tc_value
            history_ws = ws_buff = heater_sp_value

            # history_tc: numpy [state_order x state_dim 10*140]
            # history_ws: numpy [action_order x action_dim 49*40]
            for i in range(state_order - 1):
                history_tc = np.r_[history_tc, tc_buff]

            for i in range(action_order - 2):
                history_ws = np.r_[history_ws, ws_buff]

            if is_del_u:
                if from_target:
                    initial_ws = torch.zeros((H, action_dim), device=device)
                else:
                    initial_ws = target[1:H + 1, :action_dim] - target[:H, :action_dim]
            else:
                initial_ws = target[:H, :action_dim]  # [H x 40], 0-1 scale

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

            step_log.write('runner Start '.format(plc.rcp_name))
            running = True

        if running:
            start_time = time.time()

            if stepName != heatup_step and stepName != stable_step and stepName != anneal_step:
                plc.reset_heater()
                stop = True
                step_log.write('Anneal End')
                continue

            if not plc.ready:
                plc.reset_heater()
                stop = True
                step_log.write('ready state NG')
                continue

            stepTime = plc.step_time
            if stepName == stable_step:
                step_time = stepTime + int(heatup150 * 5)
            elif stepName == anneal_step:
                step_time = stepTime + int(heatup150 * 5) + \
                            int(stable150 * 5)
            else:
                step_time = stepTime  # steptime calculation

            quot = int(step_time // 5)
            step = quot

            action, log = runner.solve(history_tc, history_ws, target[step:step + H, :], initial_ws)

            # Processing Workset
            if is_del_u:
                if from_target:
                    workset = target[step:step + 1, :action_dim] + action[0:1, :]
                    if use_previous:
                        initial_ws = torch.cat([action[1:], action[-1:]], dim=0)
                    else:
                        initial_ws = torch.zeros_like(action).to(device)
                else:
                    prev_workset = (history_ws[-1:, :] - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
                    workset = torch.from_numpy(prev_workset).float().to(device) + action[0:1, :]
                    initial_ws = torch.cat([action[1:], action[-1:]], dim=0)
            else:
                workset = action[0:1, :]
                if use_previous:
                    initial_ws = torch.cat([action[1:], action[-1:]], dim=0)
                else:
                    initial_ws = target[step + 1: step + H + 1, :action_dim]

            log_history.append(log)

            observed_tc = np.array(plc.glass_tc, dtype='float32').reshape(1, state_dim)
            print("obs tc shape {}".format(observed_tc.shape))
            workset = workset * (action_scaler[1] - action_scaler[0]) + action_scaler[0]
            workset = workset.cpu().detach().numpy()  # [1 x 40] numpy.array

            # Expected input shape of plc.set_heater() = [action_dim (=40) x 1]
            wsvalue = workset.reshape(40, 1)

            bCheck = np.isnan(wsvalue).any()

            if bCheck:
                continue
            else:
                plc.set_heater(wsvalue)
                step_log.write(
                    'StepTime : {} PLC StepTime : {}, Step : {}, StepName : {}'.format(step_time, stepTime, step,
                                                                                       stepName))

            elapsed = time.time() - start_time
            wait_for = max(5.0 - elapsed, 0)
            time.sleep(wait_for)

            history_tc = np.concatenate([history_tc[1:, :],
                                         observed_tc],
                                        axis=0)
            history_ws = np.concatenate([history_ws[1:, :],
                                         workset],
                                        axis=0)

            trajectory_tc.append(observed_tc)
            trajectory_ws.append(workset)


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
    Hs = [100]  # Receding horizon
    alphas = [1000]  # will be ignored if smooth_u_type == constraint or boundary
    optimizer_modes = ['Adam']  # Adam or LBFGS
    initial_solutions = ['previous']  # target or previous
    max_iters = [50]  # Maximum number of optimizer iterations
    u_range = 0.01  # will be ignored if smooth_u_type == penalty, cannot be list

    for is_control_TC in is_control_TCs:
        for state_order in state_orders:
            for action_order in action_orders:
                for training_alpha in training_alphas:
                    for training_H in training_Hs:
                        for H in Hs:
                            for alpha in alphas:
                                for optimizer_mode in optimizer_modes:
                                    for initial_solution in initial_solutions:
                                        for max_iter in max_iters:
                                            main2(is_control_TC=is_control_TC,
                                                  state_order=state_order,
                                                  action_order=action_order,
                                                  training_H=training_H,
                                                  training_alpha=training_alpha,
                                                  H=H,
                                                  smooth_u_type=smooth_u_type,
                                                  alpha=alpha,
                                                  u_range=u_range,
                                                  optimizer_mode=optimizer_mode,
                                                  initial_solution=initial_solution,
                                                  max_iter=max_iter)
