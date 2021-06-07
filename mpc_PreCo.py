import os
import random
import torch
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from src.model.get_model import get_preco_model
from src.utils.load_data import load_preco_data
from src.utils.data_preprocess import get_preco_data
from src.control.torch_mpc import PreCoTorchMPC

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
                 timeout=5.0,
                 is_logging=True):
        self.solver = PreCoTorchMPC(model=model,
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
    state_dim = 140
    action_dim = 40
    hidden_dim = 256
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
    model_filename = 'model/PreCo.pt'
    m = get_preco_model(state_dim, action_dim, hidden_dim)
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()

    stepTime = 0

    heatup_step = '375H'  # 375
    stable_step = '375S'  # 375
    anneal_step = '375A'  # 375

    # heatup_step = '400H'    #400
    # stable_step = '400S'    #400
    # anneal_step = '400A'    #400

    state_scaler = (20.0, 420.0)
    action_scaler = (20.0, 420.0)

    # Create Target Temperature
    heatup150 = 545  # 375H, 45m
    stable150 = 181  # 375H, 15m
    anneal150 = 182  # 375H, 15m

    # heatup150 = 183    #375H, 15m
    # heatup150 = 243    #375H, 20m
    # heatup150 = 606     #400H
    # stable150 = 63     #375H, 5m
    # anneal150 = 363     #375H, 30m

    target_temps = [159.0, 375.0, 375.0, 375.0]  # 375
    # target_temps = [159.0, 400.0, 400.0, 400.0]     #400
    times = [heatup150, stable150, anneal150 + H]

    target = generate_reference2(target_temps=target_temps, times=times)

    target = torch.reshape(torch.tensor(target).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))
    target = (target - state_scaler[0]) / (state_scaler[1] - state_scaler[0])  # Change scale into 0-1, [908+H x 140]

    step = 0
    log_history = []
    trajectory_tc = []
    trajectory_ws = []

    time.sleep(1)

    glass_tc_value = np.array(plc.glass_tc, dtype='float32')  # 140*1
    glass_tc_value = np.reshape(glass_tc_value, (1, -1))  # 1*140

    heater_sp_value = np.array(plc.heater_sp, dtype='float32')
    heater_sp_value = np.reshape(heater_sp_value[:action_dim], (1, -1))

    history_tc = tc_buff = glass_tc_value
    history_ws = ws_buff = heater_sp_value

    # history_tc: numpy [state_order x state_dim 10*140]
    # history_ws: numpy [action_order x action_dim 49*40]
    for i in range(state_order - 1):
        history_tc = np.r_[history_tc, tc_buff]

    for i in range(action_order - 2):
        history_ws = np.r_[history_ws, ws_buff]

    step_log.write('Start FTC')

    while True:
        start_time = time.time()
        # time.sleep(0.1)

        global runner_state_
        runner_state_ = running
        stepName = plc.step_name

        # Enter if not processing or stop
        if not plc.processing or stop:
            # Enter if running is True, save log file
            if running:
                step = 0
                step_time = 0
                before_time = 0
                running = False
                step_log.write('runner End')

                # Save whole value when runner finished
                cur_time = setFilename()
                save_dir = 'log/{}'.format(cur_time)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(save_dir + '/control_log.txt', 'wb') as f:
                    pickle.dump(log_history, f)

                trajectory_tc = np.concatenate(trajectory_tc, axis=0)
                trajectory_ws = np.concatenate(trajectory_ws, axis=0)
                np.save(save_dir + '/trajectory_tc.npy', trajectory_tc)
                np.save(save_dir + '/trajectory_ws.npy', trajectory_ws)

                log_history = []
                trajectory_tc = []
                trajectory_ws = []

                # print(log_history)
                # print("-------------")
                # print(len(trajectory_ws))
                # print("-------------")
                # print(len(trajectory_tc))
                step_log.write('logging End')
            stop = False
            continue
        # Enter if the running starts
        elif not running and stepName == heatup_step:
            log_history = []
            trajectory_tc = []
            trajectory_ws = []

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

        # Enter if running is True
        elif running:
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
                step_time = stepTime + int(heatup150 * 5) + int(stable150 * 5)
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

            # observed_tc = np.array(plc.glass_tc, dtype='float32').reshape(1, state_dim)
            # print("obs tc shape {}".format(observed_tc.shape))
            workset = workset * (action_scaler[1] - action_scaler[0]) + action_scaler[0]
            workset = workset.cpu().detach().numpy()  # [1 x 40] numpy.array

            # Expected input shape of plc.set_heater() = [action_dim (=40) x 1]
            wsvalue = workset.reshape(action_dim, 1)

            bCheck = np.isnan(wsvalue).any()

            if bCheck:
                continue
            else:
                plc.set_heater(wsvalue)
                step_log.write(
                    'StepTime : {} PLC StepTime : {}, Step : {}, StepName : {}'.format(step_time, stepTime, step,
                                                                                       stepName))

        # Enter if heatup is not coming yet
        else:
            time.sleep(0.1)

        # Any way, observe TC and Workset
        elapsed = time.time() - start_time
        wait_for = max(4.5 - elapsed, 0)
        time.sleep(wait_for)
        observed_tc = np.array(plc.glass_tc, dtype='float32').reshape(1, state_dim)
        workset = np.array(plc.heater_sp, dtype='float32')[:action_dim].reshape(1, action_dim)

        history_tc = np.concatenate([history_tc[1:, :],
                                     observed_tc],
                                    axis=0)
        history_ws = np.concatenate([history_ws[1:, :],
                                     workset],
                                    axis=0)

        # step_log.write('TC {} WS {}'.format(observed_tc, workset))

        trajectory_tc.append(observed_tc)
        trajectory_ws.append(workset)

        elapsed = time.time() - start_time
        wait_for = max(5.0 - elapsed, 0)
        time.sleep(wait_for)
        # Add logging
        if running:
            if len(log_history) > 0:
                log_history[-1]['step_number'] = step
                log_history[-1]['total_step_time'] = time.time() - start_time


if __name__ == '__main__':
    # Model Type
    is_control_TCs = [False]
    state_orders = [50]
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
    max_iters = [100]  # Maximum number of optimize  r iterations
    u_range = 0.03  # will be ignored if smooth_u_type == penalty, cannot be list

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
