import torch
import numpy as np
import time
import pickle

from src.model.get_model import get_multi_linear_residual_model
from src.utils.load_data import load_data
from src.utils.data_preprocess import get_data
from src.control.torch_mpc import LinearTorchMPC

# Check whether the GPU (CUDA) is available or not
from src.utils.reference_generator import generate_reference

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class Runner:
    def __init__(self, m, optimizer_mode, state_scaler, action_scaler, alpha, timeout=5):
        self.solver = LinearTorchMPC(model=m,
                                     time_aggregator='sum',
                                     optimizer_mode=optimizer_mode,
                                     alpha=alpha,
                                     timeout=timeout,
                                     device=device)
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler

    def solve(self, history_tc, history_ws, target, initial_ws=None):
        """
        :param history_tc: state_order x num_state
        :param history_ws: (action_order-1) x num_action
        :param target: H x num_state
        :param initial_ws: H x num_action
        :return:
        """
        # history_tc = torch.from_numpy(history_tc).float().to(device)
        # history_ws = torch.from_numpy(history_ws).float().to(device)
        history_tc = (history_tc - self.state_scaler[0]) / (self.state_scaler[1] - self.state_scaler[0])
        history_ws = (history_ws - self.action_scaler[0]) / (self.action_scaler[1] - self.action_scaler[0])
        action, log = self.solver.solve_mpc(history_tc, history_ws, target, initial_ws)
        return action, log


def main(file_src, state_order, action_order, model_filename, horizon):
    # Setting
    state_dim = 140
    action_dim = 40
    alpha = 0  # Workset smoothness
    time_limit = 5  # seconds

    m = get_multi_linear_residual_model(state_dim, action_dim, state_order, action_order)
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()

    scaler = (20.0, 420.0)
    print('Min. state scaler: {}, Max. state scaler: {}'.format(scaler[0], scaler[1]))
    print('Min. action scaler: {}, Max. action scaler: {}'.format(scaler[0], scaler[1]))

    validate_states, validate_actions, _ = load_data(paths=file_src,
                                                     scaling=False,
                                                     preprocess=True,
                                                     history_x=state_order,
                                                     history_u=action_order,
                                                     device=device)
    validate_states = validate_states[0][:, :state_dim]
    validate_actions = validate_actions[0]
    print(validate_states.shape)
    print(validate_actions.shape)
    optimizer_mode = 'LBFGS'

    heatup150 = 545
    stable150 = 181
    anneal150 = 182

    initial_temp = 150.0
    heatup_times = [heatup150]
    anneal_times = [stable150 + anneal150 + horizon]
    target_temp = [375.0]
    target = generate_reference(initial_temp=initial_temp,
                                heatup_times=heatup_times,
                                anneal_times=anneal_times,
                                target_temps=target_temp)
    target = torch.reshape(torch.tensor(target).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))  # [908 x 140]
    target = (target - scaler[0]) / (scaler[1] - scaler[0])  # Change scale into 0-1
    # initial_ws = None
    H = horizon
    T = target.shape[0]

    # history_tc = np.ones((state_order, state_dim)) * 150  # Please fill the real data, 0~139: glass TC
    # history_ws = np.ones((action_order - 1, action_dim)) * 150  # Please fill the real data, 0~39: workset
    runner = Runner(m=m, optimizer_mode=optimizer_mode, state_scaler=scaler, action_scaler=scaler, alpha=alpha,
                    timeout=time_limit)
    log_history = []
    optimized_workset = []
    for t in range(T - H):
        print("Now time [{}] / [{}]".format(t, T - H))
        start = time.time()
        action, log = runner.solve(validate_states[t:t + state_order], validate_actions[t:t + action_order - 1],
                                   target[t:t + H, :])  # [1 x 40] torch.Tensor, use this workset to the furnace
        end = time.time()
        log_history.append(log)
        # initial_ws = torch.cat([action[1:], action[-1:]], dim=0)
        now_target = target[t].mean() * (scaler[1] - scaler[0]) + scaler[0]
        workset = action[0:1, :]
        workset = workset * (scaler[1] - scaler[0]) + scaler[0]
        optimized_workset.append(workset)
        print('Time computation : {}'.format(end - start))
        print('Target tc is {}'.format(now_target))
        print('Average ws is {}'.format(workset.mean()))
    optimized_workset = torch.stack(optimized_workset)
    original_workset = validate_actions[action_order - 1:action_order - 1 + T - H]
    torch.save(original_workset, 'validate_workset/multistep_linear_' + str(horizon) + '_original_WS.pt')
    torch.save(optimized_workset, 'validate_workset/multistep_linear_' + str(horizon) + '_optimized_WS.pt')
    with open('validate_workset/multistep_linear_'+str(horizon)+'_control_log.txt', 'wb') as f:
        pickle.dump(log_history, f)


if __name__ == '__main__':
    validate_srcs = ['experiment_result/Multistep_Linear_13.csv', 'experiment_result/Multistep_Linear_14.csv']
    # 'experiment_result/Multistep_Linear_15.csv']
    Hs = [50, 50]
    state_orders = [50, 50]
    action_orders = [50, 50]
    model_filenames = ['model/Multistep_linear/model_res_5050.pt', 'model/Multistep_linear/model_res_5050.pt']
    for i in range(1):
        main(validate_srcs[i], state_orders[i], action_orders[i], model_filenames[i], Hs[i])
