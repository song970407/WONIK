import torch
import numpy as np
import time
from box import Box

from src.model.get_model import get_reparam_multi_linear_model
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

    def solve(self, history_tc, history_ws, target):
        """
        :param history_tc: state_order x num_state
        :param history_ws: (action_order-1) x num_action
        :param target: H x num_action
        :return:
        """
        # history_tc = torch.from_numpy(history_tc).float().to(device)
        # history_ws = torch.from_numpy(history_ws).float().to(device)
        history_tc = (history_tc - self.state_scaler[0]) / (self.state_scaler[1] - self.state_scaler[0])
        history_ws = (history_ws - self.action_scaler[0]) / (self.action_scaler[1] - self.action_scaler[0])
        action = self.solver.solve_mpc(history_tc, history_ws, target)
        return action()[0:1, :]


def main(file_src, horizon):
    # Setting
    state_dim = 140
    action_dim = 40
    state_order = 5
    action_order = 5
    alpha = 0.001  # Workset smoothness
    time_limit = 5  # seconds
    train_data_path = ['docs/new_data/expert/data_1.csv', 'docs/new_data/expert/data_2.csv']

    m = get_reparam_multi_linear_model(state_dim, action_dim, state_order, action_order)
    model_filename = 'model/Multistep_linear/model_reparam_55.pt'
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()

    train_states, train_actions, info = load_data(paths=train_data_path,
                                                  scaling=True,
                                                  preprocess=True,
                                                  history_x=state_order,
                                                  history_u=action_order)
    scaler = (info['scale_min'].item(), info['scale_max'].item())
    print(train_states[0].shape)
    print(train_actions[0].shape)
    print('Min. state scaler: {}, Max. state scaler: {}'.format(scaler[0], scaler[1]))
    print('Min. action scaler: {}, Max. action scaler: {}'.format(scaler[0], scaler[1]))

    validate_states, validate_actions, _ = load_data(paths=file_src,
                                                     scaling=False,
                                                     preprocess=True,
                                                     history_x=state_order,
                                                     history_u=action_order)
    validate_states = validate_states[0][:, :state_dim]
    validate_actions = validate_actions[0]
    print(validate_states.shape)
    print(validate_actions.shape)
    optimizer_mode = 'LBFGS'

    initial_temp = 150.0
    heatup_times = [545]
    anneal_times = [363]
    target_temp = [375.0]
    target = generate_reference(initial_temp=initial_temp,
                                heatup_times=heatup_times,
                                anneal_times=anneal_times,
                                target_temps=target_temp)
    target = torch.reshape(torch.tensor(target).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))  # [908 x 140]
    target = (target - scaler[0]) / (scaler[1] - scaler[0])  # Change scale into 0-1

    H = horizon
    T = target.shape[0]

    history_tc = np.ones((state_order, state_dim)) * 150  # Please fill the real data, 0~139: glass TC
    history_ws = np.ones((action_order - 1, action_dim)) * 150  # Please fill the real data, 0~39: workset
    runner = Runner(m=m, optimizer_mode=optimizer_mode, state_scaler=scaler, action_scaler=scaler, alpha=alpha,
                    timeout=time_limit)

    """
    x0 = torch.zeros((state_order, state_dim)).to(device)
    u0 = torch.zeros((action_order-1, action_dim)).to(device)
    us = torch.zeros((H, action_dim)).to(device)
    sample_predicted = m.multi_setp_prediction(x0, u0, us)
    print(sample_predicted.shape)
    """
    optimized_workset = []
    for t in range(T - H):
        print("Now time [{}] / [{}]".format(t, T - H))
        start = time.time()
        workset = runner.solve(validate_states[t:t + state_order], validate_actions[t:t + action_order - 1],
                               target[t:t + H, :])  # [1 x 40] torch.Tensor, use this workset to the furnace
        end = time.time()
        optimized_workset.append(workset)
        print('Time computation : {}'.format(end - start))
        print('Target tc is {}'.format(target[t:t + H, :].mean()))
        print('Average ws is {}'.format(workset.mean()))


if __name__ == '__main__':
    validate_srcs = ['experiment_result/Multistep_Linear_01.csv', 'experiment_result/Multistep_Linear_02.csv',
                     'experiment_result/Multistep_Linear_03.csv']
    Hs = [10, 50, 75]
    for i in range(len(validate_srcs)):
        main(validate_srcs[i], Hs[i])
