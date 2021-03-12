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

    def solve(self, history_tc, history_ws, target, weight=None):
        """
        :param history_tc: state_order x num_state
        :param history_ws: (action_order-1) x num_action
        :param target: H x num_action
        :return:
        """
        history_tc = torch.from_numpy(history_tc).float().to(device)
        history_ws = torch.from_numpy(history_ws).float().to(device)
        history_tc = (history_tc - self.state_scaler[0]) / (self.state_scaler[1] - self.state_scaler[0])
        history_ws = (history_ws - self.action_scaler[0]) / (self.action_scaler[1] - self.action_scaler[0])
        action = self.solver.solve_mpc(history_tc, history_ws, target, weight)
        return action()[0:1, :]


def main():
    # Setting
    state_dim = 140
    action_dim = 40
    state_order = 5
    action_order = 5
    alpha = 0  # Workset smoothness
    time_limit = 5  # seconds
    weight_alpha = 5
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

    print('Min. state scaler: {}, Max. state scaler: {}'.format(scaler[0], scaler[1]))
    print('Min. action scaler: {}, Max. action scaler: {}'.format(scaler[0], scaler[1]))

    optimizer_mode = 'LBFGS'

    heatup150 = 545
    stable150 = 181
    anneal150 = 182

    initial_temp = 150.0
    heatup_times = [heatup150]
    anneal_times = [stable150 + anneal150]  # 181 stable + 182 annealing
    target_temp = [375.0]
    target = generate_reference(initial_temp=initial_temp,
                                heatup_times=heatup_times,
                                anneal_times=anneal_times,
                                target_temps=target_temp)
    target = torch.reshape(torch.tensor(target).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))  # [908 x 140]
    target = (target - scaler[0]) / (scaler[1] - scaler[0])  # Change scale into 0-1

    H = 50
    T = target.shape[0]

    weight = torch.stack([torch.ones_like(target), torch.ones_like(target)], dim=-1).to(
        device)  # [908 x 140 x 2], 0: when negative, 1: when positive

    weight[heatup150 + stable150, :, 0] = weight_alpha * weight[heatup150 + stable150, :, 0]
    weight[heatup150:, :, 1] = weight_alpha * weight[heatup150:, :, 1]

    history_tc = np.ones((state_order, state_dim)) * 150  # Please fill the real data, 0~139: glass TC
    history_ws = np.ones((action_order - 1, action_dim)) * 150  # Please fill the real data, 0~39: workset
    runner = Runner(m=m, optimizer_mode=optimizer_mode, state_scaler=scaler, action_scaler=scaler, alpha=alpha,
                    timeout=time_limit)

    """x0 = torch.zeros((state_order, state_dim)).to(device)
    u0 = torch.zeros((action_order-1, action_dim)).to(device)
    us = torch.zeros((H, action_dim)).to(device)
    sample_predicted = m.multi_setp_prediction(x0, u0, us)
    print(sample_predicted.shape)"""

    for t in range(T - H):
        print("Now time [{}] / [{}]".format(t, T - H))
        start = time.time()
        workset = runner.solve(history_tc, history_ws, target[t:t + H, :],
                               weight[t:t + H])  # [1 x 40] torch.Tensor, use this workset to the furnace
        end = time.time()
        print('Time computation : {}'.format(end - start))
        with torch.no_grad():
            x0 = torch.from_numpy(history_tc).float().to(device)
            u0 = torch.from_numpy(history_ws).float().to(device)
            x0 = (x0 - scaler[0]) / (scaler[1] - scaler[0])
            u0 = (u0 - scaler[0]) / (scaler[1] - scaler[0])
            observed_tc = m.multi_step_prediction(x0, u0, workset).cpu().detach().numpy()  # [1 x 140]

            observed_tc = observed_tc * (scaler[1] - scaler[0]) + scaler[0]  # unscaling
        workset = workset * (scaler[1] - scaler[0]) + scaler[0]
        workset = workset.cpu().detach().numpy()  # [1 x 40] numpy.array
        now_target = target[t, 0] * (scaler[1] - scaler[0]) + scaler[0]
        print('Target tc is {}'.format(now_target))
        print('Average tc is {}'.format(observed_tc.mean()))
        print('Average ws is {}'.format(workset.mean()))
        history_tc = np.concatenate([history_tc[1:, :], observed_tc], axis=0)
        history_ws = np.concatenate([history_ws[1:, :], workset], axis=0)


if __name__ == '__main__':
    main()
