import torch
import numpy as np
import time
from box import Box

from GraphPreCo import get_model
from src.control.torch_mpc import TorchMPC
from src.utils.PreCo.get_graph import get_hetero_graph
from src.utils.reference_generator import generate_reference
from src.utils.PreCo.load_data import load_data, load_validate_data

# Check whether the GPU (CUDA) is available or not
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class Runner:
    def __init__(self, g, m, optimizer_mode, state_scaler, action_scaler, alpha, timeout=30):
        self.solver = TorchMPC(model=m,
                               time_aggregator='sum',
                               optimizer_mode=optimizer_mode,
                               alpha=alpha,
                               timeout=timeout,
                               device=device)
        self.g = g
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler

    def solve(self, history_tc, history_ws, target):
        """
        :param history_tc: torch.Tensor, [180 x 20 x 1], history of TCs
        :param history_ws: torch.Tensor, [40 x 19 x 1], history of WSs
        :param target: torch.Tensor, [180 x H], target temperature of TCs
        :return:
        """
        history_tc = (history_tc - self.state_scaler[0]) / (self.state_scaler[1] - self.state_scaler[0])
        history_ws = (history_ws - self.action_scaler[0]) / (self.action_scaler[1] - self.action_scaler[0])
        action = self.solver.solve_mpc(self.g, (history_tc, history_ws), target)
        return action()[:, 0:1, :]


def main(file_src, horizon, idx):
    # Setting
    config_filename = 'model/ICPreCo/model_config.yaml'
    config = Box.from_yaml(filename=config_filename)
    m = get_model(config).to(device)
    model_filename = 'model/ICPreCo/model.pt'
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()

    history_len = config.train.history_len  # 20
    future_len = config.train.future_len  # 20
    future_len = 10  # 10
    action_ws = config.data.action_ws if config.data.get('action_ws') else True

    g = get_hetero_graph(glass_tc_pos_path=config.data.g_tc_pos,
                         control_tc_pos_path=config.data.c_tc_pos,
                         t2t_threshold=config.data.t2t_threshold)
    g = g.to(device)
    _, _, _, _, train_info = load_data(paths=config.data.train_path,
                                       history_len=history_len,
                                       future_len=future_len,
                                       action_ws=action_ws,
                                       scaling=config.data.scaling)

    hist_x, future_x, hist_u, future_u, _ = load_validate_data(paths=file_src,
                                                               history_len=history_len,
                                                               future_len=horizon,
                                                               action_ws=action_ws,
                                                               scaling=False,
                                                               device=device)
    print(hist_x.shape)
    print(future_x.shape)
    print(hist_u.shape)
    print(future_u.shape)
    state_scaler = train_info['state_scaler']
    action_scaler = train_info['action_scaler']
    state_scaler = (state_scaler[0].item(), state_scaler[1].item())
    action_scaler = (action_scaler[0].item(), action_scaler[1].item())
    print('Min. state scaler: {}, Max. state scaler: {}'.format(state_scaler[0], state_scaler[1]))
    print('Min. action scaler: {}, Max. action scaler: {}'.format(action_scaler[0], action_scaler[1]))

    H = horizon  # receding horizon
    alpha = 1  # workset smoothness
    time_limit = 5  # seconds
    optimizer_mode = 'LBFGS'

    initial_temp = 150.0
    heatup_times = [545]
    anneal_times = [363]
    target_temp = [375.0]
    target = generate_reference(initial_temp=initial_temp,
                                heatup_times=heatup_times,
                                anneal_times=anneal_times,
                                target_temps=target_temp)
    target = torch.reshape(torch.tensor(target).to(device), shape=(1, -1)).repeat(repeats=(180, 1))  # [180 x 908]
    target = (target - state_scaler[0]) / (state_scaler[1] - state_scaler[0])  # Normalized

    T = target.shape[1]

    history_tc = np.ones((180, history_len, 1)) * 150  # Please fill the real data, 0~139: glass TC, 140~179: control TC
    history_ws = np.ones((40, history_len - 1, 1)) * 150  # Please fill the real data, 0~39: Zone Workset

    runner = Runner(g=g, m=m,
                    optimizer_mode=optimizer_mode, state_scaler=state_scaler, action_scaler=action_scaler,
                    alpha=alpha, timeout=time_limit)
    optimized_worksets = []

    for t in range(hist_x.shape[0]):
        print("Now time [{}] / [{}]".format(t, hist_x.shape[0]))
        start = time.time()
        workset = runner.solve(hist_x[t], hist_u[t],
                               target[:, t:t + H])  # [40 x 1] torch.Tensor, use this workset to the furnace
        end = time.time()
        workset = workset * (action_scaler[1] - action_scaler[0]) + action_scaler[0]
        print('Time computation : {}'.format(end - start))
        print('Average ws is {}'.format(workset.mean()))
        optimized_worksets.append(workset)
    optimized_worksets = torch.stack(optimized_worksets)
    torch.save(optimized_worksets, 'validate_workset/ICPreCo_2'+str(idx+1)+'_WS.pt')

if __name__ == '__main__':
    validate_srcs = ['experiment_result/ICPreCo_01.csv', 'experiment_result/ICPreCo_02.csv']
    Hs = [10, 70]
    for i in range(len(validate_srcs)):
        main(validate_srcs[i], Hs[i], i)
