import torch
import numpy as np
import time
from box import Box

from GraphPreCo import get_model
from src.control.torch_mpc import TorchMPC
from src.utils.PreCo.get_graph import get_hetero_graph
from src.utils.reference_generator import generate_reference
from src.utils.PreCo.load_data import load_data

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
        history_tc = torch.from_numpy(history_tc).float().to(device)
        history_ws = torch.from_numpy(history_ws).float().to(device)
        history_tc = (history_tc - self.state_scaler[0]) / (self.state_scaler[1] - self.state_scaler[0])
        history_ws = (history_ws - self.action_scaler[0]) / (self.action_scaler[1] - self.action_scaler[0])
        action = self.solver.solve_mpc(self.g, (history_tc, history_ws), target)
        return action()[:, 0:1, :]

def main():
    # Setting
    config_filename = 'model/ICPreCo/model_config.yaml'
    config = Box.from_yaml(filename=config_filename)
    m = get_model(config).to(device)
    model_filename = 'model/ICPreCo/model.pt'
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()

    history_len = config.train.history_len  # 20
    future_len = config.train.future_len  # 20
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

    state_scaler = train_info['state_scaler']
    action_scaler = train_info['action_scaler']
    state_scaler = (state_scaler[0].item(), state_scaler[1].item())
    action_scaler = (action_scaler[0].item(), action_scaler[1].item())
    print('Min. state scaler: {}, Max. state scaler: {}'.format(state_scaler[0], state_scaler[1]))
    print('Min. action scaler: {}, Max. action scaler: {}'.format(action_scaler[0], action_scaler[1]))

    H = 10  # receding horizon
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
    target = torch.reshape(torch.tensor(target).to(device), shape=(1, -1)).repeat(repeats=(180, 1)) # [180 x 908]
    target = (target - state_scaler[0]) / (state_scaler[1] - state_scaler[0]) # Normalized


    T = target.shape[1]

    history_tc = np.ones((180, history_len, 1)) * 150  # Please fill the real data, 0~139: glass TC, 140~179: control TC
    history_ws = np.ones((40, history_len-1, 1)) * 150  # Please fill the real data, 0~39: Zone Workset

    runner = Runner(g=g, m=m,
                    optimizer_mode=optimizer_mode, state_scaler=state_scaler, action_scaler=action_scaler,
                    alpha=alpha, timeout=time_limit)
    worksets = []
    observed_tcs = []
    T = 50
    for t in range(T-H):
        print("Now time [{}] / [{}]".format(t, T-H))
        start = time.time()
        workset = runner.solve(history_tc, history_ws, target[:, t:t+H])  # [40 x 1] torch.Tensor, use this workset to the furnace
        end = time.time()
        print('Time computation : {}'.format(end-start))
        with g.local_scope():
            with torch.no_grad():
                x0 = torch.from_numpy(history_tc).float().to(device)
                u0 = torch.from_numpy(history_ws).float().to(device)
                x0 = (x0 - state_scaler[0]) / (state_scaler[1] - state_scaler[0])
                u0 = (u0 - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
                h = m.filter_history(g, x0, u0)
                observed_tc = m.multi_step_prediction(g, h, workset).cpu().detach().numpy()
                observed_tc = observed_tc[:, :, 0:1]  # [180 x 1 x 1] numpy.array
                observed_tc = observed_tc * (state_scaler[1] - state_scaler[0]) + state_scaler[0]  # unscaling
        workset = workset * (action_scaler[1] - action_scaler[0]) + action_scaler[0]
        workset = workset.cpu().detach().numpy()  # [40 x 1 x 1] numpy.array
        now_target = target[0, t] * (state_scaler[1] - state_scaler[0]) + state_scaler[0]
        print('Target tc is {}'.format(now_target))
        print('Average tc is {}'.format(observed_tc.mean()))
        print('Average ws is {}'.format(workset.mean()))
        observed_tcs.append(observed_tc)
        worksets.append(workset)
        history_tc = np.concatenate([history_tc[:, 1:, :], observed_tc], axis=1)
        history_ws = np.concatenate([history_ws[:, 1:, :], workset], axis=1)
    observed_tcs = np.stack(observed_tcs)
    worksets = np.stack(worksets)
    print(observed_tcs.shape)
    print(worksets.shape)

if __name__ == '__main__':
    main()
