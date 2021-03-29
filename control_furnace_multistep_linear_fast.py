import torch
import numpy as np
import time
import pickle

from box import Box

from src.model.get_model import get_multi_linear_residual_fast_model, get_multi_linear_residual_model
from src.utils.load_data import load_data
from src.utils.data_preprocess import get_data
from src.control.torch_mpc import LinearTorchMPC

# Check whether the GPU (CUDA) is available or not
from src.utils.reference_generator import generate_reference

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
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
        history_tc = torch.from_numpy(history_tc).float().to(device)
        history_ws = torch.from_numpy(history_ws).float().to(device)
        history_tc = (history_tc - self.state_scaler[0]) / (self.state_scaler[1] - self.state_scaler[0])
        history_ws = (history_ws - self.action_scaler[0]) / (self.action_scaler[1] - self.action_scaler[0])
        action, log = self.solver.solve_mpc(history_tc, history_ws, target, initial_ws)
        return action, log


def main(state_order, action_order, model_filename, H):
    # Setting
    state_dim = 140
    action_dim = 40
    alpha = 1  # Workset smoothness
    time_limit = 5  # seconds

    # m = get_multi_linear_residual_model(state_dim, action_dim, state_order, action_order)
    m = get_multi_linear_residual_fast_model(state_dim, action_dim, state_order, action_order)
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()
    scaler = (20.0, 420.0)

    print('Min. state scaler: {}, Max. state scaler: {}'.format(scaler[0], scaler[1]))
    print('Min. action scaler: {}, Max. action scaler: {}'.format(scaler[0], scaler[1]))

    optimizer_mode = 'LBFGS'

    heatup150 = 545
    stable150 = 181
    anneal150 = 182

    initial_temp = 150.0
    heatup_times = [heatup150]
    anneal_times = [stable150 + anneal150 + H]
    target_temp = [375.0]
    target = generate_reference(initial_temp=initial_temp,
                                heatup_times=heatup_times,
                                anneal_times=anneal_times,
                                target_temps=target_temp)
    target = torch.reshape(torch.tensor(target).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))  # [908 x 140]
    target = (target - scaler[0]) / (scaler[1] - scaler[0])  # Change scale into 0-1

    T = target.shape[0]

    sample_data_path = 'docs/new_data/expert/data_1.csv'
    states, actions, _ = load_data(paths=sample_data_path,
                                   scaling=False,
                                   preprocess=True,
                                   history_x=state_order,
                                   history_u=action_order,
                                   device=device)
    history_tc = states[0][:state_order, :state_dim].cpu().detach().numpy()
    history_ws = actions[0][:action_order - 1].cpu().detach().numpy()

    # history_tc = np.ones((state_order, state_dim)) * 150  # Please fill the real data, 0~139: glass TC
    # history_ws = np.ones((action_order - 1, action_dim)) * 150  # Please fill the real data, 0~39: workset
    initial_ws = target[:H, :action_dim].to(device)  # [H x 40], 0-1 scale
    runner = Runner(m=m, optimizer_mode=optimizer_mode, state_scaler=scaler, action_scaler=scaler, alpha=alpha,
                    timeout=time_limit)

    log_history = []
    trajectory_tc = []
    trajectory_ws = []
    for t in range(T - H):
        print("Now time [{}] / [{}]".format(t, T - H))
        start = time.time()
        # action, log = runner.solve(history_tc, history_ws, target[t:t + H, :],
        #                            initial_ws)  # [1 x 40] torch.Tensor, use this workset to the furnace
        action, log = runner.solve(history_tc, history_ws, target[t:t + H, :])
        print(log['total_time'])
        end = time.time()
        print('Time computation : {}'.format(end - start))
        workset = action[0:1, :]
        initial_ws = torch.cat([action[1:], action[-1:]], dim=0)
        log_history.append(log)
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
        trajectory_tc.append(observed_tc)
        trajectory_ws.append(workset)
    # print(log_history)
    with open('simulation_data/multistep_linear_residual_fast/' + str(state_order) + '_1/control_log.txt',
              'wb') as f:
        pickle.dump(log_history, f)
    trajectory_tc = np.concatenate(trajectory_tc, axis=0)
    trajectory_ws = np.concatenate(trajectory_ws, axis=0)
    np.save('simulation_data/multistep_linear_residual_fast/' + str(state_order) + '_1/trajectory_tc.npy',
            trajectory_tc)
    np.save('simulation_data/multistep_linear_residual_fast/' + str(state_order) + '_1/trajectory_ws.npy',
            trajectory_ws)


if __name__ == '__main__':
    state_orders = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    action_orders = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    model_filenames = ['model/Multistep_linear/residual_fast_model/model_05.pt',
                       'model/Multistep_linear/residual_fast_model/model_10.pt',
                       'model/Multistep_linear/residual_fast_model/model_15.pt',
                       'model/Multistep_linear/residual_fast_model/model_20.pt',
                       'model/Multistep_linear/residual_fast_model/model_25.pt',
                       'model/Multistep_linear/residual_fast_model/model_30.pt',
                       'model/Multistep_linear/residual_fast_model/model_35.pt',
                       'model/Multistep_linear/residual_fast_model/model_40.pt',
                       'model/Multistep_linear/residual_fast_model/model_45.pt',
                       'model/Multistep_linear/residual_fast_model/model_50.pt']
    state_orders = [50]
    action_orders = [50]
    model_filenames = ['model/Multistep_linear/residual_fast_model/model_50.pt']
    H = 50
    for i in range(len(state_orders)):
        main(state_orders[i], action_orders[i], model_filenames[i], H)
