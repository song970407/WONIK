import torch
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from box import Box

from src.model.get_model import get_preco_model
from src.utils.load_data import load_preco_data
from src.utils.data_preprocess import get_preco_data
from src.control.torch_mpc import PreCoTorchMPC

# Check whether the GPU (CUDA) is available or not
from src.utils.reference_generator import generate_reference

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)


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


def main(smooth_u_type, H, alpha, optimizer_mode, initial_solution, max_iter, u_range):
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

    model_filename = 'model/PreCo.pt'
    m = get_preco_model(state_dim, action_dim, hidden_dim)
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()
    state_scaler = (20.0, 420.0)
    action_scaler = (20.0, 420.0)
    receding_history = 50

    print('Min. state scaler: {}, Max. state scaler: {}'.format(state_scaler[0], state_scaler[1]))
    print('Min. action scaler: {}, Max. action scaler: {}'.format(action_scaler[0], action_scaler[1]))

    heatup150 = 545
    stable150 = 181
    anneal150 = 182

    heatup150 = 300
    stable150 = 100
    anneal150 = 100

    initial_temp = 150.0
    heatup_times = [heatup150]
    anneal_times = [stable150 + anneal150 + H]
    target_temp = [375.0]
    target = generate_reference(initial_temp=initial_temp,
                                heatup_times=heatup_times,
                                anneal_times=anneal_times,
                                target_temps=target_temp)
    target = torch.reshape(torch.tensor(target).to(device), shape=(-1, 1)).repeat(repeats=(1, state_dim))  # [908 x 140]
    target = (target - state_scaler[0]) / (state_scaler[1] - state_scaler[0])  # Change scale into 0-1

    T = target.shape[0]

    sample_data_path = 'docs/new_data/expert/data_1.csv'
    states, actions, _ = load_preco_data(paths=sample_data_path,
                                         scaling=False,
                                         preprocess=True,
                                         receding_history=receding_history)
    history_tc = states[0][:receding_history, :state_dim].cpu().detach().numpy()
    history_ws = actions[0][:receding_history - 1].cpu().detach().numpy()
    initial_ws = target[:H, :action_dim].to(device)  # [H x 40], 0-1 scale
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
                    timeout=5.0,
                    is_logging=True)
    log_history = []
    trajectory_tc = []
    trajectory_ws = []
    for t in range(T - H):
        print("Now time [{}] / [{}]".format(t, T - H))
        start = time.time()
        action, log = runner.solve(history_tc, history_ws, target[t:t + H, :], initial_ws)
        print(log['total_time'])
        end = time.time()
        print('Time computation : {}'.format(end - start))
        workset = action[0:1, :]
        initial_ws = torch.cat([action[1:], action[-1:]], dim=0)
        log_history.append(log)
        with torch.no_grad():
            x0 = torch.from_numpy(history_tc).float().to(device).unsqueeze(dim=0)
            u0 = torch.from_numpy(history_ws).float().to(device).unsqueeze(dim=0)
            x0 = (x0 - state_scaler[0]) / (state_scaler[1] - state_scaler[0])
            u0 = (u0 - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
            h0 = m.filter_history(x0, u0)
            observed_tc = m.multi_step_prediction(h0, workset.unsqueeze(dim=0)) # [1 x 140]
            observed_tc = observed_tc * (state_scaler[1] - state_scaler[0]) + state_scaler[0]  # unscaling
            observed_tc = observed_tc.cpu().detach().numpy()
        workset = workset * (action_scaler[1] - action_scaler[0]) + action_scaler[0]
        workset = workset.cpu().detach().numpy()  # [1 x 40] numpy.array
        now_target = target[t, 0] * (state_scaler[1] - state_scaler[0]) + state_scaler[0]
        print('Target tc is {}'.format(now_target))
        print('Average tc is {}'.format(observed_tc.mean()))
        print('Average ws is {}'.format(workset.mean()))
        history_tc = np.concatenate([history_tc[1:, :], observed_tc[0]], axis=0)
        history_ws = np.concatenate([history_ws[1:, :], workset], axis=0)
        trajectory_tc.append(observed_tc[0])
        trajectory_ws.append(workset)
    # print(log_history)
    # with open('simulation_data/PreCo/control_log.txt', 'wb') as f:
        # pickle.dump(log_history, f)
    trajectory_tc = np.concatenate(trajectory_tc, axis=0)
    trajectory_ws = np.concatenate(trajectory_ws, axis=0)
    # np.save('simulation_data/PreCo/trajectory_tc.npy', trajectory_tc)
    # np.save('simulation_data/PreCo/trajectory_ws.npy', trajectory_ws)
    plt.plot(trajectory_tc)
    plt.show()
    plt.plot(trajectory_ws)
    plt.show()


if __name__ == '__main__':
    smooth_u_type = 'boundary'  # penalty or constraint or boundary, cannot be list
    # Hyper-parameters for MPC optimizer
    H = 100  # Receding horizon
    alpha = 1000  # will be ignored if smooth_u_type == constraint or boundary
    optimizer_mode = 'Adam'  # Adam or LBFGS
    initial_solution = 'previous'  # target or previous
    max_iter = 100  # Maximum number of optimize  r iterations
    u_range = 0.03  # will be ignored if smooth_u_type == penalty, cannot be list
    main(smooth_u_type, H, alpha, optimizer_mode, initial_solution, max_iter, u_range)
