import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model.get_model import get_preco_model
from src.utils.load_data import load_preco_data
from src.utils.data_preprocess import get_preco_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(test_data_path,
         state_dim,
         action_dim,
         hidden_dim,
         receding_history,
         receding_horizon,
         state_scaler,
         action_scaler):
    m = get_preco_model(state_dim, action_dim, hidden_dim)
    model_filename = 'model/PreCo.pt'
    m.load_state_dict(torch.load(model_filename, map_location=DEVICE))

    test_states, test_actions, _ = load_preco_data(paths=test_data_path,
                                                   scaling=True,
                                                   state_scaler=state_scaler,
                                                   action_scaler=action_scaler,
                                                   preprocess=True,
                                                   receding_history=receding_history)
    test_history_xs, test_history_us, test_us, test_ys = get_preco_data(states=test_states,
                                                                        actions=test_actions,
                                                                        receding_history=receding_history,
                                                                        receding_horizon=receding_horizon,
                                                                        state_dim=state_dim,
                                                                        action_dim=action_dim,
                                                                        device=DEVICE)
    test_history_xs = test_history_xs[0].transpose(1, 2)
    test_history_us = test_history_us.transpose(1, 2)
    test_us = test_us.transpose(1, 2)
    test_ys = test_ys[0].transpose(1, 2)
    print(test_history_xs.shape)
    print(test_history_us.shape)
    print(test_us.shape)
    print(test_ys.shape)
    BS = test_history_xs.shape[0]
    num_tc = test_history_xs.shape[-1]
    crit = torch.nn.SmoothL1Loss(reduction='mean')
    with torch.no_grad():
        h0 = m.filter_history(test_history_xs, test_history_us)
        ys = m.multi_step_prediction(h0, test_us)
        test_ys = test_ys * (state_scaler[1] - state_scaler[0]) + state_scaler[0]
        ys = ys * (state_scaler[1] - state_scaler[0]) + state_scaler[0]
    num_plot = 10
    for i in range(num_plot):
        rand_bs, rand_tc = np.random.randint(0, BS), np.random.randint(0, num_tc)
        plt.ylim([140.0, 420.0])
        plt.plot(ys[rand_bs, :, rand_tc], label='predicted')
        plt.plot(test_ys[rand_bs, :, rand_tc], label='real')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    state_dim = 140
    action_dim = 40
    hidden_dim = 256

    receding_history = 50
    receding_horizon = 100
    state_scaler = (20.0, 420.0)
    action_scaler = (20.0, 420.0)
    test_data_path = ['docs/new_data/overshoot/data_1.csv']
    main(test_data_path,
         state_dim,
         action_dim,
         hidden_dim,
         receding_history,
         receding_horizon,
         state_scaler,
         action_scaler)