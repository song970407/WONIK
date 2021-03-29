import torch
import matplotlib.pyplot as plt

from src.model.get_model import get_reparam_multi_linear_model, get_multi_linear_model, get_multi_linear_residual_model
from src.utils.load_data import load_data
from src.utils.data_preprocess import get_data
from src.control.torch_mpc import LinearTorchMPC

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def main(state_order, action_order, model_filename):
    state_dim = 140
    action_dim = 40
    # m = get_reparam_multi_linear_model(state_dim, action_dim, state_order, action_order).to(device)
    # m = get_multi_linear_model(state_dim, action_dim, state_order, action_order).to(device)
    m = get_multi_linear_residual_model(state_dim, action_dim, state_order, action_order).to(device)
    m.load_state_dict(torch.load(model_filename, map_location=device))
    m.eval()

    scaler = (20.0, 420.0)

    print('Min. state scaler: {}, Max. state scaler: {}'.format(scaler[0], scaler[1]))
    print('Min. action scaler: {}, Max. action scaler: {}'.format(scaler[0], scaler[1]))

    test_data_path = ['docs/new_data/overshoot/data_1.csv']
    # test_data_path = ['docs/new_data/icgrnn/data_3.csv']

    test_states, test_actions, _ = load_data(paths=test_data_path,
                                             scaling=True,
                                             scaler=scaler,
                                             preprocess=True,
                                             history_x=state_order,
                                             history_u=action_order)

    # print(test_states[0].shape)
    # print(test_actions[0].shape)

    rollout_window = 2400
    glass_tc_path = 'docs/new_location/glass_TC.csv'
    control_tc_path = 'docs/new_location/control_TC.csv'

    history_xs, history_us, us, ys, _ = get_data(states=test_states,
                                                 actions=test_actions,
                                                 rollout_window=rollout_window,
                                                 history_x_window=state_order,
                                                 history_u_window=action_order,
                                                 glass_tc_pos_path=glass_tc_path,
                                                 control_tc_pos_path=control_tc_path,
                                                 num_glass_tc=140,
                                                 num_control_tc=40,
                                                 device=device)

    history_xs = history_xs[0].transpose(1, 2)
    history_us = history_us.transpose(1, 2)
    us = us.transpose(1, 2)
    ys = ys[0].transpose(1, 2)

    predicted_ys = []

    with torch.no_grad():
        for idx in range(1):
            print(idx)
            predicted_y = m.multi_step_prediction(history_xs[idx], history_us[idx], us[idx])
            predicted_ys.append(predicted_y)
    predicted_ys = torch.stack(predicted_ys, dim=0)
    ys = ys * (scaler[1] - scaler[0]) + scaler[0]
    predicted_ys = predicted_ys * (scaler[1] - scaler[0]) + scaler[0]
    ys = ys.cpu().detach().numpy()
    predicted_ys = predicted_ys.cpu().detach().numpy()
    print(ys.shape)
    print(predicted_ys.shape)

    us = us * (scaler[1] - scaler[0]) + scaler[0]
    us = us.cpu().detach().numpy()

    return ys, predicted_ys, us


if __name__ == '__main__':
    state_orders = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    action_orders = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    model_filenames = ['model/Multistep_linear/residual_model/model_05.pt',
                       'model/Multistep_linear/residual_model/model_10.pt',
                       'model/Multistep_linear/residual_model/model_15.pt',
                       'model/Multistep_linear/residual_model/model_20.pt',
                       'model/Multistep_linear/residual_model/model_25.pt',
                       'model/Multistep_linear/residual_model/model_30.pt',
                       'model/Multistep_linear/residual_model/model_35.pt',
                       'model/Multistep_linear/residual_model/model_40.pt',
                       'model/Multistep_linear/residual_model/model_45.pt',
                       'model/Multistep_linear/residual_model/model_50.pt']
    yss = []
    predicted_yss = []
    uss = []
    for i in range(len(model_filenames)):
        ys, predicted_ys, us = main(state_orders[i], action_orders[i], model_filenames[i])
        yss.append(ys)
        predicted_yss.append(predicted_ys)
        uss.append(us)
    plt.figure(123)
    plt.plot(yss[0][0])
    plt.ylim([100, 450])
    plt.show()

    for i in range(len(predicted_yss)):
        plt.figure(i)
        plt.title('History Length: '+str(state_orders[i]))
        plt.plot(predicted_yss[i][0])
        plt.ylim([100, 450])
        plt.show()

    plt.figure(12345)
    plt.plot(uss[0][0])
    plt.ylim([100, 450])
    plt.show()
