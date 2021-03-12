import torch
import matplotlib.pyplot as plt

from src.model.get_model import get_reparam_multi_linear_model
from src.utils.load_data import load_data
from src.utils.data_preprocess import get_data
from src.control.torch_mpc import LinearTorchMPC

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

state_dim = 140
action_dim = 40
state_order = 5
action_order = 5
alpha = 0.001  # Workset smoothness
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

print(train_states[0].shape)
print(train_actions[0].shape)

test_data_path = ['docs/new_data/icgrnn/data_3.csv']

test_states, test_actions, _ = load_data(paths=test_data_path,
                                         scaling=True,
                                         scaler=scaler,
                                         preprocess=True,
                                         history_x=state_order,
                                         history_u=action_order)

# print(test_states[0].shape)
# print(test_actions[0].shape)

rollout_window = 900
glass_tc_path = 'docs/new_location/glass_TC.csv'
control_tc_path = 'docs/new_location/control_TC.csv'

history_xs, history_us, us, ys, _ = get_data(states=test_states,
                                             actions=test_actions,
                                             rollout_window=rollout_window,
                                             history_x_window=5,
                                             history_u_window=5,
                                             glass_tc_pos_path=glass_tc_path,
                                             control_tc_pos_path=control_tc_path,
                                             num_glass_tc=140,
                                             num_control_tc=40,
                                             device=device)

history_xs = history_xs[0].transpose(1, 2)
history_us = history_us.transpose(1, 2)
us = us.transpose(1, 2)
ys = ys[0].transpose(1, 2)
# ys[1] = ys[1].transpose(1, 2)
# print(history_xs.shape)
# print(history_us.shape)
# print(us.shape)
# print(ys.shape)

predicted_ys = []

with torch.no_grad():
    for idx in range(history_xs.shape[0]):
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

plt.figure(123)
plt.plot(ys[0])
plt.show()

plt.figure(12)
plt.plot(predicted_ys[0])
plt.show()

plt.figure(1234)
plt.plot(us[0])
plt.show()