import pickle
import numpy as np

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
H = 50

for i in range(len(state_orders)):
    with open('simulation_data/' + str(state_orders[i]) + '/control_log.txt', 'rb') as f:
        control_log = pickle.load(f)
    print(len(control_log))
    print(len(control_log[0]))
    print(len(control_log[0].keys()))
    trajectory_tc = np.load('simulation_data/' + str(state_orders[i]) + '/trajectory_tc.npy')
    trajectory_ws = np.load('simulation_data/' + str(state_orders[i]) + '/trajectory_ws.npy')
    print(trajectory_tc.shape)
    print(trajectory_ws.shape)
