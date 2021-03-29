import pickle
import numpy as np
import matplotlib.pyplot as plt


def main(state_order, action_order, model_filename):
    with open('simulation_data/' + str(state_order) + '/control_log.txt', 'rb') as f:
        control_log = pickle.load(f)
    print(len(control_log))
    print(len(control_log[0]))
    print(control_log[0].keys())
    trajectory_tc = np.load('simulation_data/' + str(state_order) + '/trajectory_tc.npy')
    trajectory_ws = np.load('simulation_data/' + str(state_order) + '/trajectory_ws.npy')
    # print(trajectory_tc.shape)
    # print(trajectory_ws.shape)
    # num_iters = []
    # for j in range(len(control_log)):
    #    num_iters.append(len(control_log[j]['trajectory_us_gradient']))
    # plt.figure(state_order)
    # plt.plot(trajectory_tc)
    # plt.title('Glass TC, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    # plt.show()
    # plt.figure(state_order + 1)
    # plt.plot(trajectory_ws)
    # plt.title('Work Set, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    # plt.show()
    # plt.figure(state_order + 2)
    # plt.plot(num_iters)
    # plt.title('Number of Iterations, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    # plt.show()


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
    H = 50

    for i in range(len(state_orders)):
        main(state_orders[i], action_orders[i], model_filenames[i])
