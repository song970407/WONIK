import pickle
import numpy as np
import matplotlib.pyplot as plt


def main(state_order, action_order, alpha):
    with open('simulation_data/multistep_linear_residual_fast_previous/' + str(state_order) + alpha + '/control_log.txt', 'rb') as f:
        control_log = pickle.load(f)
    print(len(control_log))
    print(len(control_log[0]))
    print(control_log[0].keys())
    trajectory_tc = np.load(
        'simulation_data/multistep_linear_residual_fast_previous/' + str(state_order) + alpha + '/trajectory_tc.npy')
    trajectory_ws = np.load(
        'simulation_data/multistep_linear_residual_fast_previous/' + str(state_order) + alpha + '/trajectory_ws.npy')
    # print(trajectory_tc.shape)
    # print(trajectory_ws.shape)
    num_iters = []
    traj_loss_obj = []
    traj_loss_del = []
    times = []
    for j in range(len(control_log)):
        num_iters.append(len(control_log[j]['trajectory_us_gradient']))
        traj_loss_obj.append(control_log[j]['trajectory_loss_objective'].cpu().detach().numpy())
        traj_loss_del.append(control_log[j]['trajectory_loss_delta_u'].cpu().detach().numpy())
        times.append(control_log[j]['total_time'])
    traj_loss_obj = np.stack(traj_loss_obj, axis=-1)
    traj_loss_del = np.stack(traj_loss_del, axis=-1)
    plt.figure(state_order)
    plt.plot(trajectory_tc)
    plt.title('Glass TC, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    plt.show()
    plt.figure(state_order + 1)
    plt.plot(trajectory_ws)
    plt.title('Work Set, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    plt.show()
    # plt.figure(state_order + 2)
    # plt.plot(num_iters)
    # plt.title('# of Optimization Iterations, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    # plt.show()
    # plt.figure(state_order + 3)
    # plt.plot(traj_loss_obj)
    # plt.title('Objective Loss, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    # plt.show()
    # plt.figure(state_order + 4)
    # plt.plot(traj_loss_del)
    # plt.title('Delta Work Set Loss, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    # plt.show()
    plt.figure(state_order + 123)
    plt.plot(traj_loss_obj + traj_loss_del)
    plt.title('Loss, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
    plt.show()
    # plt.figure(state_order + 12345)
    # plt.plot(times)
    # plt.title('Computation time, State Order: ' + str(state_order) + ' Action Order: ' + str(action_order))
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
    state_orders = [50, 50, 50, 50, 50]
    action_orders = [50, 50, 50, 50, 50]
    alphas = ['_0', '_0.01', '_0.1', '_1', '_10']
    model_filenames = ['model/Multistep_linear/residual_model/model_50.pt']

    for i in range(len(alphas)):
        main(state_orders[i], action_orders[i], alphas[i])
