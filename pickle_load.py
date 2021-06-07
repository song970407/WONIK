import pickle
import numpy as np
import torch

with open('./log/control_log.txt', 'rb') as fr:
    data = pickle.load(fr)
    tensor1 = np.array(data[0])
print(tensor1) # same as step count


with open('control_log_test.txt', 'rb') as f:
    load_log = pickle.load(f)
print(len(load_log))
print(len(load_log[0]))
print(load_log[0].keys())
print(len(load_log[0]['trajectory_us_value']))
print(len(load_log[0]['trajectory_loss_objective']))
for i in range(len(load_log)):
    # print(load_log[i]['trajectory_loss_objective'])
    # print(load_log[i]['total_time'])
    print('Initial')
    print(load_log[i]['trajectory_us_value'][0][:4])
    print('Last')
    print(load_log[i]['trajectory_us_value'][-1][:4])