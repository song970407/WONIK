import pickle
import torch

with open('control_log.txt', 'rb') as f:
    data = pickle.load(f)
print(data[0])
print(data[0]['trajectory_us_value'][0].shape)
