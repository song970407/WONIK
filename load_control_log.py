import pickle

with open('control_log.txt', 'rb') as f:
    load_log = pickle.load(f)
print(len(load_log))
print(len(load_log[0]))
print(load_log[0].keys())
for i in load_log[0].keys():
    print(load_log[0][i])

with open('control_log_test.txt', 'rb') as f:
    load_log = pickle.load(f)
print(len(load_log))
print(len(load_log[0]))
print(load_log[0].keys())
for i in load_log[0].keys():
    print(load_log[0][i])