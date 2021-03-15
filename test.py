import torch
import pickle

def save_list():
    a = torch.ones(4)
    b = torch.zeros(3)

    c = [a, b]

    with open('test_list.txt', 'wb') as f:
        pickle.dump(c, f)

def load_list():
    with open('test_list.txt', 'rb') as f:
        data = pickle.load(f)
    a = data[0]
    return a

if __name__ == '__main__':
    output = load_list()
    print(output)
    print(output*2)