from os.path import join

import torch
import wandb
from box import Box
from torch.utils.data import TensorDataset, DataLoader

from src.model.LinearStateSpaceModels import LinearSSM
from src.utils.adjacent_mat_utils import get_adj_matrix
from src.utils.data_preprocess import get_xuy
from src.utils.load_data import load_data


def get_config():
    conf = Box({
        'scaling': True,
        'train_time_window': 50,
        'perturb_x_param': 1e-3,
        'batch_size': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epoch': 500,
        'test_every': 25,
        'lr': 2 * 1e-4,
        'lambda_A': 1e-6,
        'lambda_B': 1e-3
    })
    return conf


def train(config):
    wandb.init(project='graph_ssm',
               config=config.to_dict())

    # save config
    config.to_yaml(join(wandb.run.dir, "exp_config.yaml"))

    train_data_path = './docs/100_slm_data/data_1.csv'
    train_states, train_actions, info = load_data(train_data_path,
                                                  scaling=config.scaling)

    test_data_path = './docs/100_slm_data/data_2.csv'
    test_scaler = (info['scale_min'], info['scale_max'])
    test_states, test_actions, info = load_data(test_data_path,
                                                scaling=config.scaling,
                                                scaler=test_scaler)

    state_dim = train_states.shape[1]
    action_dim = train_actions.shape[1]

    adj_xx = torch.eye(state_dim)
    glass_tc_pos_path = './docs/location/1_glass_Tc_r1.csv'
    control_tc_pos_path = './docs/location/5_controlTC_r1.csv'
    threshold = 1200
    weight = (1., 1., 10.0)

    adj_xu = get_adj_matrix(glass_tc_pos_path,
                            control_tc_pos_path,
                            threshold,
                            weight)

    m = LinearSSM(state_dim, action_dim, adj_xx, adj_xu).to(config.device)
    wandb.watch(m)
    criteria = torch.nn.SmoothL1Loss()  # Huberloss
    opt = torch.optim.Adam(m.parameters(), lr=2 * 1e-4)

    train_xs, train_us, train_ys = get_xuy(train_states,
                                           train_actions,
                                           config.train_time_window,
                                           config.device)

    test_xs, test_us, test_ys = get_xuy(test_states,
                                        test_actions,
                                        test_states.shape[0] - 1,
                                        config.device)

    ds = TensorDataset(train_xs, train_us, train_ys)
    train_loader = DataLoader(ds, batch_size=config.batch_size)

    n_update = 0
    for ep in range(config.epoch):
        print("Epoch [{}] / [{}]".format(ep, config.epoch))
        for x, u, y in train_loader:
            x = x + (torch.randn_like(x) * config.perturb_x_param).to(config.device)
            pred = m.rollout(x, u)
            pred_loss = criteria(pred, y)

            # Regularization

            A_2_norm = torch.norm(m.A.weight, p=2)
            B_1_norm = torch.norm(m.B.weight, p=1)
            loss = pred_loss + B_1_norm * config.lambda_B - A_2_norm * config.lambda_A

            # Update
            opt.zero_grad()
            loss.backward()
            opt.step()

            n_update += 1

            # Clip params
            m.A.clip_params(min=0.0, max=1.0)
            m.B.clip_params(min=0.0, max=1.0)

            log_dict = dict()
            log_dict['train_loss'] = pred_loss
            log_dict['A_2_norm'] = A_2_norm
            log_dict['B_1_norm'] = B_1_norm

            if n_update % config.test_every == 0:
                test_pred = m.rollout(test_xs, test_us)
                test_pred_loss = criteria(test_pred, test_ys)
                log_dict['test_loss'] = test_pred_loss

            wandb.log(log_dict)
    torch.save(m.state_dict(), join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    train(get_config())
