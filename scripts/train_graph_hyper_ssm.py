from os.path import join

import torch
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from box import Box
from torch.utils.data import TensorDataset, DataLoader

from src.nn.MLP import MultiLayerPerceptron as MLP
from src.model.LinearStateSpaceModels import HyperLinearSSM
from src.utils.adjacent_mat_utils import get_adj_matrix
from src.utils.data_preprocess import get_xuy
from src.utils.load_data import load_data


def get_config():
    conf = Box({
        'scaling': True,
        'action_ws': False,
        'preprocess': False,
        'train_time_window': 15,
        'perturb_x_param': 1e-3,
        'batch_size': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epoch': 500,
        'test_every': 25,
        'lr': 1e-3,
        'lr_scheduler': {
            'name': 'CosineAnnealingWarmRestarts',
            'T_0': 5,
        },
        'wandb': {
            'group': 'HyperLinearSSM'
        },
        'nn': {
            'a_mlp': {'num_neurons': [128, 128],
                      'use_residual': True,
                      'normalization': 'batch',
                      'out_activation': 'Clip1'},
            'b_mlp': {'num_neurons': [128, 128],
                      'use_residual': True,
                      'normalization': 'batch',
                      'out_activation': 'Clip1'}
        },
        'adjmat': {
            'threshold': 1200,
            'weight': (1., 1., 10.0)
        }
    })
    return conf


def train(config):
    wandb.init(project='graph_ssm',
               entity='sentinel',
               group=config.wandb.group,
               config=config.to_dict())

    # save config
    config.to_yaml(join(wandb.run.dir, "exp_config.yaml"))

    train_data_path = './docs/100_slm_data/data_1.csv'
    train_states, train_actions, info = load_data(train_data_path,
                                                  scaling=config.scaling,
                                                  action_ws=config.action_ws,
                                                  preprocess=config.preprocess)

    test_data_path = './docs/100_slm_data/data_2.csv'
    test_scaler = (info['scale_min'], info['scale_max'])
    test_states, test_actions, info = load_data(test_data_path,
                                                scaling=config.scaling,
                                                scaler=test_scaler,
                                                action_ws=config.action_ws,
                                                preprocess=config.preprocess)

    state_dim = train_states.shape[1]
    action_dim = train_actions.shape[1]

    adj_xx = torch.eye(state_dim)
    glass_tc_pos_path = './docs/location/1_glass_Tc_r1.csv'
    control_tc_pos_path = './docs/location/5_controlTC_r1.csv'
    threshold = config.adjmat.threshold
    weight = config.adjmat.weight

    adj_xu = get_adj_matrix(glass_tc_pos_path,
                            control_tc_pos_path,
                            threshold,
                            weight)
    a_mlp = MLP(state_dim, adj_xx.nonzero(as_tuple=True)[0].size(0),
                **config.nn.a_mlp)
    b_mlp = MLP(action_dim, adj_xu.nonzero(as_tuple=True)[0].size(0),
                **config.nn.b_mlp)
    m = HyperLinearSSM(state_dim, action_dim, a_mlp, b_mlp, adj_xx, adj_xu).to(config.device)
    wandb.watch(m)
    criteria = torch.nn.SmoothL1Loss()  # Huberloss
    opt = torch.optim.Adam(m.parameters(), lr=config.lr)

    # setup LR scheduler
    use_lr_schedule = config.lr_scheduler
    if use_lr_schedule:
        scheduler_name = config.lr_scheduler.pop('name')
        scheduler = getattr(lr_scheduler, scheduler_name)(opt,
                                                          **config.lr_scheduler.to_dict())

    train_xs, train_us, train_ys = get_xuy(train_states,
                                           train_actions,
                                           config.train_time_window,
                                           config.device)

    test_xs, test_us, test_ys = get_xuy(test_states,
                                        test_actions,
                                        test_states.shape[0] - 1,
                                        config.device)

    ds = TensorDataset(train_xs, train_us, train_ys)
    train_loader = DataLoader(ds,
                              batch_size=config.batch_size,
                              shuffle=True)

    iters = len(train_loader)
    n_update = 0
    min_test_loss = 100
    for iter in range(config.epoch):
        print("Epoch [{}] / [{}]".format(iter, config.epoch))
        for i, (x, u, y) in enumerate(train_loader):
            x = x + (torch.randn_like(x) * config.perturb_x_param).to(config.device)
            pred = m.rollout(x, u)
            pred_loss = criteria(pred, y)

            # Regularization
            loss = pred_loss

            # Update
            opt.zero_grad()
            loss.backward()
            opt.step()

            n_update += 1

            log_dict = dict()
            log_dict['train_loss'] = pred_loss

            if n_update % config.test_every == 0:
                m.eval()
                test_pred = m.rollout(test_xs, test_us)
                test_pred_loss = criteria(test_pred, test_ys)
                log_dict['test_loss'] = test_pred_loss
                m.train()

                if test_pred_loss <= min_test_loss:
                    print("BEST model found")
                    torch.save(m.state_dict(), join(wandb.run.dir, '{}_model.pt'.format(n_update)))
                    min_test_loss = test_pred_loss

            if use_lr_schedule:
                scheduler.step(iter + i / iters)
                log_dict['lr'] = opt.param_groups[0]['lr']

            wandb.log(log_dict)

    torch.save(m.state_dict(), join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    train(get_config())
