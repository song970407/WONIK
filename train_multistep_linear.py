from os.path import join

import torch
import wandb

from torch.utils.data import DataLoader, TensorDataset
from src.utils.load_data import load_data
from src.model.get_model import get_multi_linear_residual_model
from src.utils.data_preprocess import get_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Hyperparameters

def main(state_order, action_order):
    state_dim = 140
    action_dim = 40
    # state_order = 20
    # action_order = 20
    BS = 128
    H = 100
    scale_min = 20.0
    scale_max = 420.0
    scaler = (scale_min, scale_max)
    # Prepare Model and Dataset

    m = get_multi_linear_residual_model(state_dim, action_dim, state_order, action_order).to(DEVICE)

    train_data_path = ['docs/new_data/expert/data_3.csv', 'docs/new_data/icgrnn/data_3.csv',
                       'docs/new_data/overshoot/data_2.csv']
    test_data_path = ['docs/new_data/icgrnn/data_4.csv', 'docs/new_data/overshoot/data_1.csv']
    # 'docs/new_data/icgrnn/data_2.csv', 'docs/new_data/linear/data_2.csv']
    glass_tc_pos_path = 'docs/new_location/glass_TC.csv'
    control_tc_pos_path = 'docs/new_location/control_TC.csv'

    train_states, train_actions, info = load_data(paths=train_data_path,
                                                  scaling=True,
                                                  scaler=scaler,
                                                  preprocess=True,
                                                  history_x=state_order,
                                                  history_u=action_order)

    # Set te minimum and maximum temperature as 20 and 420

    test_states, test_actions, _ = load_data(paths=test_data_path,
                                             scaling=True,
                                             scaler=scaler,
                                             preprocess=True,
                                             history_x=state_order,
                                             history_u=action_order)

    train_history_xs, train_history_us, train_us, train_ys, pos_tc = get_data(states=train_states,
                                                                              actions=train_actions,
                                                                              rollout_window=H,
                                                                              history_x_window=state_order,
                                                                              history_u_window=action_order,
                                                                              glass_tc_pos_path=glass_tc_pos_path,
                                                                              control_tc_pos_path=control_tc_pos_path,
                                                                              num_glass_tc=state_dim,
                                                                              num_control_tc=action_dim,
                                                                              device=DEVICE)
    train_history_xs = train_history_xs[0].transpose(1, 2)
    train_history_us = train_history_us.transpose(1, 2)
    train_us = train_us.transpose(1, 2)
    train_ys = train_ys[0].transpose(1, 2)

    test_history_xs, test_history_us, test_us, test_ys, _ = get_data(states=test_states,
                                                                     actions=test_actions,
                                                                     rollout_window=H,
                                                                     history_x_window=state_order,
                                                                     history_u_window=action_order,
                                                                     glass_tc_pos_path=glass_tc_pos_path,
                                                                     control_tc_pos_path=control_tc_pos_path,
                                                                     num_glass_tc=state_dim,
                                                                     num_control_tc=action_dim,
                                                                     device=DEVICE)
    test_history_xs = test_history_xs[0].transpose(1, 2)
    test_history_us = test_history_us.transpose(1, 2)
    test_us = test_us.transpose(1, 2)
    test_ys = test_ys[0].transpose(1, 2)

    # Training Route Setting
    criteria = torch.nn.MSELoss()
    train_ds = TensorDataset(train_history_xs, train_history_us, train_us, train_ys)
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    test_criteria = torch.nn.L1Loss()

    opt = torch.optim.Adam(m.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    iters = len(train_loader)
    EPOCHS = 100
    TEST_EVERY = 100
    run = wandb.init(entity='sentinel',
                     name='State Order ' + str(state_order) + ' Action Order ' + str(action_order) + ' H ' + str(H),
                     reinit=True,
                     project='Multistep_Linear_Res')
    run.config['state_order'] = state_order
    run.config['action_order'] = action_order
    run.config['scaler'] = (info['scale_min'], info['scale_max'])
    num_updates = 0

    # Start Training
    for ep in range(EPOCHS):
        print("Epoch [{}] / [{}]".format(ep, EPOCHS))
        for i, (x0, u0, u, y) in enumerate(train_loader):
            y_predicted = m.rollout(x0, u0, u)
            loss = criteria(y_predicted, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(ep + i / iters)
            num_updates += 1
            log_dict = {}
            log_dict['train_loss'] = loss.item()
            log_dict['lr'] = opt.param_groups[0]['lr']
            if num_updates % TEST_EVERY == 0:
                with torch.no_grad():
                    test_predicted_y = m.rollout(test_history_xs, test_history_us, test_us)
                    test_loss = test_criteria(test_predicted_y, test_ys)
                    log_dict['test_loss'] = test_loss.item()
            torch.save(m.state_dict(),
                       join(wandb.run.dir, 'model_' + str(state_order) + '_' + str(action_order) + '.pt'))
            wandb.log(log_dict)

    run.finish()


if __name__ == '__main__':
    state_orders = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    action_orders = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in range(len(state_orders)):
        for j in range(len(action_orders)):
            main(state_orders[i], action_orders[j])
