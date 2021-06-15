from os.path import join
from timeit import default_timer

import torch
import wandb

from torch.utils.data import DataLoader, TensorDataset
from src.utils.load_data import load_preco_data
from src.model.get_model import get_preco_model
from src.utils.data_preprocess import get_data, get_preco_data

MANUAL_SEED = 0
torch.random.manual_seed(MANUAL_SEED)
torch.cuda.random.manual_seed(MANUAL_SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_preco(train_dict):
    model_name = train_dict['model_name']
    state_dim = train_dict['state_dim']
    action_dim = train_dict['action_dim']
    hidden_dim = train_dict['hidden_dim']
    state_scaler = train_dict['state_scaler']
    action_scaler = train_dict['action_scaler']
    receding_history = train_dict['receding_history']
    receding_horizon = train_dict['receding_horizon']
    EPOCHS = train_dict['EPOCHS']
    BS = train_dict['BS']
    lr = train_dict['lr']
    train_data_path = train_dict['train_data_path']
    test_data_path = train_dict['test_data_path']

    load_saved = False
    m = get_preco_model(model_name, load_saved, state_dim, action_dim, hidden_dim).to(DEVICE)

    train_states, train_actions, _ = load_preco_data(paths=train_data_path,
                                                     scaling=True,
                                                     state_scaler=state_scaler,
                                                     action_scaler=action_scaler,
                                                     preprocess=True,
                                                     receding_history=receding_history)
    test_states, test_actions, _ = load_preco_data(paths=test_data_path,
                                                   scaling=True,
                                                   state_scaler=state_scaler,
                                                   action_scaler=action_scaler,
                                                   preprocess=True,
                                                   receding_history=receding_history)
    train_history_xs, train_history_us, train_us, train_ys = get_preco_data(states=train_states,
                                                                            actions=train_actions,
                                                                            receding_history=receding_history,
                                                                            receding_horizon=receding_horizon,
                                                                            state_dim=state_dim,
                                                                            action_dim=action_dim,
                                                                            device=DEVICE)
    train_history_xs = train_history_xs[0].transpose(1, 2)
    train_history_us = train_history_us.transpose(1, 2)
    train_us = train_us.transpose(1, 2)
    train_ys = train_ys[0].transpose(1, 2)
    test_history_xs, test_history_us, test_us, test_ys = get_preco_data(states=test_states,
                                                                        actions=test_actions,
                                                                        receding_history=receding_history,
                                                                        receding_horizon=receding_horizon,
                                                                        state_dim=state_dim,
                                                                        action_dim=action_dim,
                                                                        device=DEVICE)
    test_history_xs = test_history_xs[0].transpose(1, 2)
    test_history_us = test_history_us.transpose(1, 2)
    test_us = test_us.transpose(1, 2)
    test_ys = test_ys[0].transpose(1, 2)

    # Training Route Setting
    train_dataset = TensorDataset(train_history_xs, train_history_us, train_us, train_ys)
    criteria = torch.nn.SmoothL1Loss(reduction='none')
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    test_every = 25
    save_every = 100
    run = wandb.init(entity='sentinel',
                     config=train_dict,
                     name='{}, History: {}, Horizon: {}'.format(model_name, receding_history, receding_horizon),
                     reinit=True,
                     project='WONIK_PreCo')
    num_updates = 0
    best_test_loss = None
    # Start Training
    for ep in range(EPOCHS):
        print('{} th epoch'.format(ep))
        train_dl = DataLoader(train_dataset, batch_size=BS, shuffle=True)
        iters = len(train_dl)
        for i, (history_x, history_u, u, y) in enumerate(train_dl):
            start_time = default_timer()
            h0 = m.filter_history(history_x, history_u)
            ret = m.rollout(h0, y, u)
            hcs_dec = ret['hcs_dec']
            hps_dec = ret['hps_dec']
            latent_overshoot_dec = ret['latent_overshoot_dec']
            latent_overshoot_mask = ret['latent_overshoot_mask']
            repeated_y = y.unsqueeze(dim=1).repeat(1, receding_horizon, 1, 1)

            hcs_dec_loss = criteria(hcs_dec, y).mean()
            hps_dec_loss = criteria(hps_dec, y).mean()
            latent_overshoot_dec_loss = criteria(latent_overshoot_dec, repeated_y)
            latent_overshoot_dec_loss = latent_overshoot_dec_loss * latent_overshoot_mask
            latent_overshoot_dec_loss = latent_overshoot_dec_loss.sum() / latent_overshoot_mask.sum()
            loss = hcs_dec_loss + hps_dec_loss + latent_overshoot_dec_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(ep + i / iters)
            end_time = default_timer()

            log_dict = dict()
            log_dict['lr'] = opt.param_groups[0]['lr']
            log_dict['fit_time'] = end_time - start_time
            log_dict['train_corrector_l1_loss'] = hcs_dec_loss.item()
            log_dict['train_predictor_l1_loss'] = hps_dec_loss.item()
            log_dict['train_overshoot_l1_loss'] = latent_overshoot_dec_loss.item()
            num_updates += 1
            if num_updates % test_every == 0:
                with torch.no_grad():
                    history_x = test_history_xs
                    history_u = test_history_us
                    u = test_us
                    y = test_ys
                    h0 = m.filter_history(history_x, history_u)
                    ret = m.rollout(h0, y, u)
                    hcs_dec = ret['hcs_dec']
                    hps_dec = ret['hps_dec']
                    latent_overshoot_dec = ret['latent_overshoot_dec']
                    latent_overshoot_mask = ret['latent_overshoot_mask']
                    repeated_y = y.unsqueeze(dim=1).repeat(1, receding_horizon, 1, 1)

                    hcs_dec_loss = criteria(hcs_dec, y).mean()
                    hps_dec_loss = criteria(hps_dec, y).mean()
                    latent_overshoot_dec_loss = criteria(latent_overshoot_dec, repeated_y)
                    latent_overshoot_dec_loss = latent_overshoot_dec_loss * latent_overshoot_mask
                    latent_overshoot_dec_loss = latent_overshoot_dec_loss.sum() / latent_overshoot_mask.sum()
                    log_dict['test_corrector_l1_loss'] = hcs_dec_loss.item()
                    log_dict['test_predictor_l1_loss'] = hps_dec_loss.item()
                    log_dict['test_overshoot_l1_loss'] = latent_overshoot_dec_loss.item()
                    test_loss = hcs_dec_loss.item() + hps_dec_loss.item() + latent_overshoot_dec_loss.item()
                    if best_test_loss == None:
                        best_test_loss = test_loss
                        torch.save(m.state_dict(), 'model/{}.pt')
                    elif test_loss < best_test_loss:
                        best_test_loss = test_loss
                        torch.save(m.state_dict(), 'model/{}.pt')
            if num_updates % save_every == 0:
                torch.save(m.state_dict(), join(wandb.run.dir, 'model.pt'))
            wandb.log(log_dict)
    run.finish()


if __name__ == '__main__':
    train_dict = {
        'model_name': 'PreCo1',
        'state_dim': 140,
        'action_dim': 40,
        'hidden_dim': 256,
        'state_scaler': (20.0, 420.0),
        'action_scaler': (20.0, 420.0),
        'receding_history': 50,
        'receding_horizon': 20,
        'EPOCHS': 1000,
        'BS': 64,
        'lr': 1e-4,
        'train_data_path': ['docs/new_data/expert/data_3.csv', 'docs/new_data/icgrnn/data_3.csv', 'docs/new_data/overshoot/data_2.csv'],
        # 'test_data_path': ['docs/new_data/expert/data_2.csv', 'docs/new_data/icgrnn/data_4.csv', 'docs/new_data/overshoot/data_1.csv']
        'test_data_path': ['docs/new_data/expert/data_2.csv']
    }
    train_preco(train_dict)
