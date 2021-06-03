from os.path import join
from timeit import default_timer

import torch
import wandb
from box import Box
from torch.utils.data import TensorDataset

from src.PreCo.Correctors import Corrector
from src.PreCo.Decoders import get_convex_decoder, get_decoder, get_linear_decoder
from src.PreCo.PreCo import GraphPreCo
from src.PreCo.Predictors import Predictor, ConvexPredictor, LinearPredictor
from src.utils.PreCo.GraphDataLoader import GraphDataLoader
from src.utils.PreCo.compte_gaussian_nll import compute_gaussian_nll
from src.utils.PreCo.get_graph import get_hetero_graph
from src.utils.PreCo.load_data import load_data


def get_config():
    conf = Box({
        'data': {
            'train_path': ['docs/new_data/expert/data_1.csv',
                           'docs/new_data/expert/data_2.csv',
                           'docs/new_data/expert/data_3.csv',
                           'docs/new_data/linear/data_1.csv',
                           'docs/new_data/icgrnn/data_1.csv'],
            'test_path': ['docs/new_data/expert/data_4.csv'],
            'g_tc_pos': 'docs/new_location/glass_TC.csv',
            'c_tc_pos': 'docs/new_location/control_TC.csv',
            'scaling': True,
            't2t_threshold': 850.0,
            'action_ws': True,
        },
        'train': {
            'batch_size': 12,
            'history_len': 20,
            'future_len': 20,
            'log_every': 25,
            'epoch': 20,
            'opt_config': {
                'name': 'Adam',
                'kwargs': {
                    'lr': 1e-4
                },
            },
            'scheduler_config': {
                'use_scheduler': False,
                'name': 'CosineAnnealingWarmRestarts',
                'kwargs': {
                    'T_0': 50
                },
            },
        },
        'model': {
            'linear': True,
            'convex': True,
            'deterministic': True,
            'preco_hidden_dim': 32,
            'predictor': {
                'mlp_h_dim': 32
            },
            'corrector': {
                'mlp_h_dim': 32
            },
            'decoder': {
                'out_dim': 16
            },
        },
        'wandb': {}
    })

    action_str = 'ws' if conf.data.action_ws else 'power'
    output_type = 'deterministic' if conf.model.deterministic else 'stochastic'
    convex = 'cvx' if conf.model.convex else 'n-cvx'
    linear = 'linear' if conf.model.linear else 'n-linear'
    conf.wandb.tags = ["{}".format(action_str),
                       "{}".format(output_type),
                       "{}".format(convex),
                       "{}".format(linear)]
    return conf


def get_model(config):
    model_config = config.model
    is_cvx = model_config.convex if config.model.get('convex') else False
    is_deterministic = model_config.deterministic if config.model.get('deterministic') else False
    is_linear = model_config.linear if config.model.get('linear') else False

    corrector = Corrector(x_dim=1,
                          h_dim=model_config.preco_hidden_dim,  # PreCo hidden dim
                          mlp_h_dim=model_config.corrector.mlp_h_dim)

    predictor_kwargs = {'u_dim': 1,
                        'h_dim': model_config.preco_hidden_dim,
                        'mlp_h_dim': model_config.predictor.mlp_h_dim}

    decoder_kwargs = {'is_deterministic': is_deterministic,
                      'preco_hidden_dim': model_config.preco_hidden_dim}
    if is_cvx:
        predictor = ConvexPredictor(**predictor_kwargs)
        decoder = get_convex_decoder(**decoder_kwargs)
    else:
        predictor = Predictor(**predictor_kwargs)
        decoder = get_decoder(**decoder_kwargs)

    if is_linear:
        predictor = LinearPredictor(**predictor_kwargs)
        decoder = get_linear_decoder(**decoder_kwargs)

    obs_dim = 1 if model_config.deterministic else 2
    m = GraphPreCo(hidden_dim=model_config.preco_hidden_dim,
                   predictor=predictor,
                   corrector=corrector,
                   decoder=decoder,
                   obs_dim=obs_dim)
    return m


def main(config):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # prepare training data
    g = get_hetero_graph(glass_tc_pos_path=config.data.g_tc_pos,
                         control_tc_pos_path=config.data.c_tc_pos,
                         t2t_threshold=config.data.t2t_threshold)

    hx, fx, hu, fu, train_info = load_data(paths=config.data.train_path,
                                           history_len=config.train.history_len,
                                           future_len=config.train.future_len,
                                           action_ws=config.data.action_ws,
                                           scaling=config.data.scaling)
    ds = TensorDataset(hx, fx, hu, fu)
    train_dl = GraphDataLoader(g=g, dataset=ds,
                               batch_size=config.train.batch_size,
                               device=device, shuffle=True,
                               num_workers=16, pin_memory=True)

    hx, fx, hu, fu, test_info = load_data(paths=config.data.test_path,
                                          history_len=config.train.history_len,
                                          future_len=config.train.future_len,
                                          action_ws=config.data.action_ws,
                                          scaling=config.data.scaling,
                                          state_scaler=train_info['state_scaler'],
                                          action_scaler=train_info['action_scaler'])
    ds = TensorDataset(hx, fx, hu, fu)
    test_dl = GraphDataLoader(g=g, dataset=ds,
                              batch_size=512, device=device, shuffle=False,
                              num_workers=16, pin_memory=True)

    m = get_model(config).to(device)

    # optimizer and lr scheduler
    opt_config = config.train.opt_config
    opt = getattr(torch.optim, opt_config.name)(m.parameters(), **opt_config.kwargs)
    scheduler_config = config.train.scheduler_config
    if scheduler_config.use_scheduler:
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.name)(opt, **scheduler_config.kwargs)

    deterministic = config.model.deterministic
    deterministic_criteria = torch.nn.L1Loss(reduction='none')

    run = wandb.init(entity='sentinel',
                     config=config.to_dict(),
                     reinit=True,
                     tags=config.wandb.tags,
                     project='PreCo')
    config.to_yaml(join(wandb.run.dir, "model_config.yaml"))

    iters = len(train_dl)
    grad_step = 0
    for ep in range(config.train.epoch):
        for i, g in enumerate(train_dl):
            g = g.to(torch.device(device))
            start_time = default_timer()
            h0 = m.filter_history(g,
                                  g.nodes['tc'].data['history'],
                                  g.nodes['control'].data['history'])
            ret = m.rollout(g,
                            h0,
                            g.nodes['tc'].data['future'],
                            g.nodes['control'].data['future'])

            # single-step predictions
            hcs_dec = ret['hcs_dec']  # predicted [x_t+1, ..., x_t+k] from [hc_t+1, ..., hc_t+k]
            hps_dec = ret['hps_dec']  # predicted [x_t+1, ..., x_t+k] from [hp_t+1, ..., hp_t+k]

            # multi-step predictions
            latent_overshoot_dec = ret[
                'latent_overshoot_dec']  # [#.total state nodes x future len x future len x obs_dim]
            # if the index of the second dimension (1th dim in python) is smaller than the onf the third dimension -> requires to igonre.
            latent_overshoot_mask = ret['latent_overshoot_mask']

            target = g.nodes['tc'].data['future']
            repeated_target = target.unsqueeze(dim=1).repeat(1, config.train.future_len, 1, 1)

            if deterministic:
                hcs_dec_loss = deterministic_criteria(hcs_dec, target).mean()
                hps_dec_loss = deterministic_criteria(hps_dec, target).mean()
                latent_overshoot_dec_loss = deterministic_criteria(latent_overshoot_dec, repeated_target)
                latent_overshoot_dec_loss = latent_overshoot_dec_loss * latent_overshoot_mask
                latent_overshoot_dec_loss = latent_overshoot_dec_loss.sum() / latent_overshoot_mask.sum()
            else:
                hcs_dec_loss = compute_gaussian_nll(mu=hcs_dec[:, :, 0],
                                                    log_var=hcs_dec[:, :, 1],
                                                    target=target.squeeze()).mean()

                hps_dec_loss = compute_gaussian_nll(mu=hps_dec[:, :, 0],
                                                    log_var=hps_dec[:, :, 1],
                                                    target=target.squeeze()).mean()

                latent_overshoot_dec_loss = compute_gaussian_nll(mu=latent_overshoot_dec[:, :, :, 0],
                                                                 log_var=latent_overshoot_dec[:, :, :, 1],
                                                                 target=repeated_target.squeeze())

                latent_overshoot_dec_loss = latent_overshoot_dec_loss * latent_overshoot_mask[:, :, :, 0]
                latent_overshoot_dec_loss = latent_overshoot_dec_loss.sum() / latent_overshoot_mask.sum()

            loss = hcs_dec_loss + hps_dec_loss + latent_overshoot_dec_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            if scheduler_config.use_scheduler:
                scheduler.step(ep + i / iters)
            end_time = default_timer()
            grad_step += 1

            log_dict = dict()
            log_dict['lr'] = opt.param_groups[0]['lr']
            log_dict['fit_time'] = end_time - start_time
            if deterministic:
                log_dict['corrector_l1_loss'] = hcs_dec_loss.item()
                log_dict['predictor_l1_loss'] = hps_dec_loss.item()
                log_dict['overshoot_l1_loss'] = latent_overshoot_dec_loss.item()
            else:
                log_dict['corrector_nll_loss'] = hcs_dec_loss.item()
                log_dict['predictor_nll_loss'] = hps_dec_loss.item()
                log_dict['overshoot_nll_loss'] = latent_overshoot_dec_loss.item()

            if grad_step % config.train.log_every == 0:
                with torch.no_grad():
                    start_time = default_timer()
                    m.eval()
                    multi_step_loss = 0.0
                    num_pred = 0
                    for test_g in test_dl:
                        test_g = test_g.to(torch.device(device))
                        h0 = m.filter_history(test_g,
                                              test_g.nodes['tc'].data['history'],
                                              test_g.nodes['control'].data['history'])
                        multi_step_pred = m.multi_step_prediction(test_g,
                                                                  h0,
                                                                  test_g.nodes['control'].data['future'])
                        if deterministic:
                            multi_step_pred_loss = deterministic_criteria(multi_step_pred,
                                                                          test_g.nodes['tc'].data['future'])
                        else:
                            # check only mean trend
                            multi_step_pred_loss = deterministic_criteria(multi_step_pred[:, :, 0].unsqueeze(dim=-1),
                                                                          test_g.nodes['tc'].data['future'])

                        multi_step_loss += multi_step_pred_loss.sum().item()
                        num_pred += multi_step_pred_loss.numel()
                    multi_step_loss = multi_step_loss / num_pred
                    m.train()
                    end_time = default_timer()
                    log_dict['test_time'] = end_time - start_time
                    log_dict['multi_step_loss'] = multi_step_loss
                    torch.save(m.state_dict(), join(wandb.run.dir, 'model.pt'))
            wandb.log(log_dict)
    run.finish()


if __name__ == '__main__':
    config = get_config()
    main(config)
