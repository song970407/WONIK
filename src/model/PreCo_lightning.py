import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.PreCo.compte_gaussian_nll import compute_gaussian_nll


class PreCo(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 predictor: nn.Module,
                 corrector: nn.Module,
                 decoder: nn.Module,
                 obs_dim: int = 1):
        super(PreCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.predictor = predictor
        self.corrector = corrector
        self.decoder = decoder
        self.obs_dim = obs_dim

    def correct(self, g, h, x):
        return self.corrector(g, h, x)

    def predict(self, g, h, x):
        return self.predictor(g, h, x)

    def filter_history(self, g, xs, us, h0=None):
        """ Estimate hidden from the sequences of the observations and controls.
                Args:
                    g:
                    xs: [#.total state nodes x history_len x state_dim]
                    us: [#.total control nodes x (history_len-1) x control_dim]
                    h0: initial hidden [#.total state nodes x hidden_dim]

                Returns:
                """

        if h0 is None:
            h = torch.zeros(g.number_of_nodes('tc'), self.hidden_dim, device=self.device)
        else:
            h = h0

        xs = xs.unbind(dim=1)
        us = us.unbind(dim=1)

        for i in range(len(us)):
            h = self.correct(g, h, xs[i])  # correct hidden
            h = self.predict(g, h, us[i])  # predict next hidden
        h = self.correct(g, h, xs[-1])
        return h

    def multi_step_prediction(self, g, h, us):
        """
        Args:
            g:
            h: initial hidden [#.total state nodes x hidden_dim]
            us: [#.total control nodes x prediction_length x control_dim]
        Returns:

        """

        hs = []
        for u in us.unbind(dim=1):
            h = self.predict(g, h, u)
            hs.append(h)
        hs = torch.stack(hs, dim=1)  # [#. total state nodes x prediction_length x  hidden_dim]
        xs = self.decoder(hs)  # [#. total state nodes x prediction_length x  hidden_dim]
        return xs

    def rollout(self, g, hc, xs, us):
        """
        Args:
            g:
            hc: initial hidden
            us: u_t ~ u_t+(k-1) # [#. total control nodes x rollout length x control dim]
            xs: x_t+1 ~ x_t+k

        Returns:
        """

        K = us.shape[1]
        xs = xs.unbind(dim=1)
        us = us.unbind(dim=1)

        hps = []  # predicted hiddens
        hcs = []  # corrected hiddens

        # performs one-step prediction recursively.
        for k in range(K):
            hp = self.predict(g, hc, us[k])
            hc = self.correct(g, hp, xs[k])
            hps.append(hp)
            hcs.append(hc)

        # one-step prediction results
        # hcs = [hc_t+1, hc_t+2, ..., hc_t+k]
        hcs = torch.stack(hcs, dim=1)  # [#. total state nodes x rollout length x hidden_dim]

        # hps = [hp_t+1, hp_t+2, ..., hp_t+k]
        hps = torch.stack(hps, dim=1)  # [#. total state nodes x rollout length x hidden_dim]

        # performs latent overshooting
        latent_overshoot_hps = torch.zeros(g.number_of_nodes('tc'), K, K, self.hidden_dim, device=self.device)
        latent_overshoot_mask = torch.zeros(g.number_of_nodes('tc'), K, K, self.obs_dim, device=self.device)
        for i, hp in enumerate(hps.unbind(dim=1)[:-1]):
            latent_hps = []
            for j in range(i + 1, K):
                hp = self.predict(g, hp, us[j])
                latent_hps.append(hp)
            latent_hps = torch.stack(latent_hps, dim=1)
            latent_overshoot_hps[:, i, i + 1:, :] = latent_hps
            latent_overshoot_mask[:, i, i + 1:, :] = 1.0

        # decoding the one-step prediction results
        hcs_dec = self.decoder(hcs)  # [x_t+1, ..., x_t+k]
        hps_dec = self.decoder(hps)  # [x_t+1, ..., x_t+k]

        # latent the latent overshooting results
        latent_overshoot_dec = self.decoder(latent_overshoot_hps)

        ret = dict()
        ret['hcs_dec'] = hcs_dec
        ret['hps_dec'] = hps_dec
        ret['latent_overshoot_dec'] = latent_overshoot_dec
        ret['latent_overshoot_mask'] = latent_overshoot_mask
        return ret

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
        ret = dict()
        ret['optimizer'] = optimizer
        ret['lr_scheduler'] = lr_scheduler
        return ret

    def training_step(self, batch, batch_idx):
        g = batch
        g = g.to(self.device)
        h0 = self.filter_history(g,
                                 g.nodes['tc'].data['history'],
                                 g.nodes['control'].data['history'])
        ret = self.rollout(g,
                           h0,
                           g.nodes['tc'].data['future'],
                           g.nodes['control'].data['future'])

        # single-step predictions
        hcs_dec = ret['hcs_dec']  # predicted [x_t+1, ..., x_t+k] from [hc_t+1, ..., hc_t+k]
        hps_dec = ret['hps_dec']  # predicted [x_t+1, ..., x_t+k] from [hp_t+1, ..., hp_t+k]

        # multi-step predictions
        latent_overshoot_dec = ret['latent_overshoot_dec']  # [#.total state nodes x future len x future len x obs_dim]
        # if the index of the second dimension (1th dim in python) is smaller than the onf the third dimension -> requires to igonre.
        latent_overshoot_mask = ret['latent_overshoot_mask']

        # prepare target
        target = g.nodes['tc'].data['future']
        future_len = latent_overshoot_dec.shape[1]
        repeated_target = target.unsqueeze(dim=1).repeat(1, future_len, 1, 1)

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
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('corrector_nll_loss', hcs_dec_loss, on_step=True)
        self.log('predictor_nll_loss', hps_dec_loss, on_step=True)
        self.log('overshoot_nll_loss', latent_overshoot_dec_loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        g = batch
        g = g.to(self.device)
        h0 = self.filter_history(g,
                                 g.nodes['tc'].data['history'],
                                 g.nodes['control'].data['history'])

        multi_step_pred = self.multi_step_prediction(g,
                                                     h0,
                                                     g.nodes['control'].data['future'])

        multi_step_pred_loss = F.mse_loss(multi_step_pred[:, :, 0].unsqueeze(dim=-1),
                                          g.nodes['tc'].data['future'])

        self.log('multi_step_loss', multi_step_pred_loss, sync_dist=True)
