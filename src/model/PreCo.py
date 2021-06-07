import torch
import torch.nn as nn

class PreCo(nn.Module):
    def __init__(self,
                 predictor: nn.Module,
                 corrector: nn.Module,
                 decoder: nn.Module,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int):
        super(PreCo, self).__init__()
        self.predictor = predictor
        self.corrector = corrector
        self.decoder = decoder
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def correct(self, h, x):
        return self.corrector(torch.cat([h, x], dim=-1))

    def predict(self, h, u):
        return self.predictor(torch.cat([h, u], dim=-1))

    def decode(self, h):
        return self.decoder(h)

    def filter_history(self, xs, us, h0=None):
        """
        Estimate hidden from the sequences of the observations and controls.
        Args:
            xs: [B x history_len x state_dim]
            us: [B x (history_len-1) x action_dim]
            h0: initial hidden [B x hidden_dim]
        Returns:
        """
        if h0 is None:
            h = torch.zeros(xs.shape[0], self.hidden_dim).to(xs.device)
        else:
            h = h0

        xs = xs.unbind(dim=1)
        us = us.unbind(dim=1)

        for i in range(len(us)):
            h = self.correct(h, xs[i])  # correct hidden
            h = self.predict(h, us[i])  # predict next hidden
        h = self.correct(h, xs[-1])
        return h

    def multi_step_prediction(self, h, us):
        """
        Args:
            h: [hidden_dim]
            us: [H x action_dim]
        Returns:
        """
        hs = []
        for u in us.unbind(dim=1):
            h = self.predict(h, u)
            hs.append(h)
        hs = torch.stack(hs, dim=1)  # [H x  hidden_dim]
        xs = self.decode(hs)  # [H x  state_dim]
        return xs

    def rollout(self, hc, xs, us):
        """
        Args:
            hc: [B x hidden_dim]
            xs: [B x H x state_dim]
            us: [B x H x action_dim]
        Returns:
        """
        B = us.shape[0]
        H = us.shape[1]
        xs = xs.unbind(dim=1)
        us = us.unbind(dim=1)

        hps = []  # predicted hiddens
        hcs = []  # corrected hiddens

        # performs one-step prediction recursively.
        for h in range(H):
            hp = self.predict(hc, us[h])  # B x hidden_dim
            hc = self.correct(hp, xs[h])  # B x hidden_dim
            hps.append(hp)
            hcs.append(hc)

        # one-step prediction results
        # hcs = [hc_t+1, hc_t+2, ..., hc_t+k]
        hcs = torch.stack(hcs, dim=1)  # [B x H x hidden_dim]

        # hps = [hp_t+1, hp_t+2, ..., hp_t+k]
        hps = torch.stack(hps, dim=1)  # [B x H x hidden_dim]

        # performs latent overshooting
        latent_overshoot_hps = torch.zeros(B, H, H, self.hidden_dim).to(us[0].device)
        latent_overshoot_mask = torch.zeros(B, H, H, self.state_dim).to(us[0].device)
        for i, hp in enumerate(hps.unbind(dim=1)[:-1]):
            latent_hps = []
            for j in range(i + 1, H):
                hp = self.predict(hp, us[j])  # [B x hidden_dim]
                latent_hps.append(hp)
            latent_hps = torch.stack(latent_hps, dim=1)
            latent_overshoot_hps[:, i, i + 1:, :] = latent_hps
            latent_overshoot_mask[:, i, i + 1:, :] = 1.0

        # decoding the one-step prediction results
        hcs_dec = self.decode(hcs)  # [x_t+1, ..., x_t+k], [B x H x state_dim]
        hps_dec = self.decode(hps)  # [x_t+1, ..., x_t+k]

        # latent the latent overshooting results
        latent_overshoot_dec = self.decode(latent_overshoot_hps)

        ret = dict()
        ret['hcs_dec'] = hcs_dec
        ret['hps_dec'] = hps_dec
        ret['latent_overshoot_dec'] = latent_overshoot_dec
        ret['latent_overshoot_mask'] = latent_overshoot_mask
        return ret
