import dgl
import torch
import torch.nn as nn


class GraphSSM(nn.Module):

    def __init__(self, encoder, transition_model, decoder, is_residual=False, graph=None):
        super(GraphSSM, self).__init__()

        self.encoder = encoder
        self.transition_model = transition_model
        self.decoder = decoder
        self.is_residual = is_residual
        self._g = graph
        if self._g is not None:
            self._num_nodes = self.g.number_of_nodes()
        else:
            self._num_nodes = None

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, graph):
        self._g = graph
        self._num_nodes = graph.number_of_nodes()

    @property
    def num_nodes(self):
        return self._num_nodes

    def transit(self, g, h, u):
        """
        :param g: dgl.graph
        :param h: torch.tensor [#. nodes x hidden dim]
        :param u: torch.tensor [#. action nodes x action dim]
        :return: torch.tensor [#. nodes x hidden dim]
        """
        g = g.local_var()
        u_padded = torch.zeros(g.batch_size * self.num_nodes, 1).to(h.device)
        u_padded[g.ndata['is_control'].bool()] = u.squeeze().to(h.device)  # [#. nodes x hidden_dim]
        return self._transit(g, h, u_padded)

    def _transit(self, g, h, u):
        h = self.transtion_model(g, h, u)
        return h

    def rollout(self, history, us):
        """
        :param history: torch.tensor (dim: #.nodes x [history length x history feature dim])
        :param us: torch.tensor(dim: #.control nodes x rollout length x action dim)
        :return: predicted states (dim: #.nodes x rollout length x state dim)
        """
        h = self.encoder(history)
        # infer batch size
        bs, remainder = divmod(history.shape[0], self.num_nodes)
        msg = "'#.nodes' {} in history tensor is not multiple of 'num_nodes' {}".format(history.shape[0],
                                                                                        self.num_nodes)
        assert remainder == 0, msg
        g = dgl.batch([self.g for _ in range(bs)])  # batched graph (if required)
        hs = []
        for u in us.split(split_size=1, dim=1):
            h = self.transit(g, h, u)  # [#.nodes x hidden dim]
            hs.append(h)
        hs = torch.stack(hs, dim=1)  # [#.nodes x rollout length x hidden dim]
        xs = self.decoder(hs)  # [#. nodes x rollout length x state dim]
        if self.is_residual:  # residual from initial state x_t
            for i in range(xs.shape[1]):
                if i == 0:
                    xs[:, i, 0] = xs[:, i, 0] + history[:, -1]
                else:
                    xs[:, i, :] = xs[:, i-1, :] + xs[:, i, :]
        info = dict()
        with torch.no_grad():
            init_norm = hs[:, 0, :].reshape(self.num_nodes, bs, -1).norm(dim=(0, 2)).mean()
            info['init_norm'] = init_norm
            last_norm = hs[:, -1, :].reshape(self.num_nodes, bs, -1).norm(dim=(0, 2)).mean()
            info['last_norm'] = last_norm
        return xs, info

    def get_hidden(self, history, us=None):
        """
        :param history: torch.tensor (dim: #.nodes x [history length x history feature dim])
        :param us: torch.tensor(dim: #.control nodes x rollout length x action dim)
        :return: hidden states (dim: #.nodes x hidden_dim)
        """
        h = self.encoder(history)
        bs, remainder = divmod(history.shape[0], self.num_nodes)
        msg = "'#.nodes' {} in history tensor is not multiple of 'num_nodes' {}".format(history.shape[0],
                                                                                        self.num_nodes)
        assert remainder == 0, msg
        if us is not None:
            g = dgl.batch([self.g for _ in range(bs)])  # batched graph (if required)
            for u in us.split(split_size=1, dim=1):
                h = self.transit(g, h, u)  # [#.nodes x hidden dim]
        return h


class GraphSSM_GAT(GraphSSM):

    def _transit(self, g, h, u):
        hu = torch.cat([h, -u], dim=-1)
        pos = g.ndata['position']
        hs = self.transition_model(g, hu, pos)
        hs = hs.mean(dim=1)
        return hs
