import dgl
import torch
import torch.nn as nn

from src.graph_config import u2g, u2c, g2g, g2c, c2g, c2c


class HeteroGraphSSM(nn.Module):

    def __init__(self,
                 alpha_enc: nn.Module,
                 beta_enc: nn.Module,
                 hist_enc: nn.Module,
                 control_to_hidden: nn.Module,
                 hidden_to_tc: nn.Module):
        super(HeteroGraphSSM, self).__init__()

        self.alpha_enc = alpha_enc  # f(p_i, p_j) -> a_ij
        self.beta_enc = beta_enc  # f(p_i, p_j) -> b_ij
        self.hist_enc = hist_enc  # f(xs) -> h0
        self.c2h = control_to_hidden  # f(u) -> q
        self.h2x = hidden_to_tc  # f(h) -> x

    def get_history_hidden(self, history: torch.Tensor):
        """ Compute hidden embedding from history
        assuming "hidden embedding" can be inferred from the sequences of each observation.
        Args:
            history: (torch.Tensor) History of observations. expected shape is [obs. dim x history len]
        Returns:
            hidden embedding (torch.Tensor)
        """
        hidden = self.hist_enc(history)
        return hidden

    @staticmethod
    def set_hidden(g, hs):
        g.nodes['glass-tc'].data['hidden'] = hs[0]
        g.nodes['control-tc'].data['hidden'] = hs[1]

    def set_alpha(self, g):
        def _compute_alpha(edges):
            src_pos = edges.src['position']
            dst_pos = edges.dst['position']
            inp = torch.cat([src_pos, dst_pos], dim=-1)
            alpha = self.alpha_enc(inp)
            return {'alpha': alpha}

        g.apply_edges(_compute_alpha, etype=g2g)
        g.apply_edges(_compute_alpha, etype=g2c)
        g.apply_edges(_compute_alpha, etype=c2g)
        g.apply_edges(_compute_alpha, etype=c2c)

    def set_beta(self, g):
        def _compute_beta(edges):
            src_pos = edges.src['position']
            dst_pos = edges.dst['position']
            inp = torch.cat([src_pos, dst_pos], dim=-1)
            beta = self.beta_enc(inp)
            return {'beta': beta}

        g.apply_edges(_compute_beta, etype=u2g)
        g.apply_edges(_compute_beta, etype=u2c)

    def rollout(self, g, history_x, history_u, us, xs=None):
        """
        Args:
            g: (batched) dgl.graph

            history_x: (list[torch.Tensor, torch.Tensor])
                Input for initializing hidden embeddings
                - The 1st tensor is for (batched) glass-tc history.
                - The 2nd tensor is for (batched) control-tc history.
            history_u: (torch.Tensor)
                Input for initial exogenous embedding
                [(#.batch x #.control node) x u_history_length-1]
            us: (torch.Tensor) expected to get 'torch.tensor' with dimension
                [(#.batch x #.control node) x unroll length]
                the data for the first dimension is prepared as:
                [controller 1, ..., controller n, controller 1, ... , controller n, ... ]
            xs: (optional: torch.Tensor)

        Returns:
            predicted xs, hidden embeddings
        """

        if xs is not None:  # Teacher forcing
            pass
        else:
            pass

        self.set_alpha(g)
        self.set_beta(g)

        num_total_glass_tc = history_x[0].shape[0]  # num. of (batched) glass tc
        num_total_control_tc = history_x[1].shape[0]  # num. of (batched) glass tc
        h = self.get_history_hidden(torch.cat(history_x, dim=0))  # h0
        h = h.split(split_size=(num_total_glass_tc, num_total_control_tc), dim=0)  # h0-glass-tc, h0-control-tc
        hs_glass_tc, hs_control_tc = [], []
        u = torch.cat([history_u, us], dim=1)  # [(#.batch x # glass tc) x (u_history_length + unrooll_length - 1)]
        for idx in range(us.shape[1]):
            h = self.transit(g, h, u[:, idx:idx + history_u.shape[
                1] + 1])  # [(#.batch x # glass tc) x hidden dim], [(#.batch x # control tc) x hidden dim]
            hs_glass_tc.append(h[0])
            hs_control_tc.append(h[1])

        hs_glass_tc = torch.stack(hs_glass_tc, dim=1)  # [(#.batch x # glass tc) x time x  hidden dim]
        hs_control_tc = torch.stack(hs_control_tc, dim=1)  # [(#.batch x # control tc) x time x  hidden dim]
        xs = self.h2x(torch.cat([hs_glass_tc, hs_control_tc], dim=0)).squeeze(dim=-1)
        xs_glass_tc, xs_control_tc = xs.split(split_size=(num_total_glass_tc, num_total_control_tc), dim=0)

        return (xs_glass_tc, xs_control_tc), (hs_glass_tc, hs_control_tc)

    def transit(self, g, hs, u):
        """
        Args:
            g:
            u:
            alpha:
        Returns:

        """
        with g.local_scope():
            g.nodes['control'].data['u'] = u

            # compute "control" -> "hidden" effect
            g.multi_update_all(
                {u2g: (self._control_to_tc_msg, dgl.function.sum('msg', 'q')),
                 u2c: (self._control_to_tc_msg, dgl.function.sum('msg', 'q'))},
                'sum')

            # compute "tc" -> "tc" effect
            self.set_hidden(g, hs)
            g.multi_update_all(
                {g2g: (self._tc_to_tc_msg, dgl.function.sum('msg', 'sum_h')),
                 g2c: (self._tc_to_tc_msg, dgl.function.sum('msg', 'sum_h')),
                 c2g: (self._tc_to_tc_msg, dgl.function.sum('msg', 'sum_h')),
                 c2c: (self._tc_to_tc_msg, dgl.function.sum('msg', 'sum_h'))},
                'sum')

            # compute "next hidden"
            h_glass_tc_next = g.nodes['glass-tc'].data['sum_h'] + g.nodes['glass-tc'].data['q']
            h_control_tc_next = g.nodes['control-tc'].data['sum_h'] + g.nodes['control-tc'].data['q']
            return h_glass_tc_next, h_control_tc_next

    def _control_to_tc_msg(self, edges):
        u = edges.src['u']
        beta = edges.data['beta']
        return {'msg': beta * self.c2h(u)}

    @staticmethod
    def _tc_to_tc_msg(edges):
        src_h = edges.src['hidden']
        u = edges.data['alpha']
        return {'msg': u * src_h}
