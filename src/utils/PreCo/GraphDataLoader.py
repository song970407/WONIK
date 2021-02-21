import dgl
import torch
from torch.utils.data import DataLoader


class GraphDataLoader(DataLoader):

    def __init__(self, g,
                 device: str = 'cpu',
                 use_position_feat: bool = False,
                 *args, **kwargs):
        self.g = g
        self.device = device
        self.use_position_feat = use_position_feat

        collate_fn = self.collate_fn
        kwargs['collate_fn'] = collate_fn
        super(GraphDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        # history xs [batch size x #. tc x history len x state_dim]
        hist_x = torch.stack([item[0] for item in batch])
        # future xs [batch size x #. tc x future len x state_dim]
        future_x = torch.stack([item[1] for item in batch])
        # history us [batch size x #. control x history len x control_dim]
        hist_u = torch.stack([item[2] for item in batch])
        # future us [batch size x #. control x future len x control_dim]
        future_u = torch.stack([item[3] for item in batch])

        hist_len, future_len = hist_x.shape[-2], future_x.shape[-2]
        state_dim, control_dim = hist_x.shape[-1], hist_u.shape[-1]
        n_graphs = hist_x.shape[0]
        g = dgl.batch([self.g for _ in range(n_graphs)])
        g.nodes['tc'].data['history'] = hist_x.reshape(-1, hist_len, state_dim)
        g.nodes['tc'].data['future'] = future_x.reshape(-1, future_len, state_dim)
        g.nodes['control'].data['history'] = hist_u.reshape(-1, hist_len - 1, control_dim)
        g.nodes['control'].data['future'] = future_u.reshape(-1, future_len, control_dim)

        # if self.use_position_feat:
        #     hist_len = g.nodes['tc'].data['history'].shape[1]
        #     future_len = g.nodes['tc'].data['future'].shape[1]
        #     tc_pos = g.nodes['tc'].data['position'].unsqueeze(dim=1)
        #     control_pos = g.nodes['control'].data['position'].unsqueeze(dim=1)
        #
        #     g.nodes['tc'].data['history_aug'] = torch.cat([g.nodes['tc'].data['history'],
        #                                                    tc_pos.repeat(1, hist_len, 1)], dim=-1)
        #     g.nodes['tc'].data['future_aug'] = torch.cat([g.nodes['tc'].data['future'],
        #                                                   tc_pos.repeat(1, future_len, 1)], dim=-1)
        #     g.nodes['control'].data['history_aug'] = torch.cat([g.nodes['control'].data['history'],
        #                                                         control_pos.repeat(1, hist_len-1, 1)], dim=-1)
        #     g.nodes['control'].data['future_aug'] = torch.cat([g.nodes['control'].data['future'],
        #                                                        control_pos.repeat(1, future_len, 1)], dim=-1)

        # g = g.to(self.device)
        return g
