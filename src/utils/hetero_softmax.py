import dgl
import torch


def hetero_edge_softmax(g, etype,
                        logit_from: str,
                        logit_field='z',
                        attn_field='attn'):
    with g.local_scope():
        g.edges[etype].data[logit_field] = torch.exp(g.edges[etype].data[logit_from])
        g.update_all(dgl.function.copy_e(logit_field, logit_field),
                     dgl.function.sum(logit_field, 'sum_logit'),
                     etype=etype)

        def _compute_sm(edges):
            summed = edges.dst['sum_logit']
            logit = edges.data[logit_field]
            return {attn_field: logit / summed}

        g.apply_edges(_compute_sm, etype=etype)
        return g.edges[etype].data[attn_field]
