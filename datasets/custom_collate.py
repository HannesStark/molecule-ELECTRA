import copy
from typing import List, Tuple

import dgl
import torch
from torch.nn.utils.rnn import pad_sequence

from commons.utils import get_adj_matrix


def graph_collate(batch: List[Tuple]):
    graphs, targets = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    return [batched_graph], torch.stack(targets).float()


def s_norm_graph_collate(batch: List[Tuple]):
    graphs, targets = map(list, zip(*batch))
    tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
    snorm_n = torch.cat(tab_snorm_n).sqrt()
    batched_graph = dgl.batch(graphs)
    return [batched_graph], snorm_n, torch.stack(targets).float()


class MaskedCollate(object):
    def __init__(self, mask_probability=0.15):
        self.mask_probability = mask_probability

    def __call__(self, batch: List[dgl.DGLGraph]):
        graph = dgl.batch(batch)
        graph.ndata['true_feat'] = graph.ndata['feat']
        graph.edata['true_feat'] = graph.edata['feat']
        feat = graph.ndata['feat']
        # mask contains False if the value is masked out
        mask = (torch.rand(graph.num_nodes(), device=graph.device) > self.mask_probability).unsqueeze(1)  # [n_atoms, 1]
        feat = mask * feat - (~mask) * torch.ones_like(feat)
        graph.ndata['feat'] = feat

        return graph, mask


def padded_collate(batch):
    features = pad_sequence([item[0] for item in batch], batch_first=True)
    targets = torch.stack([item[1] for item in batch])

    # create mask corresponding to the zero padding used for the shorter sequences in the batch.
    # All values corresponding to padding are True and the rest is False.
    n_atoms = torch.tensor([len(item[0]) for item in batch])
    mask = torch.arange(features.shape[1])[None, :] >= n_atoms[:, None]  # [batch_size, n_atoms]
    return [features, mask], targets.float()
