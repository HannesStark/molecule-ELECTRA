import torch
from ogb.utils.features import get_atom_feature_dims
from torch import nn
from torch.nn import CrossEntropyLoss

from models.base_layers import MLP
from models.pna import PNAGNN


class ELECTRA(nn.Module):

    def __init__(self, pna_generator_args, pna_discriminator_args, device, generator_out_layers=1,
                 discriminator_out_layers=1, **kwargs):
        super(ELECTRA, self).__init__()
        self.generator = PNAGNN(padding=True, **pna_generator_args)
        self.feature_dims = torch.tensor([0] + get_atom_feature_dims(), device=device)
        self.feature_dims_cumsum = torch.cumsum(self.feature_dims, dim=0)
        self.generator_out = MLP(in_dim=pna_generator_args['hidden_dim'], out_dim=self.feature_dims.sum(),
                                 layers=generator_out_layers)

        # discriminator:
        self.node_gnn = PNAGNN(padding=True, **pna_discriminator_args)
        self.discriminator_out = MLP(in_dim=pna_discriminator_args['hidden_dim'], out_dim=1,
                                     layers=discriminator_out_layers)
        self.cross_entropy_loss = CrossEntropyLoss()

    def forward(self, graph, mask):
        # shape of mask is [n_atoms, 1]
        self.generator(graph)
        reconstructions = graph.ndata['feat']
        reconstructions = self.generator_out(reconstructions)

        fake_features = []
        generator_loss = 0
        for i, start in enumerate(self.feature_dims_cumsum[:-1]):
            reconstruction = reconstructions[:, start: self.feature_dims_cumsum[i + 1]]
            true_feat = graph.ndata['true_feat'][:, i]
            generator_loss += self.cross_entropy_loss(reconstruction, true_feat)
            reconstruction = reconstruction.detach()
            reconstruction_probs = torch.softmax(reconstruction, dim=1)
            fake_features.append(torch.argmax(reconstruction_probs, dim=1))
        fake_features = torch.stack(fake_features, dim=1)

        # take fake features for masked tokens and the original ones for the rest
        graph.ndata['feat'] = mask * graph.ndata['true_feat'] + (~mask) * fake_features
        graph.edata['feat'] = graph.edata['true_feat']

        # discriminator part:
        self.node_gnn(graph)
        predictions = self.discriminator_out(graph.ndata['feat'])

        return generator_loss, predictions
