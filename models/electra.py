import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.mpn = MPNEncoder(args)
        self.atom_fdim = get_atom_fdim()
        self.W_o = nn.Linear(args.hidden_size, self.atom_fdim)
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    # masked_graph is molecules with masked atoms
    # true_fatoms is the original atom features
    def forward(self, masked_graph, true_fatoms):
        mol_vecs = self.mpn(masked_graph)
        loss = tot = 0.
        pred_fatoms = []

        for atom_hiddens, flabels in zip(mol_vecs, true_fatoms):
            out = self.W_o(atom_hiddens)
            loss = loss + self.loss(out, flabels)
            tot = tot + len(atom_hiddens)
            # In ELECTRA, generator and discriminator are disconnected (see paper)
            out = torch.sigmoid(out).detach()
            pred_fatoms.append(out)

class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.mpn = MPNEncoder(args)
        self.W_o = nn.Linear(args.hidden_size, 1)
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    # masked_graph is molecules with corrupted atom featuress
    # mask_labels = 1 means this atom is corrupted
    def forward(self, corrupt_graph, mask_labels):
        mol_vecs = self.mpn(masked_graph)
        loss = tot = 0.
        for atom_hiddens, mlabels in zip(mol_vecs, mask_labels):
            out = self.W_o(atom_hiddens).squeeze(-1)
            loss = loss + self.loss(out, mlabels)
            tot = tot + len(atom_hiddens)
        return loss / tot