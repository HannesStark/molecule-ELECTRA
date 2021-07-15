import os
import torch.nn.functional as F
import dgl
import numpy as np
import torch
import pandas as pd
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem

from torch.utils.data import Subset
from tqdm import tqdm


class ChEMBLDataset(torch.utils.data.Dataset):
    def __init__(self, root='dataset/chembl', device='cuda:0', **kwargs):
        super(ChEMBLDataset, self).__init__()
        self.dgl_graphs = {}
        self.root = root
        self.device = device
        self.processed_file = 'processed.pt'
        smiles = pd.read_csv(os.path.join(self.root, 'chembl_smiles.csv'))
        smiles = smiles[smiles.smiles != '[Cl-].[Li+]']
        self.smiles = smiles[smiles.smiles != '[Cl-].[Cl-].[Zn+2]']['smiles']

    def __getitem__(self, idx):
        if idx in self.dgl_graphs:
            return self.dgl_graphs[idx].to(self.device)
        else:
            ndata, edata, edge_indices, n_atoms = self.get_features(self.smiles[idx])

            g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms, device=self.device)
            g.ndata['feat'] = ndata.to(self.device)
            g.edata['feat'] = edata.to(self.device)
            self.dgl_graphs[idx] = g.to('cpu')
            return g.to(self.device)

    def __len__(self):
        return len(self.smiles)

    def get_features(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        # add hydrogen bonds to molecule because they are not in the smiles representation
        mol = Chem.AddHs(mol)
        atom_features_list = []
        n_atoms = len(mol.GetAtoms())
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))

        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(edges_list, dtype=torch.long).T
        edge_features = torch.tensor(edge_features_list, dtype=torch.long)
        return torch.tensor(atom_features_list), edge_features, edge_index, n_atoms
