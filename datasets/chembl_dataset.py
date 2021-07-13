import os
import torch.nn.functional as F
import dgl
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Subset


class ChEMBLDataset(torch.utils.data.Dataset):
    def __init__(self, root='dataset/chembl'):
        super(ChEMBLDataset, self).__init__()
        
        self.root = root


        if not os.path.exists(os.path.join(self.root, 'process.pt')):
            self.process()


    def process(self):
        smiles = pd.read_csv(os.path.join(self.root,'chembl.csv'))
        for mol
