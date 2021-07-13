import torch 
import dgl
from datasets.ogbg_dataset_extension import OGBGDatasetExtension

dataset = OGBGDatasetExtension(return_types=['dgl_graph', ])