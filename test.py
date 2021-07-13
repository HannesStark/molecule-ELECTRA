import torch 
import dgl
from datasets.ogbg_dataset_extension import OGBGDatasetExtension

datas = OGBGDatasetExtension(return_types=['dgl_graph', 'targets'], name='ogbg-molbace', device='cpu')
print(datas[0])
print(datas[0][0].ndata)