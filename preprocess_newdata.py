import os
import random
import torch
import torch.nn as nn
import numpy as np
import scipy
import scipy.stats as st

from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected, degree
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.utils import get_laplacian
from torch_geometric.transforms import ToUndirected
import scipy as sp
import scipy.sparse as sps
import time
from scipy.io import loadmat
from collections import Counter
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Actor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def load_blog(path="../dataset/"):
    dataset = "blog"
    # print('Loading {} dataset...'.format(dataset))
    adj = sps.load_npz(path+dataset+"/adj.npz")
    features = sps.load_npz(path+dataset+"/feat.npz")
    labels = np.load(path+dataset+"/label.npy")
    idx_train20 = np.load(path+dataset+"/train20.npy")
    idx_val = np.load(path+dataset+"/val.npy")
    idx_test = np.load(path+dataset+"/test.npy")

    adj = adj.todense()
    row, col = np.where(adj != 0)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    x = torch.tensor(features.todense(), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.int64)
    data = Data(x=x, edge_index=edge_index, y=y)

    num_nodes = x.size(0)  # Get the number of nodes in the dataset
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[idx_train20] = 1
    val_mask[idx_val] = 1  # Set the validation indices to 1
    test_mask[idx_test] = 1  # Set the test indices to 1
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


## load blog
# data = load_blog()


### load chameleon
# dataset = WikipediaNetwork(root='../dataset/WikipediaNetwork', name='chameleon', transform=NormalizeFeatures())
# data = dataset[0]


### load squirrel
# dataset = WikipediaNetwork(root='../dataset/WikipediaNetwork', name='squirrel', transform=NormalizeFeatures())
# data = dataset[0]


### load actor
# dataset = Actor(root='../dataset/Actor', transform=NormalizeFeatures())
# data = dataset[0]

### load roman-empire
# 导入所需的库
# T 是 torch_geometric.transforms 的通用别名
import torch_geometric.transforms as T
from torch_geometric.datasets import HeterophilousGraphDataset


# # 4. 从数据集中获取图数据对象（这个集合里只有一个图）
# data = dataset[0]

dataname = 'flickr'
# 1. 指定一个根目录用于存放数据集，建议使用数据集本身的名字命名文件夹
#    这样更清晰，避免混淆
root_path = '../dataset/{}'.format(dataname)

from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import DeezerEurope

# dataset = DeezerEurope(root='../dataset/DeezerEurope', transform=NormalizeFeatures())
# data = dataset[0]
from torch_geometric.datasets import AttributedGraphDataset

dataset = AttributedGraphDataset(root='../dataset/flickr',name='flickr')#, transform=NormalizeFeatures())
data = dataset[0]
def print_stats(data):
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    feat_dim = data.x.size(1) if hasattr(data, 'x') and data.x is not None else 0
    num_classes = int(data.y.max().item() + 1) if hasattr(data, 'y') and data.y is not None else '未知'
    print('=== Flickr 数据集统计 ===')
    print(f'- 节点数: {num_nodes}')
    print(f'- 边数: {num_edges}')
    print(f'- 特征维度: {feat_dim}')
    print(f'- 类别数: {num_classes}')
    if hasattr(data, 'train_mask'):
        print(f'- 训练节点: {int(data.train_mask.sum())}')
    if hasattr(data, 'val_mask'):
        print(f'- 验证节点: {int(data.val_mask.sum())}')
    if hasattr(data, 'test_mask'):
        print(f'- 测试节点: {int(data.test_mask.sum())}')
print_stats(data)
# 先把稀疏特征转为稠密
# if hasattr(data.x, 'to_dense'):
#     data.x = data.x.to_dense().float()

# # 再做特征归一化
# data = NormalizeFeatures()(data)
# data = torch.load('../dataset/DeezerEurope/processed/data.pt')
# transform = NormalizeFeatures()
# data = transform(data)
# 加载 Cornell,Wisconsin,Texas
# dataset = WebKB(root='../dataset/WebKB', name=dataname, transform=NormalizeFeatures())
# # 4. 从数据集中获取图数据对象（这个集合里只有一个图）
# data = dataset[0]

# if data.is_directed():
#     print("is not undirected")
#     data.edge_index = to_undirected(data.edge_index)

# index, attr = get_laplacian(data.edge_index, normalization='sym')
# L = to_scipy_sparse_matrix(index, attr)

# L = torch.FloatTensor(L.todense())
# e, u = torch.linalg.eigh(L)

# # e, u = scipy.sparse.linalg.eigsh(L, k=800, which='SA', tol=1e-5)

# data.e = torch.FloatTensor(e)
# data.u = torch.FloatTensor(u)

# torch.save(data, '../dataset/{}.pt'.format(dataname.lower()))