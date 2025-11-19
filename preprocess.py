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
from torch_geometric.datasets import Planetoid
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

### load CiteSeer (Planetoid)
# dataset = Planetoid(root='../dataset/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
# data = dataset[0]

### load PubMed (Planetoid)
dataset = Planetoid(root='../dataset/Planetoid', name='PubMed', transform=NormalizeFeatures())
data = dataset[0]

### load roman-empire
# 导入所需的库
# T 是 torch_geometric.transforms 的通用别名
# import torch_geometric.transforms as T
# from torch_geometric.datasets import HeterophilousGraphDataset

# 1. 指定一个根目录用于存放数据集，建议使用数据集本身的名字命名文件夹
#    这样更清晰，避免混淆
# root_path = '../dataset/roman-empire'

# # 2. 定义预处理转换，NormalizeFeatures() 会对节点特征进行L2范数归一化
# transform = T.NormalizeFeatures()

# # 3. 使用正确的 HeterophilousGraphDataset 类来加载 'roman-empire' 数据集
# #    通过 name 参数指定具体是哪个图
# dataset = HeterophilousGraphDataset(root=root_path, name='roman-empire', transform=transform)

# # 4. 从数据集中获取图数据对象（这个集合里只有一个图）
# data = dataset[0]

# ### load amazon-ratings
# dataset = WikipediaNetwork(root='../dataset/WikipediaNetwork', name='amazon-ratings', transform=NormalizeFeatures())
# 1. 指定一个根目录用于存放数据集，建议使用数据集本身的名字命名文件夹
#    这样更清晰，避免混淆
# root_path = '../dataset/amazon-ratings'

# # 2. 定义预处理转换，NormalizeFeatures() 会对节点特征进行L2范数归一化
# transform = T.NormalizeFeatures()

# # 3. 使用正确的 HeterophilousGraphDataset 类来加载 'roman-empire' 数据集
# #    通过 name 参数指定具体是哪个图
# dataset = HeterophilousGraphDataset(root=root_path, name='amazon-ratings', transform=transform)

# # 4. 从数据集中获取图数据对象（这个集合里只有一个图）
# data = dataset[0]

# ### load minesweeper
# dataset = WikipediaNetwork(root='../dataset/WikipediaNetwork', name='minesweeper', transform=NormalizeFeatures())
# 1. 指定一个根目录用于存放数据集，建议使用数据集本身的名字命名文件夹
#    这样更清晰，避免混淆
# root_path = '../dataset/minesweeper'

# # 2. 定义预处理转换，NormalizeFeatures() 会对节点特征进行L2范数归一化
# transform = T.NormalizeFeatures()

# # 3. 使用正确的 HeterophilousGraphDataset 类来加载 'roman-empire' 数据集
# #    通过 name 参数指定具体是哪个图
# dataset = HeterophilousGraphDataset(root=root_path, name='minesweeper', transform=transform)

# # 4. 从数据集中获取图数据对象（这个集合里只有一个图）
# data = dataset[0]

if data.is_directed():
    print("is not undirected")
    data.edge_index = to_undirected(data.edge_index)

index, attr = get_laplacian(data.edge_index, normalization='sym')
L = to_scipy_sparse_matrix(index, attr)

L = torch.FloatTensor(L.todense())
e, u = torch.linalg.eigh(L)

# e, u = scipy.sparse.linalg.eigsh(L, k=800, which='SA', tol=1e-5)

data.e = torch.FloatTensor(e)
data.u = torch.FloatTensor(u)

torch.save(data, '../dataset/{}.pt'.format(dataset.name.lower()))