#!/usr/bin/env python3
# 三角形模体管理器
# 高效存储三角形信息，支持快速采样和损失计算
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import igraph as ig
from itertools import combinations
import pickle
import os
import hashlib
from torch.utils.data import Dataset, DataLoader

class TriangleMotifManager:
    """三角形模体管理器 - 高效存储和采样"""
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device=None, cache_dir='./cache'):
        """
        初始化三角形模体管理器

        Args:
            edge_index: [2, E] 边索引
            num_nodes: 节点数量
            device: torch设备
            cache_dir: 缓存目录
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.device = 'cpu'#device or edge_index.device
        self.cache_dir = cache_dir
        # 基于图结构生成唯一ID，避免不同图但节点数一致导致缓存冲突
        self.graph_id = self._compute_graph_id()
        # 缓存文件路径（包含节点数与图ID）
        self.cache_file = os.path.join(cache_dir, f'triangle_motifs_{num_nodes}_{self.graph_id}.pkl')

        # 加载或计算三角形信息
        self._load_or_compute_triangles()
        self._build_adj_list()
        # 在初始化结束时，就将巨大的列表转换为张量，并存储起来
        print("💡 Converting triangle list to tensor for efficient sampling...")
        if len(self.triangles) > 0:
            # 这个张量只在主进程中创建一次
            self.triangle_tensor = torch.tensor(self.triangles, dtype=torch.long)
        else:
            self.triangle_tensor = torch.empty(0, 3, dtype=torch.long)
        print("✅ Conversion complete.")

        # 预计算角度信息（用于损失计算）
        # self._precompute_angles()

    def _compute_graph_id(self) -> str:
        """
        计算当前图的稳定哈希ID：
        - 使用无向边（u<=v）
        - 对边按字典序排序
        - 使用md5生成短ID（前8位）
        """
        edge_index_np = self.edge_index.detach().cpu().numpy()
        # 规范化为无向边表示 (min(u,v), max(u,v))
        u = edge_index_np[0]
        v = edge_index_np[1]
        uv_min = np.minimum(u, v)
        uv_max = np.maximum(u, v)
        edges = np.stack([uv_min, uv_max], axis=1)
        # 去重并排序，确保哈希稳定
        edges = np.unique(edges, axis=0)
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
        hasher = hashlib.md5()
        hasher.update(edges.tobytes())
        return hasher.hexdigest()[:8]

    def _load_or_compute_triangles(self):
        """加载缓存的三角形信息或重新计算"""
        if os.path.exists(self.cache_file):
            print(f"📂 从缓存加载三角形信息: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                cached_gid = cache_data.get('graph_id')
                if cached_gid != self.graph_id:
                    # 缓存与当前图不匹配，重新计算
                    print("⚠️ 缓存的图ID与当前图不一致，重新计算三角形信息...")
                    self._compute_triangles()
                    self._save_to_cache()
                else:
                    self.triangles = cache_data['triangles']
                    self.triangle_set = cache_data['triangle_set']
                    self.node_to_triangles = cache_data['node_to_triangles']
        else:
            print("🔍 计算三角形模体信息...")
            self._compute_triangles()
            self._save_to_cache()

    def _compute_triangles(self):
        """使用motifs_randesu计算所有三角形"""
        # 转换为igraph图
        edge_index_np = self.edge_index.cpu().numpy()
        edges = list(zip(edge_index_np[0], edge_index_np[1]))

        G = ig.Graph(directed=False)
        G.add_vertices(self.num_nodes)
        G.add_edges(edges)
        G.simplify()

        # 存储三角形信息
        triangles = []
        triangle_set = set()
        node_to_triangles = defaultdict(list)

        # igraph的回调函数在某些环境中可能不稳定，我们定义一个内部版本
        found_triangles = G.cliques(min=3, max=3)
        for triangle in found_triangles:
            triangle_tuple = tuple(sorted(triangle))
            triangles.append(triangle_tuple)
            triangle_set.add(triangle_tuple)
            for node in triangle_tuple:
                node_to_triangles[node].append(triangle_tuple)

        self.triangles = list(set(triangles)) # 去重
        self.triangle_set = triangle_set
        self.node_to_triangles = dict(node_to_triangles)

        print(f"✅ 找到 {len(self.triangles)} 个三角形")

    def _save_to_cache(self):
        """保存三角形信息到缓存"""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_data = {
            'triangles': self.triangles,
            'triangle_set': self.triangle_set,
            'node_to_triangles': self.node_to_triangles,
            'graph_id': self.graph_id,
            'num_nodes': self.num_nodes
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"💾 三角形信息已缓存到: {self.cache_file}")

    def _precompute_angles(self):
        """预计算三角形角度信息"""
        print("📐 预计算三角形角度信息...")
        self.triangle_angles = {}

        for triangle in self.triangles:
            i, j, k = triangle
            # 这里可以预计算一些角度相关的信息
            # 具体实现取决于你的角度计算方式
            self.triangle_angles[triangle] = {
                'nodes': (i, j, k),
                'center': j  # 示例：以j为中心的角度
            }

    def get_triangle_tensor(self) -> torch.Tensor:
        """获取三角形张量 [N_triangles, 3]"""
        if len(self.triangles) == 0:
            return torch.empty(0, 3, dtype=torch.long, device=self.device)
        return torch.tensor(self.triangles, dtype=torch.long, device=self.device)
    # 在 TriangleMotifManager 类中添加一个邻接表
    def _build_adj_list(self):
        self.adj = defaultdict(set)
        edge_index_np = self.edge_index.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            u, v = edge_index_np[0, i], edge_index_np[1, i]
            self.adj[u].add(v)
            self.adj[v].add(u)

    def sample_triangles(self, num_samples: int, replace: bool = True) -> torch.Tensor:
        """采样三角形"""
        # if len(self.triangles) == 0:
        #     return torch.empty(0, 3, dtype=torch.long, device=self.device)

        # if replace:
        #     indices = torch.randint(0, len(self.triangles), (num_samples,), device=self.device)
        # else:
        #     num_samples = min(num_samples, len(self.triangles))
        #     indices = torch.randperm(len(self.triangles), device=self.device)[:num_samples]

        # triangle_tensor = self.get_triangle_tensor()
        """采样三角形 (优化版)"""
        if len(self.triangle_tensor) == 0:
            return self.triangle_tensor.to(device='cpu')

        if replace:
            indices = torch.randint(0, len(self.triangle_tensor), (num_samples,), device='cpu')
        else:
            num_samples = min(num_samples, len(self.triangle_tensor))
            indices = torch.randperm(len(self.triangle_tensor), device='cpu')[:num_samples]
        return self.triangle_tensor[indices].cpu()

    def sample_negative_triplets(self, num_samples: int) -> torch.Tensor:
        """采样负样本（非三角形）"""
        negative_triplets = []
        max_attempts = num_samples * 10
        attempts = 0
        while len(negative_triplets) < num_samples and attempts < max_attempts:
            attempts += 1
            # 随机生成三元组
            nodes = torch.randint(0, self.num_nodes, (3,), device=self.device)
            triplet = tuple(sorted(nodes.cpu().tolist()))

            # 检查是否为有效三元组且不是三角形
            if len(set(triplet)) == 3 and triplet not in self.triangle_set:
                negative_triplets.append(list(triplet))

        if len(negative_triplets) == 0:
            return torch.empty(0, 3, dtype=torch.long, device=self.device)

        return torch.tensor(negative_triplets, dtype=torch.long, device=self.device)
    # 优化后的负采样方法
    def sample_negative_triplets_optimized(self, num_samples: int) -> torch.Tensor:
        """
        通过采样“边+随机节点”的方式高效生成负样本
        """
        if not hasattr(self, 'adj'):
            self._build_adj_list()

        negative_triplets = []
        num_edges = self.edge_index.shape[1]

        while len(negative_triplets) < num_samples:
            # 1. 随机选择一条边 (u, v)
            edge_idx = torch.randint(0, num_edges, (1,)).item()
            u, v = self.edge_index[:, edge_idx].tolist()

            # 2. 随机选择一个节点 w
            w = torch.randint(0, self.num_nodes, (1,)).item()

            # 3. 检查是否构成三角形或为无效节点
            if w != u and w != v and w not in self.adj[u]:
                triplet = tuple(sorted([u, v, w]))
                negative_triplets.append(list(triplet))

        if len(negative_triplets) == 0:
            return torch.empty(0, 3, dtype=torch.long, device=self.device)

        return torch.tensor(negative_triplets, dtype=torch.long, device=self.device)


    def get_node_triangles(self, node_id: int) -> List[Tuple[int, int, int]]:
        """获取指定节点参与的所有三角形"""
        return self.node_to_triangles.get(node_id, [])

    def get_triangle_count(self) -> int:
        """获取三角形总数"""
        return len(self.triangles)

    def is_triangle(self, triplet: Tuple[int, int, int]) -> bool:
        """判断元组是否为三角形"""
        return tuple(sorted(triplet)) in self.triangle_set



    # def forward(self, node_embeddings: torch.Tensor, triangles: torch.Tensor,
    #             original_embeddings: torch.Tensor = None) -> torch.Tensor:
    #     """
    #     计算三角形角度重构损失

    #     Args:
    #         node_embeddings: [N, embed_dim] 重构的节点嵌入
    #         triangles: [B, 3] 三角形节点索引
    #         original_embeddings: [N, embed_dim] 原始节点嵌入（可选）

    #     Returns:
    #         loss: 角度重构损失
    #     """
    #     if len(triangles) == 0:
    #         return torch.tensor(0.0, device=node_embeddings.device)

    #     # 获取三角形中每个节点的嵌入
    #     node1_emb = node_embeddings[triangles[:, 0]]  # [B, embed_dim]
    #     node2_emb = node_embeddings[triangles[:, 1]]  # [B, embed_dim]
    #     node3_emb = node_embeddings[triangles[:, 2]]  # [B, embed_dim]

    #     # 计算角度（以node2为中心）
    #     angles_pred = self._compute_angles(node1_emb, node2_emb, node3_emb)

    #     if original_embeddings is not None:
    #         # 计算原始角度
    #         orig_node1_emb = original_embeddings[triangles[:, 0]]
    #         orig_node2_emb = original_embeddings[triangles[:, 1]]
    #         orig_node3_emb = original_embeddings[triangles[:, 2]]
    #         angles_orig = self._compute_angles(orig_node1_emb, orig_node2_emb, orig_node3_emb)

    #         # 计算角度差异损失
    #         angle_loss = F.mse_loss(angles_pred, angles_orig)
    #     else:
    #         # 如果没有原始嵌入，使用正则化损失
    #         # 鼓励角度接近60度（等边三角形）
    #         target_angle = torch.full_like(angles_pred, 60.0 * np.pi / 180.0)
    #         angle_loss = F.mse_loss(angles_pred, target_angle)

    #     return angle_loss
class TriangleAngleLoss(nn.Module):
    """三角形角度重构损失"""
    def __init__(self, temperature: float = 1.0,aggregation: str = 'mean'):
        super(TriangleAngleLoss, self).__init__()
        self.temperature = temperature
        self.aggregation = aggregation
    def _compute_all_angles(self, emb1, emb2, emb3):
        # 计算以 emb2 为中心的角
        angle2 = self._compute_single_angle(emb1, emb2, emb3)
        # 计算以 emb1 为中心的角
        angle1 = self._compute_single_angle(emb2, emb1, emb3)
        # 计算以 emb3 为中心的角
        angle3 = self._compute_single_angle(emb1, emb3, emb2)
        return angle1, angle2, angle3
    def forward(self, node_embeddings: torch.Tensor, triangles: torch.Tensor,
                original_embeddings = None) -> torch.Tensor:
        if len(triangles) == 0:
            return torch.tensor(0.0, device=node_embeddings.device)

        node1_emb = node_embeddings[triangles[:, 0]]
        node2_emb = node_embeddings[triangles[:, 1]]
        node3_emb = node_embeddings[triangles[:, 2]]

        # 计算预测的三个角
        angles_pred_1, angles_pred_2, angles_pred_3 = self._compute_all_angles(node1_emb, node2_emb, node3_emb)

        # if original_embeddings is not None:
        orig_node1_emb = original_embeddings[triangles[:, 0]]
        orig_node2_emb = original_embeddings[triangles[:, 1]]
        orig_node3_emb = original_embeddings[triangles[:, 2]]
        
        # 计算原始的三个角
        angles_orig_1, angles_orig_2, angles_orig_3 = self._compute_all_angles(orig_node1_emb, orig_node2_emb, orig_node3_emb)
        
        # 分别计算三个角的损失
        loss1 = F.mse_loss(angles_pred_1, angles_orig_1)
        loss2 = F.mse_loss(angles_pred_2, angles_orig_2)
        loss3 = F.mse_loss(angles_pred_3, angles_orig_3)
        # else:
        #     # 鼓励所有角都接近60度
        #     target_angle = torch.full_like(angles_pred_1, 60.0 * np.pi / 180.0)
        #     loss1 = F.mse_loss(angles_pred_1, target_angle)
        #     loss2 = F.mse_loss(angles_pred_2, target_angle)
        #     loss3 = F.mse_loss(angles_pred_3, target_angle)
        
        # 聚合损失
        if self.aggregation == 'mean':
            return (loss1 + loss2 + loss3) / 3.0
        else:
            return loss1 + loss2 + loss3

    def _compute_single_angle(self, node1_emb: torch.Tensor, node2_emb: torch.Tensor,
                        node3_emb: torch.Tensor) -> torch.Tensor:
        """计算以node2为中心的角度"""
        # 计算向量
        vec1 = node1_emb - node2_emb  # [B, embed_dim]
        vec3 = node3_emb - node2_emb  # [B, embed_dim]

        # 计算余弦值
        cos_angle = F.cosine_similarity(vec1, vec3, dim=1, eps=1e-8)

        # --- 修改后的代码 ---
        # 引入一个微小的 epsilon 来避免边界问题
        eps = 1e-7
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)

        # 转换为角度（弧度）
        angles = torch.acos(cos_angle)

        return angles


class TriangleMotifLoss(nn.Module):
    """三角形模体预测损失"""
    def __init__(self, embed_dim: int, temperature: float = 1.0):
        super(TriangleMotifLoss, self).__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        # 三角形预测器
        self.triangle_predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings: torch.Tensor, triangles: torch.Tensor,
                negative_triplets: torch.Tensor = None) -> torch.Tensor:
        """
        计算三角形模体预测损失

        Args:
            node_embeddings: [N, embed_dim] 节点嵌入
            triangles: [B_pos, 3] 正样本三角形
            negative_triplets: [B_neg, 3] 负样本三元组

        Returns:
            loss: 模体预测损失
        """
        if len(triangles) == 0:
            return torch.tensor(0.0, device=node_embeddings.device)

        # 正样本预测
        pos_predictions = self._predict_triangles(node_embeddings, triangles)
        pos_labels = torch.ones_like(pos_predictions)

        # 负样本预测
        if negative_triplets is not None and len(negative_triplets) > 0:
            neg_predictions = self._predict_triangles(node_embeddings, negative_triplets)
            neg_labels = torch.zeros_like(neg_predictions)

            # 合并正负样本
            all_predictions = torch.cat([pos_predictions, neg_predictions], dim=0)
            all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        else:
            all_predictions = pos_predictions
            all_labels = pos_labels

        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy(all_predictions, all_labels)

        return loss

    def _predict_triangles(self, node_embeddings: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
        """预测元组是否为三角形"""
        # 获取三元组中每个节点的嵌入
        node1_emb = node_embeddings[triplets[:, 0]]
        node2_emb = node_embeddings[triplets[:, 1]]
        node3_emb = node_embeddings[triplets[:, 2]]

        # 拼接三个节点的嵌入
        triplet_emb = torch.cat([node1_emb, node2_emb, node3_emb], dim=1)

        # 预测
        predictions = self.triangle_predictor(triplet_emb).squeeze(-1)

        return predictions


def create_triangle_motif_losses(embed_dim: int,
                                 device: torch.device) -> Dict[str, nn.Module]:
    """创建三角形模体相关的损失函数"""
    losses = {
        'angle_loss': TriangleAngleLoss().to(device),
        'motif_loss': TriangleMotifLoss(embed_dim).to(device)
    }
    return losses


def sample_triangle_batch(triangle_manager: TriangleMotifManager, batch_size: int,
                          negative_ratio: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样三角形批次用于训练

    Args:
        triangle_manager: 三角形管理器
        batch_size: 批次大小
        negative_ratio: 负样本比例

    Returns:
        triangles: [B_pos, 3] 正样本三角形
        negative_triplets: [B_neg, 3] 负样本三元组
        labels: [B_pos + B_neg] 标签 (为兼容性保留，但通常损失函数内部会生成)
    """
    # 采样正样本
    num_positive = batch_size
    triangles = triangle_manager.sample_triangles(num_positive)

    # 采样负样本
    num_negative = int(batch_size * negative_ratio)
    negative_triplets = triangle_manager.sample_negative_triplets_optimized(num_negative)

    # 创建标签 (可选，因为损失函数内部处理)
    pos_labels = torch.ones(len(triangles), device=triangles.device)
    neg_labels = torch.zeros(len(negative_triplets), device=negative_triplets.device)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    return triangles, negative_triplets, labels



class TriangleDataset(Dataset):
    def __init__(self, manager: TriangleMotifManager, epoch_size: int, batch_size: int, negative_ratio: float):
        self.manager = manager
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # 每次调用，都采样一个新的批次所需的数据
        num_positive = self.batch_size
        num_negative = int(self.batch_size * self.negative_ratio)

        pos_triangles = self.manager.sample_triangles(num_positive)
        neg_triplets = None#self.manager.sample_negative_triplets_optimized(num_negative) # 使用优化后的采样

        return pos_triangles, neg_triplets
class TriangleDataset_sample(Dataset):
    def __init__(self, manager: TriangleMotifManager):
        self.manager = manager
        # 使用预先转换好的张量，效率更高
        self.triangle_tensor = manager.triangle_tensor 
        # 现在，数据集的大小就是三角形的总数
        self.num_triangles = len(self.triangle_tensor)

    def __len__(self):
        return self.num_triangles

    def __getitem__(self, idx):
        # 返回索引为idx的单个三角形
        return self.triangle_tensor[idx]