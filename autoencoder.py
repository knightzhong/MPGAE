from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import remove_self_loops, to_undirected
from utils import get_activation, noise_fn

class DistanceHead(nn.Module):
    def __init__(self, node_feat_dim, heads, activation_fn, edge_batch_size: int = 200000):
        super().__init__()
        # 输入维度是两个节点特征拼接后的维度
        self.dense = nn.Linear(node_feat_dim * 2, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.out_proj = nn.Linear(128, 1) # 输出一个距离值
        self.activation_fn = get_activation(activation_fn)
        # 为大图提供分块计算，缓解显存峰值
        self.edge_batch_size = edge_batch_size

    def forward(self, h, edge_index):
        # 分块处理边，避免在大图上一次性拼接造成显存峰值
        src_nodes, dst_nodes = edge_index
        num_edges = src_nodes.numel()
        batch_size = self.edge_batch_size if self.edge_batch_size and self.edge_batch_size > 0 else num_edges

        outputs = []
        for start in range(0, num_edges, batch_size):
            end = min(start + batch_size, num_edges)
            src_batch = src_nodes[start:end]
            dst_batch = dst_nodes[start:end]

            h_src = h[src_batch]
            h_dst = h[dst_batch]

            edge_h = torch.cat([h_src, h_dst], dim=-1)

            dist = self.dense(edge_h)
            dist = self.activation_fn(dist)
            dist = self.layer_norm(dist)
            dist = self.out_proj(dist)
            outputs.append(dist)

        dist_all = torch.cat(outputs, dim=0)
        return dist_all.squeeze(), None # 返回预测的距离

class MaskLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, mask_tokens=None):
        if mask_tokens is not None:
            features = features[mask_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias

        return x

class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, num_atom_type=0, args=None):
        super(GraphAutoEncoder, self).__init__()
        self.args = args
        self.encoder = encoder

        self.mask_ratio = args.mask_ratio
        self.replace_ratio = args.replace_ratio
        self.noise_val = args.noise_val
        self.masked_atom_loss = float(args.masked_atom_loss)
        self.masked_pe_loss = float(args.masked_pe_loss)
        self.atom_recon_type = args.atom_recon_type
        self.num_atom_type = num_atom_type
        self.alpha_l = args.alpha_l

        self.enc_mask_token = nn.Parameter(torch.zeros(1, args.feat_dim))
        self.node_pred = MaskLMHead(args.embed_dim, output_dim=self.num_atom_type, activation_fn=args.task_head_activation)
        self.pe_reconstruct_heads = DistanceHead(
            node_feat_dim=args.embed_dim,
            heads=args.heads,
            activation_fn=args.task_head_activation,
            edge_batch_size=getattr(args, 'edge_batch_size', None)#250000
        )
    def forward(self, x, edge_index, u, PE, edge_index_pe=None, triangles=None):
        u_transformed = u
        x_masked, u_masked, mask_tokens = self.encoding_mask_noise(
            x=x,
            u=u_transformed,
            mask_ratio=self.mask_ratio,
            replace_ratio=self.replace_ratio,
        )

        PE_noise = torch.linalg.norm(
            u_masked[edge_index_pe[0]] - u_masked[edge_index_pe[1]], dim=-1
        )
        enc_rep_ori, pe = self.encoder(
            x, x_masked, edge_index, PE=PE, PE_noise=PE_noise, motif_tensor=triangles
        )

        enc_rep = self.node_pred(enc_rep_ori, mask_tokens)
        reconstruct_dist, _ = self.pe_reconstruct_heads(pe, edge_index_pe)
        atom_loss = self.cal_atom_loss(
            pred_node=enc_rep,
            target_atom=x,
            mask_tokens=mask_tokens,
            loss_fn=self.atom_recon_type,
            alpha_l=self.alpha_l,
        )
        pe_loss = self.cal_pe_loss(
            reconstruct_dis=reconstruct_dist,
            target_dis=PE,
            edge_index_pe=edge_index_pe,
            mask_tokens=mask_tokens,
        )

        loss = self.masked_atom_loss * atom_loss + self.masked_pe_loss * pe_loss
        return loss, u_transformed
    def encoding_mask_noise(self, x, u, mask_ratio, replace_ratio):
        mask_token_ratio = 1 - replace_ratio
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        if replace_ratio > 0:
            num_noise_nodes = int(replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(mask_token_ratio * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(replace_ratio * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        u_masked = None

        u_masked = u.clone()
        pos_noise = noise_fn(self.noise_val, len(mask_nodes), u.size(1)).to(u_masked.device)
        u_masked[mask_nodes] += pos_noise

        return out_x, u_masked, mask_nodes
    
    def embed(self, x, edge_index,u, edge_index_pe, triangles, return_gates=False):        
        # 推理时也使用相同的尺度稳定机制
        u_transformed = u
        
        PE = torch.linalg.norm(u_transformed[edge_index_pe[0]] - u_transformed[edge_index_pe[1]], dim=-1)
        
        # 如果PE统计信息已初始化，则进行尺度匹配
        # if self._pe_stats_initialized and hasattr(self, '_pe_stats'):
        #     PE = (PE - self._pe_stats['mean']) / (self._pe_stats['std'] + 1e-8)
        #     PE = PE * self._pe_stats['std'] + self._pe_stats['mean']
        
        result = self.encoder.embed(x, edge_index, PE=PE, motif_tensor=triangles, return_gates=return_gates)
        if return_gates:
            enc_rep, _, all_gates = result
            return enc_rep, all_gates
        else:
            enc_rep, _ = result
            return enc_rep


    def cal_pe_loss(self, reconstruct_dis, target_dis, edge_index_pe, mask_tokens):
        # 对 target 与 reconstruct 同步执行：去自环 + 无向聚合，确保对齐同一条边集合
        edge_index_t, target_dis = remove_self_loops(edge_index=edge_index_pe, edge_attr=target_dis)
        edge_index_t, target_dis = to_undirected(edge_index=edge_index_t, edge_attr=target_dis, reduce="mean")

        edge_index_r, reconstruct_dis = remove_self_loops(edge_index=edge_index_pe, edge_attr=reconstruct_dis)
        edge_index_r, reconstruct_dis = to_undirected(edge_index=edge_index_r, edge_attr=reconstruct_dis, reduce="mean")

        # 现在两者应具有相同的边顺序和长度
        row = edge_index_t[0]
        idx = torch.isin(row, mask_tokens)

        reconstruct_dis = reconstruct_dis[idx]
        target_dis = target_dis[idx]
        
        # 检查是否有有效的边（防止空张量导致NaN）
        if reconstruct_dis.numel() == 0 or target_dis.numel() == 0:
            # 从edge_index_pe获取device信息
            device = edge_index_pe.device if isinstance(edge_index_pe, torch.Tensor) else target_dis.device
            return torch.tensor(0.0, device=device, dtype=target_dis.dtype if target_dis.numel() > 0 else reconstruct_dis.dtype)
        
        # 检查是否包含NaN或Inf
        if torch.isnan(reconstruct_dis).any() or torch.isnan(target_dis).any():
            return torch.tensor(0.0, device=reconstruct_dis.device, dtype=reconstruct_dis.dtype)
        if torch.isinf(reconstruct_dis).any() or torch.isinf(target_dis).any():
            # 将Inf值裁剪到合理范围
            reconstruct_dis = torch.clamp(reconstruct_dis, min=-1e6, max=1e6)
            target_dis = torch.clamp(target_dis, min=-1e6, max=1e6)

        pe_reconstruct_loss = F.smooth_l1_loss(
            reconstruct_dis,
            target_dis,
            reduction="mean",
            beta=1.0,
        )
        
        # 检查loss是否为NaN
        if torch.isnan(pe_reconstruct_loss):
            print("pe_reconstruct_loss is NaN")
            return torch.tensor(0.0, device=reconstruct_dis.device, dtype=reconstruct_dis.dtype)

        return pe_reconstruct_loss


    def cal_atom_loss(self, pred_node, target_atom, mask_tokens, loss_fn, alpha_l=0.0):
        target_atom = target_atom[mask_tokens]

        if loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
            atom_loss = criterion(pred_node, target_atom)
        elif loss_fn == "mse":
            atom_loss = self.mse_loss(pred_node, target_atom)
        else:
            criterion = nn.CrossEntropyLoss()
            atom_loss = criterion(pred_node, target_atom)

        return atom_loss

    def sce_loss(self, x, y, alpha=1):
        # 检查输入是否包含NaN或Inf
        if torch.isnan(x).any() or torch.isnan(y).any():
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # 裁剪极端值，防止normalize时出现问题
        x = torch.clamp(x, min=-1e6, max=1e6)
        y = torch.clamp(y, min=-1e6, max=1e6)
        
        x = F.normalize(x, p=2, dim=-1, eps=1e-8)
        y = F.normalize(y, p=2, dim=-1, eps=1e-8)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        
        # 检查loss是否为NaN
        if torch.isnan(loss):
            print("sce loss is NaN")
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        return loss

    def mse_loss(self, x, y):
        loss = ((x - y) ** 2).mean()
        return loss
    
    # def _init_pe_stats(self, PE):
    #     """
    #     初始化PE统计信息，用于后续的尺度匹配
    #     Args:
    #         PE: 位置编码张量
    #     """
    #     with torch.no_grad():
    #         self._pe_stats = {
    #             'mean': PE.mean().item(),
    #             'std': PE.std().item()
    #         }
    #         self._pe_stats_initialized = True
    #         print(f"[PE Stats] Initialized - Mean: {self._pe_stats['mean']:.6f}, Std: {self._pe_stats['std']:.6f}")