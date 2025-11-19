import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.typing import SparseTensor#, torch_sparse
import torch_sparse
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax,
)

class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        args,
        concat = True,
        negative_slope = 0.2,
        add_self_loops = True,
        edge_dim = None,
        fill_value = 'mean',
        bias = True,
        residual = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dropout = args.gnn_edge_dropout
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual

        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = nn.Linear(in_channels, self.heads * out_channels, bias=False)
        else:
            self.lin_src = nn.Linear(in_channels[0], self.heads * out_channels, False)
            self.lin_dst = nn.Linear(in_channels[1], self.heads * out_channels, False)

        self.att_src = Parameter(torch.FloatTensor(size=(1, self.heads, out_channels)))
        self.att_dst = Parameter(torch.FloatTensor(size=(1, self.heads, out_channels)))

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, self.heads * out_channels, bias=False)
            self.att_edge = Parameter(torch.FloatTensor(size=(1, self.heads, out_channels)))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        total_out_channels = out_channels * (self.heads if concat else 1)

        if residual:
            self.res = nn.Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False
            )
            self.add_linear = nn.Linear(total_out_channels*2, total_out_channels, bias=True)
            self.fusion_norm_res = nn.LayerNorm(total_out_channels * 2)

        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.FloatTensor((total_out_channels)))
        else:
            self.register_parameter('bias', None)
        self.use_motif = True
        if self.use_motif == True:
            # 添加motif self-attention：让motif之间相互交互
            # 注意：这里需要使用heads * out_channels作为embed_dim，因为concat=True时输出维度是heads * out_channels
            total_out_channels = out_channels * (self.heads if self.concat else 1)

            self.fusion_norm = nn.LayerNorm(total_out_channels * 2)
            
            # self.motif_down_proj = nn.Sequential(
            #     nn.Linear(total_out_channels, total_out_channels, bias=True),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(total_out_channels, total_out_channels, bias=True)
            # )
            # --- 新增：定义可学习的全局知识库和相关的线性投影层 ---
            # 这个是全局共享的、可学习的“知识向量”
            self.motif_global_query = nn.Parameter(torch.FloatTensor(1, total_out_channels))
            
            # 线性层，用于从模体特征生成Query
            self.q_proj = nn.Linear(total_out_channels, total_out_channels, bias=False)
            
            # 线性层，用于从全局知识向量生成Key和Value
            self.kv_proj = nn.Linear(total_out_channels, total_out_channels * 2, bias=False)
            
            # --- 保留并复用：融合门控机制，现在它将融合原始模体特征和全局上下文 ---
            # 注意：这里应该是D_motif * 2，因为concat_features的维度是[D_motif * 2]
            self.motif_fusion_gate_linear = nn.Linear(total_out_channels * 2, total_out_channels)
            # 用于融合节点特征的层
            self.node_fusion_gate_linear = nn.Linear(total_out_channels * 2, total_out_channels)
        # self.pe_scale = nn.Parameter(torch.tensor([-2.94]))  #mark
        # self.pe_scale = nn.Parameter(torch.tensor([0.0]*heads))  #mark
        # self.pe_enhance_linear = nn.Linear(self.heads * 2, self.heads)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'lin'):
            nn.init.xavier_normal_(self.lin.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.lin_src.weight, gain=gain)
            nn.init.xavier_normal_(self.lin_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.att_src, gain=gain)
        nn.init.xavier_normal_(self.att_dst, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res, nn.Linear):
            nn.init.xavier_normal_(self.res.weight, gain=gain)
            nn.init.xavier_normal_(self.add_linear.weight, gain=gain)
            nn.init.constant_(self.add_linear.bias, -5.0)
        # nn.init.xavier_normal_(self.pe_enhance_linear.weight, gain=gain)
        # nn.init.constant_(self.pe_enhance_linear.bias, 0)
        # 初始化motif self-attention
        # if hasattr(self, 'motif_self_attention'):
        #     # 对MultiheadAttention的权重进行Xavier初始化
        #     for name, param in self.motif_self_attention.named_parameters():
        #         if 'weight' in name:
        #             nn.init.xavier_normal_(param, gain=gain)
        #         elif 'bias' in name:
        #             nn.init.constant_(param, 0)
        if self.use_motif == True:
            # 初始化全局知识向量
            nn.init.xavier_normal_(self.motif_global_query, gain=gain)
            nn.init.xavier_normal_(self.q_proj.weight, gain=gain)
            nn.init.xavier_normal_(self.kv_proj.weight, gain=gain)
            nn.init.xavier_normal_(self.motif_fusion_gate_linear.weight, gain=gain)
            nn.init.constant_(self.motif_fusion_gate_linear.bias, 0)
            nn.init.xavier_normal_(self.node_fusion_gate_linear.weight, gain=gain)
            nn.init.constant_(self.node_fusion_gate_linear.bias, 0)
        
    def forward(
        self,
        x,
        pe,
        edge_index,
        edge_attr = None,
        size = None,
        return_attention_weights = None,
        motif_tensor = None,
        return_gates = False,
    ):

        H, C = self.heads, self.out_channels
        res = None
        # 用于存储gate值
        gate_info = {}
        # 将return_gates存储为实例属性，以便edge_update可以访问
        self._return_gates = return_gates
        
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.res is not None:
                res = self.res(x)

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                raw_edge_index = edge_index
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    raw_edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
                # if pe is not None:
                #     edge_index_pe, pe = remove_self_loops(raw_edge_index, pe)
                #     _, pe = add_self_loops(edge_index_pe, pe, fill_value=1, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha, pe_out = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,
                                  size=size, pe=pe)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if motif_tensor is not None:
            # print('--------------------------------------------------')
            # 获取维度信息
            num_nodes = out.shape[0]
            if len(out.shape) == 3:  # (num_nodes, heads, out_channels)
                heads, h = out.shape[1], out.shape[2]
                out_reshaped = out.view(num_nodes, -1)  # (num_nodes, heads * out_channels)
            else:  # (num_nodes, out_channels)
                heads, h = 1, out.shape[1]
                out_reshaped = out
            
            num_motifs = motif_tensor.shape[0]
            
            # 提取motif对应的特征
            motif_features = out_reshaped[motif_tensor]  # (n, 3, heads*out_channels)
            
            # 使用平均池化 (Mean Pooling)
            out_motif = torch.mean(motif_features, dim=1)
            # fused_motif_features = self.motif_down_proj(out_motif)
            # # 对齐 dtype，避免 Float/Half 混用导致的 index_add_ 报错
            # if fused_motif_features.dtype != out.dtype:
            #     fused_motif_features = fused_motif_features.to(out.dtype)
            #====================================qkv====================================
             # 2. 计算全局上下文 (新的简化注意力机制)
            q = self.q_proj(out_motif)  # Query: 来自每个模体 [N, D]
            
            # Key & Value: 来自可学习的全局知识向量 [1, D]
            k, v = self.kv_proj(self.motif_global_query).chunk(2, dim=-1) # k,v都是[1, D]
            
            # 计算注意力得分并加权
            d_k = q.size(-1)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (d_k**0.5) # [N, 1]
            # 裁剪注意力分数，防止softmax时数值溢出
            attn_scores = torch.clamp(attn_scores, min=-50, max=50)
            attn_weights = F.softmax(attn_scores, dim=0) # 在N个模体上softmax
            
            # 每个模体得到的全局上下文 (理论上每行都接近V)
            motif_global_info = attn_weights * v # [N, D_motif]
            
            # 3. 融合原始模体信息与全局上下文 (复用之前的门控)
            # concat_features = torch.cat([out_motif, motif_global_info], dim=1)
            # alpha = torch.sigmoid(self.motif_fusion_gate_linear(concat_features))
            # fused_motif_features = alpha * out_motif + (1 - alpha) * motif_global_info
            fused_motif_features = motif_global_info
            #====================================qkv====================================
            # 4. 将融合后的模体信息聚合回节点 (复用之前的逻辑)
            num_motifs = motif_tensor.shape[0]

            # 初始化聚合张量
            aggregated_features = torch.zeros(num_nodes, heads * h, device=out.device, dtype=out.dtype)
            node_motif_count = torch.zeros(num_nodes, device=out.device, dtype=torch.float)

            # 将融合后的模体特征扩展到原始维度
            # fused_motif_features_expanded = self.motif_up_proj(fused_motif_features)  # [N, heads*out_channels]
            
            # 展平与扩展
            node_indices = motif_tensor.view(-1)  # (num_motifs * 3,)
            out_motif_expanded = fused_motif_features.repeat_interleave(3, dim=0)  # (num_motifs * 3, heads*out_channels)
            if out_motif_expanded.dtype != aggregated_features.dtype:
                out_motif_expanded = out_motif_expanded.to(aggregated_features.dtype)

            # 分散式累加
            aggregated_features.index_add_(0, node_indices, out_motif_expanded)
            node_motif_count.index_add_(0, node_indices, torch.ones_like(node_indices, dtype=torch.float))

            # 求平均 (处理除零问题)
            node_motif_count.clamp_(min=1)
            motif_info_agg = aggregated_features / node_motif_count.unsqueeze(1)
            # 2. 确保维度匹配
            if len(out.shape) == 3:  # (num_nodes, heads, h)
                out_flat = out.view(num_nodes, heads * h)  # (num_nodes, heads*h)
            else:  # (num_nodes, h)
                out_flat = out
                heads, h = 1, out.shape[1]
            

            # 3. 拼接两种特征
            concat_features = torch.cat([out_flat, motif_info_agg], dim=1)
            # 在送入线性层之前进行归一化
            norm_concat_features = self.fusion_norm(concat_features) 
            # 4. 计算门控权重 alpha (为每个节点独立计算)
            alpha = torch.sigmoid(self.node_fusion_gate_linear(norm_concat_features))
            
            # 保存gate值用于可视化
            if self._return_gates:
                gate_info['node_motif_fusion_gate'] = alpha.detach()

            # 5. 使用门控权重进行融合
            fused_features = alpha* out_flat + (1 - alpha) * motif_info_agg
            
            # 6. 恢复原始维度
            if len(out.shape) == 3:  # 需要reshape回去
                out = fused_features.view(num_nodes, heads, h)
            else:
                out = fused_features


        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            # 3. 拼接两种特征
            concat_features = torch.cat([out, res], dim=1)
            # 在送入线性层之前进行归一化
            norm_concat_features = self.fusion_norm_res(concat_features) 
            # 4. 计算门控权重 alpha (为每个节点独立计算)
            alpha = torch.sigmoid(self.add_linear(norm_concat_features))
            
            # 保存gate值用于可视化
            if self._return_gates:
                gate_info['residual_gate'] = alpha.detach()

            out = out + alpha*res

        if self.bias is not None:
            out = out + self.bias
        
        if return_gates:
            # 添加PE gate值到gate_info（从实例属性中获取）
            if hasattr(self, '_pe_gate_value') and self._pe_gate_value is not None:
                gate_info['pe_gate'] = torch.tensor(self._pe_gate_value, device=out.device)
                # 清除临时属性
                delattr(self, '_pe_gate_value')
            # 清除return_gates临时属性
            if hasattr(self, '_return_gates'):
                delattr(self, '_return_gates')
            # 返回输出、PE和gate信息
            return out, pe_out, gate_info
        
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if edge_index.is_sparse:
                    # 直接使用稀疏张量的值设置
                    edge_index = edge_index.coalesce()
                    edge_index._values = alpha
                    return out, (edge_index, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, pe_out

    def edge_update(self, alpha_j, alpha_i,
                    edge_attr, index, ptr,
                    dim_size, pe, gate):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        # gate = torch.sigmoid(self.pe_scale)
        # h_cat = torch.cat([alpha_i, alpha_j], dim=-1) 
        # gate = torch.sigmoid(self.pe_enhance_linear(h_cat))
        # 保存PE gate值（如果外层需要）
        # 注意：这个gate值在每个边上是相同的，所以我们可以只保存标量值
        # 使用实例属性来访问return_gates
        # pe_gate_value = gate.item() if hasattr(self, '_return_gates') and self._return_gates else None
        # alpha = alpha + gate * pe #(1 - gate)* 减号是错的，因为高斯核让他符合距离均值相应越大了。
        pe_out = pe
            
        alpha = F.leaky_relu(alpha, self.negative_slope)

        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.edge_dropout, training=self.training)

        # 将pe_gate_value存储到实例属性中，以便forward方法可以访问
        # if pe_gate_value is not None:
        #     if not hasattr(self, '_pe_gate_value'):
        #         self._pe_gate_value = None
        #     self._pe_gate_value = pe_gate_value
        
        return alpha, pe_out

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels},'
                f'{self.out_channels}, heads={self.heads})')