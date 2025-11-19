"""
模型参数可视化脚本
用于提取和可视化模型的关键参数，用于论文分析
包括：
1. Gate融合权重（节点-模体融合gate）
2. 残差连接gate
3. PE缩放因子
4. 全局知识向量
5. 注意力权重统计
6. 节点嵌入和模体嵌入的统计信息
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path
import os
from argparse import ArgumentParser

from encoder import GraphEncoder
from autoencoder import GraphAutoEncoder
from utils import load_config, set_random_seed
from triangle_motif_manager import TriangleMotifManager
from sklearn.decomposition import TruncatedSVD


class ModelVisualizer:
    """模型参数可视化类"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def extract_gate_weights(self, x, edge_index, PE, edge_index_pe, triangles, u):
        """
        提取所有层的gate权重
        返回：dict包含各层的gate值
        """
        gate_info = {}
        
        with torch.no_grad():
            # 使用模型的embed方法，设置return_gates=True
            # 需要传入有效的 u 以计算 PE，避免在 embed 中对 None 进行索引
            result = self.model.embed(x, edge_index, u, edge_index_pe, triangles, return_gates=True)
            if isinstance(result, tuple):
                embed, all_gates = result
            else:
                all_gates = []
                embed = result
            
            # 组织gate信息
            node_fusion_gates = []
            residual_gates = []
            pe_gates = []
            
            for layer_idx, layer_gates in enumerate(all_gates):
                if 'node_motif_fusion_gate' in layer_gates:
                    node_fusion_gates.append(layer_gates['node_motif_fusion_gate'])
                if 'residual_gate' in layer_gates:
                    residual_gates.append(layer_gates['residual_gate'])
                if 'pe_gate' in layer_gates:
                    pe_gates.append(layer_gates['pe_gate'])
            
            gate_info = {
                'node_fusion_gates': node_fusion_gates,
                'residual_gates': residual_gates,
                'pe_gates': pe_gates,
                'all_layers': all_gates
            }
        
        return gate_info
    
    def extract_gate_values_during_forward(self, x, edge_index, PE, edge_index_pe, triangles):
        """
        通过hook机制在前向传播中提取gate值
        这是更好的方法，因为它可以捕获实际的gate值
        """
        gate_values = {
            'node_fusion_gates': [],  # 每层的节点-模体融合gate
            'residual_gates': [],     # 每层的残差连接gate
            'pe_scales': [],          # 每层的PE缩放因子
        }
        
        def register_gate_hooks():
            """注册hook来捕获gate值"""
            handles = []
            
            for layer_idx, gat_layer in enumerate(self.model.encoder.gnns):
                # Hook for node fusion gate (在forward中修改代码来返回gate)
                # 由于gate在forward内部计算，我们需要修改forward方法
                # 或者使用更巧妙的方式
                pass
            
            return handles
        
        # 修改conv.py的forward方法来返回gate值（需要修改源代码）
        # 或者我们创建一个包装器
        
        return gate_values
    
    def visualize_node_motif_fusion_gate(self, gate_values, save_path='gate_visualization.png'):
        """
        可视化节点-模体融合gate的分布
        """
        if len(gate_values) == 0:
            print("警告：没有gate值可可视化")
            return
        
        # 统计信息
        gate_tensor = torch.cat([v.flatten() for v in gate_values])
        gate_np = gate_tensor.cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 直方图
        axes[0, 0].hist(gate_np, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Gate Value', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Distribution of Node-Motif Fusion Gates', fontsize=14, fontweight='bold')
        axes[0, 0].axvline(gate_np.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gate_np.mean():.3f}')
        axes[0, 0].axvline(np.median(gate_np), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(gate_np):.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 每层的平均gate值
        layer_means = [v.mean().item() for v in gate_values]
        axes[0, 1].plot(range(len(layer_means)), layer_means, 'o-', linewidth=2, markersize=8, color='coral')
        axes[0, 1].set_xlabel('Layer Index', fontsize=12)
        axes[0, 1].set_ylabel('Mean Gate Value', fontsize=12)
        axes[0, 1].set_title('Mean Gate Value per Layer', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plot (如果有多个样本)
        if len(gate_values) > 1:
            data_for_box = [v.cpu().numpy().flatten() for v in gate_values]
            axes[1, 0].boxplot(data_for_box, labels=[f'Layer {i}' for i in range(len(gate_values))])
            axes[1, 0].set_ylabel('Gate Value', fontsize=12)
            axes[1, 0].set_title('Gate Value Distribution per Layer', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 统计表格
        stats_text = f"""
Statistics:
  Mean: {gate_np.mean():.4f}
  Std: {gate_np.std():.4f}
  Min: {gate_np.min():.4f}
  Max: {gate_np.max():.4f}
  Median: {np.median(gate_np):.4f}
  Q25: {np.percentile(gate_np, 25):.4f}
  Q75: {np.percentile(gate_np, 75):.4f}
  
Interpretation:
  Gate value > 0.5: Node features dominate
  Gate value < 0.5: Motif features dominate
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gate可视化已保存至: {save_path}")
        plt.close()
    
    def extract_pe_scale_factors(self):
        """
        提取所有层的PE缩放因子
        """
        pe_scales = {}
        
        for layer_idx, gat_layer in enumerate(self.model.encoder.gnns):
            if hasattr(gat_layer, 'pe_scale'):
                scale_param = gat_layer.pe_scale
                # pe_scale经过sigmoid后才是实际的gate值
                actual_gate = torch.sigmoid(scale_param).item()
                raw_value = scale_param.item()
                
                pe_scales[f'layer_{layer_idx}'] = {
                    'raw': raw_value,
                    'gate_value': actual_gate
                }
        
        return pe_scales
    
    def visualize_pe_scales(self, pe_scales, save_path='pe_scale_visualization.png'):
        """
        可视化PE缩放因子
        """
        if len(pe_scales) == 0:
            print("警告：没有PE缩放因子可可视化")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        layers = sorted([int(k.split('_')[1]) for k in pe_scales.keys()])
        raw_values = [pe_scales[f'layer_{l}']['raw'] for l in layers]
        gate_values = [pe_scales[f'layer_{l}']['gate_value'] for l in layers]
        
        # 原始值
        axes[0].plot(layers, raw_values, 'o-', linewidth=2, markersize=8, color='purple')
        axes[0].set_xlabel('Layer Index', fontsize=12)
        axes[0].set_ylabel('Raw PE Scale Parameter', fontsize=12)
        axes[0].set_title('PE Scale Parameter (Raw)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Gate值（经过sigmoid）
        axes[1].plot(layers, gate_values, 's-', linewidth=2, markersize=8, color='orange')
        axes[1].axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (0.5)')
        axes[1].set_xlabel('Layer Index', fontsize=12)
        axes[1].set_ylabel('PE Gate Value (sigmoid)', fontsize=12)
        axes[1].set_title('PE Scale Gate Value', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PE缩放因子可视化已保存至: {save_path}")
        plt.close()
    
    def extract_motif_global_queries(self):
        """
        提取所有层的全局知识向量（motif_global_query）
        """
        global_queries = {}
        
        for layer_idx, gat_layer in enumerate(self.model.encoder.gnns):
            if hasattr(gat_layer, 'motif_global_query'):
                query = gat_layer.motif_global_query.detach().cpu().numpy()
                global_queries[f'layer_{layer_idx}'] = query
        
        return global_queries
    
    def visualize_global_queries(self, global_queries, save_path='global_query_visualization.png'):
        """
        可视化全局知识向量
        """
        if len(global_queries) == 0:
            print("警告：没有全局知识向量可可视化")
            return
        
        num_layers = len(global_queries)
        fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers))
        
        if num_layers == 1:
            axes = [axes]
        
        for idx, (layer_name, query) in enumerate(global_queries.items()):
            query_flat = query.flatten()
            
            axes[idx].plot(query_flat, linewidth=1.5, alpha=0.7)
            axes[idx].fill_between(range(len(query_flat)), query_flat, alpha=0.3)
            axes[idx].set_xlabel('Dimension Index', fontsize=11)
            axes[idx].set_ylabel('Value', fontsize=11)
            axes[idx].set_title(f'Global Knowledge Vector - {layer_name}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"全局知识向量可视化已保存至: {save_path}")
        plt.close()


def modify_conv_to_return_gates():
    """
    这个函数说明了如何在conv.py中修改forward方法来返回gate值
    实际使用时需要在conv.py中修改forward方法
    """
    pass


def load_model_and_visualize(checkpoint_path, data, config_path, output_dir='./visualization_results'):
    """
    加载模型并进行可视化
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载配置
    from utils import load_config
    config = load_config(config_path)
    args = type('Args', (), config)()
    
    # 设置设备
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # 处理特征维度
    x = data.x.float().to(device)
    if x.shape[1] > 4096:
        svd = TruncatedSVD(n_components=4096, random_state=0)
        x_cpu = x.detach().cpu().numpy()
        x_reduced = svd.fit_transform(x_cpu)
        x = torch.from_numpy(x_reduced).float().to(device)
    
    args.feat_dim = x.shape[1]
    args.num_node = x.shape[0]
    
    # 准备数据
    edge = data.edge_index.long().to(device)
    u = data.u[:, :args.max_freqs].float().to(device)
    e = data.e[:args.max_freqs].float().to(device)
    
    from torch_geometric.utils import remove_self_loops, add_self_loops
    edge_index_pe, _ = remove_self_loops(edge, None)
    edge_index_pe, _ = add_self_loops(edge_index_pe, fill_value='mean', num_nodes=u.shape[0])
    PE = torch.linalg.norm(u[edge_index_pe[0]] - u[edge_index_pe[1]], dim=-1)
    
    # 加载三角形模体
    from triangle_motif_manager import TriangleMotifManager
    triangle_manager = TriangleMotifManager(edge, x.shape[0], device)
    triangles_all = triangle_manager.get_triangle_tensor()
    triangles = triangles_all[:min(1000, len(triangles_all))].to(device)  # 限制数量用于可视化
    
    # 重建模型
    from encoder import GraphEncoder
    from autoencoder import GraphAutoEncoder
    encoder = GraphEncoder(out_dim=args.embed_dim, args=args).to(device)
    model = GraphAutoEncoder(encoder=encoder, num_atom_type=args.feat_dim, args=args).to(device)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 如果没有model_state_dict，可能是直接保存的状态字典
        model.load_state_dict(checkpoint)
    print(f"模型已从 {checkpoint_path} 加载")
    
    # 创建可视化器
    visualizer = ModelVisualizer(model, device)
    
    # 组织输出子目录，避免覆盖：使用 数据集名 + checkpoint所在目录名
    run_name = Path(checkpoint_path).parent.name if checkpoint_path else 'unknown_run'
    dataset_name = args.dataset if hasattr(args, 'dataset') else 'unknown_dataset'
    final_output_dir = os.path.join(output_dir, f"{run_name}")
    os.makedirs(final_output_dir, exist_ok=True)

    # 提取PE缩放因子
    pe_scales = visualizer.extract_pe_scale_factors()
    print(f"\nPE缩放因子:")
    for layer, values in pe_scales.items():
        print(f"  {layer}: raw={values['raw']:.4f}, gate={values['gate_value']:.4f}")
    
    # 可视化PE缩放因子
    if pe_scales:
        visualizer.visualize_pe_scales(pe_scales, 
                                     save_path=os.path.join(final_output_dir, 'pe_scale.png'))
    
    # 提取全局知识向量
    global_queries = visualizer.extract_motif_global_queries()
    print(f"\n找到 {len(global_queries)} 层的全局知识向量")
    
    # 可视化全局知识向量
    if global_queries:
        visualizer.visualize_global_queries(global_queries,
                                           save_path=os.path.join(final_output_dir, 'global_query.png'))
    
    # 提取gate值
    print("\n正在提取gate权重...")
    gate_info = visualizer.extract_gate_weights(x, edge, PE, edge_index_pe, triangles, u)
    
    # 可视化节点-模体融合gate
    if gate_info['node_fusion_gates']:
        visualizer.visualize_node_motif_fusion_gate(
            gate_info['node_fusion_gates'],
            save_path=os.path.join(final_output_dir, 'node_motif_fusion_gate.png')
        )
        print(f"\n节点-模体融合Gate统计:")
        for i, gate in enumerate(gate_info['node_fusion_gates']):
            gate_np = gate.cpu().numpy().flatten()
            print(f"  Layer {i}: mean={gate_np.mean():.4f}, std={gate_np.std():.4f}, "
                  f"min={gate_np.min():.4f}, max={gate_np.max():.4f}")
    
    # 可视化残差gate
    if gate_info['residual_gates']:
        visualizer.visualize_node_motif_fusion_gate(
            gate_info['residual_gates'],
            save_path=os.path.join(final_output_dir, 'residual_gate.png')
        )
        print(f"\n残差连接Gate统计:")
        for i, gate in enumerate(gate_info['residual_gates']):
            gate_np = gate.cpu().numpy().flatten()
            print(f"  Layer {i}: mean={gate_np.mean():.4f}, std={gate_np.std():.4f}")
    
    # 保存gate数据为numpy文件
    if gate_info['node_fusion_gates'] or gate_info['residual_gates']:
        gate_data = {
            'node_fusion_gates': [g.cpu().numpy() for g in gate_info['node_fusion_gates']],
            'residual_gates': [g.cpu().numpy() for g in gate_info['residual_gates']],
            'pe_scales': pe_scales,
        }
        np.savez(os.path.join(final_output_dir, 'gate_data.npz'), **gate_data)
        print(f"\nGate数据已保存至: {os.path.join(final_output_dir, 'gate_data.npz')}")
    
    return visualizer, model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./logdata/20251105_175422flickr12000.01exp0encoderUhat11/model_checkpoint_exp0.pt', help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='flickr', help='数据集名称')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./visualization_results', help='输出目录')
    args = parser.parse_args()
    
    # 加载数据
    data = torch.load(f'../dataset/{args.dataset}.pt')
    
    # 确定配置文件
    config_path = args.config or f'./config/{args.dataset}.yaml'
    
    if args.checkpoint:
        load_model_and_visualize(args.checkpoint, data, config_path, args.output_dir)
    else:
        print("请提供checkpoint路径。使用方式:")
        print("python visualize_model_params.py --checkpoint <path> --dataset <dataset_name>")

