"""
Node-Motif Fusion Gate 可视化脚本
专门用于可视化节点-模体融合门控的分布
包括：
1. U 形分布的直方图
2. 每层的 Gate 值分布（Box plot）
每个图片单独保存，不带标题
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from argparse import ArgumentParser

from encoder import GraphEncoder
from autoencoder import GraphAutoEncoder
from utils import load_config, set_random_seed
from triangle_motif_manager import TriangleMotifManager
from sklearn.decomposition import TruncatedSVD


class NodeMotifFusionVisualizer:
    """Node-Motif Fusion Gate 可视化类"""
    
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
            result = self.model.embed(x, edge_index, u, edge_index_pe, triangles, return_gates=True)
            if isinstance(result, tuple):
                embed, all_gates = result
            else:
                all_gates = []
                embed = result
            
            # 组织gate信息
            node_fusion_gates = []
            
            for layer_idx, layer_gates in enumerate(all_gates):
                if 'node_motif_fusion_gate' in layer_gates:
                    node_fusion_gates.append(layer_gates['node_motif_fusion_gate'])
            
            gate_info = {
                'node_fusion_gates': node_fusion_gates,
                'all_layers': all_gates
            }
        
        return gate_info
    
    def visualize_u_shaped_histogram(self, gate_values, save_path='node_motif_fusion_histogram.pdf'):
        """
        可视化 U 形分布的直方图（保存为PDF矢量图）
        """
        if len(gate_values) == 0:
            print("警告：没有gate值可可视化")
            return
        
        # 合并所有层的gate值
        gate_tensor = torch.cat([v.flatten() for v in gate_values])
        gate_np = gate_tensor.cpu().numpy()
        
        # 创建直方图
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # 绘制直方图，使用更多bins以更好地显示U形分布
        n_bins = 50
        counts, bins, patches = ax.hist(gate_np, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
        
        # 设置坐标轴标签
        ax.set_xlabel('Gate Value', fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        
        # 添加均值和中位数的垂直线
        mean_val = gate_np.mean()
        median_val = np.median(gate_np)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        ax.axvline(0.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Neutral (0.5)')
        
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # 移除标题
        # ax.set_title('Distribution of Node-Motif Fusion Gates', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        # 保存为PDF矢量图格式
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"直方图已保存至: {save_path}")
        plt.close()
    
    def visualize_gate_distribution_per_layer(self, gate_values, save_path='node_motif_fusion_per_layer.pdf'):
        """
        可视化每层的 Gate 值分布（Box plot，保存为PDF矢量图）
        """
        if len(gate_values) == 0:
            print("警告：没有gate值可可视化")
            return
        
        # 准备数据
        data_for_box = [v.cpu().numpy().flatten() for v in gate_values]
        layer_labels = [f'Layer {i}' for i in range(len(gate_values))]
        
        # 创建箱线图
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(gate_values) * 1.5), 6))
        
        # 绘制箱线图
        bp = ax.boxplot(data_for_box, labels=layer_labels, patch_artist=True)
        
        # 设置箱线图颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(gate_values)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        # 设置中位线颜色为红色
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2.0)
        
        # 设置坐标轴标签
        ax.set_ylabel('Gate Value', fontsize=16)
        ax.set_xlabel('Layer Index', fontsize=16)
        
        # 添加中性线
        ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Neutral (0.5)')
        
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 移除标题
        # ax.set_title('Gate Value Distribution per Layer', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        # 保存为PDF矢量图格式
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"每层分布图已保存至: {save_path}")
        plt.close()


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
    visualizer = NodeMotifFusionVisualizer(model, device)
    
    # 组织输出子目录
    run_name = Path(checkpoint_path).parent.name if checkpoint_path else 'unknown_run'
    dataset_name = args.dataset if hasattr(args, 'dataset') else 'unknown_dataset'
    final_output_dir = os.path.join(output_dir, f"{run_name}")
    os.makedirs(final_output_dir, exist_ok=True)

    # 提取gate值
    print("\n正在提取gate权重...")
    gate_info = visualizer.extract_gate_weights(x, edge, PE, edge_index_pe, triangles, u)
    
    # 可视化节点-模体融合gate
    if gate_info['node_fusion_gates']:
        # 1. U 形分布直方图（PDF矢量图）
        visualizer.visualize_u_shaped_histogram(
            gate_info['node_fusion_gates'],
            save_path=os.path.join(final_output_dir, 'node_motif_fusion_histogram.pdf')
        )
        
        # 2. 每层的 Gate 值分布（PDF矢量图）
        visualizer.visualize_gate_distribution_per_layer(
            gate_info['node_fusion_gates'],
            save_path=os.path.join(final_output_dir, 'node_motif_fusion_per_layer.pdf')
        )
        
        # 打印统计信息
        print(f"\n节点-模体融合Gate统计:")
        for i, gate in enumerate(gate_info['node_fusion_gates']):
            gate_np = gate.cpu().numpy().flatten()
            print(f"  Layer {i}: mean={gate_np.mean():.4f}, std={gate_np.std():.4f}, "
                  f"min={gate_np.min():.4f}, max={gate_np.max():.4f}")
    else:
        print("警告：未找到节点-模体融合gate值")
    
    return visualizer, model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='./logdata/20251112_111843blog12000.01exp0encodermotifself/model_checkpoint_exp0.pt', 
                       help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='blog', help='数据集名称')
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
        print("python visualize_node_motif_fusion.py --checkpoint <path> --dataset <dataset_name>")

