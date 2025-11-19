# 模型参数可视化说明

本工具用于可视化训练后的模型关键参数，帮助分析模型的有效性。

## 功能特性

### 1. Gate融合权重可视化
- **节点-模体融合Gate**: 显示每个节点在融合节点特征和模体特征时的gate权重
  - Gate值 > 0.5: 节点特征占主导
  - Gate值 < 0.5: 模体特征占主导
- **残差连接Gate**: 显示残差连接的控制权重

### 2. PE缩放因子可视化
- 显示每层的Position Encoding缩放因子
- 帮助理解模型如何平衡结构信息和位置信息

### 3. 全局知识向量可视化
- 可视化motif_global_query，展示模型学习的全局模体知识

### 4. 其他可分析参数
- 注意力权重统计
- 节点嵌入和模体嵌入的统计信息

## 使用方法

### 方法1: 使用训练脚本自动保存的checkpoint

训练脚本会在每个实验结束后自动保存模型checkpoint到`logdata/`目录。

```bash
python visualize_model_params.py \
    --checkpoint ./logdata/YYYYMMDD_HHMMSS<dataset><epochs><angle>exp0/model_checkpoint_exp0.pt \
    --dataset actor \
    --output_dir ./visualization_results/actor_exp0
```

### 方法2: 手动指定配置文件

```bash
python visualize_model_params.py \
    --checkpoint <checkpoint_path> \
    --dataset actor \
    --config ./config/actor.yaml \
    --output_dir ./visualization_results
```

## 输出内容

可视化脚本会生成以下文件：

1. **node_motif_fusion_gate.png**: 节点-模体融合gate的分布和统计
2. **residual_gate.png**: 残差连接gate的分布
3. **pe_scale.png**: PE缩放因子随层的变化
4. **global_query.png**: 全局知识向量的可视化
5. **gate_data.npz**: 所有gate数据的numpy存档（用于进一步分析）

## 可视化图表示例

### Gate分布直方图
显示gate值的分布，帮助理解模型在节点特征和模体特征之间的平衡。

### 每层平均Gate值
展示gate值在不同层的变化趋势，反映模型在不同深度如何利用不同信息源。

### PE缩放因子
展示模型如何调整位置编码的重要性。

## 论文分析建议

1. **Gate值分析**:
   - 如果gate值整体偏大（>0.5），说明模型更依赖节点特征
   - 如果gate值整体偏小（<0.5），说明模型更依赖模体特征
   - 不同层的gate值变化可以反映信息融合的层次性

2. **PE缩放因子分析**:
   - 较小的值（接近0）表示模型较少使用位置信息
   - 较大的值（接近1）表示位置信息很重要

3. **统计对比**:
   - 可以对比不同数据集上的gate值分布
   - 可以对比不同超参数设置下的gate值
   - 可以分析gate值与模型性能的关系

## 代码修改说明

为了支持gate值提取，以下文件已被修改：

1. **conv.py**: 
   - 添加了`return_gates`参数到`forward`方法
   - 在gate计算处保存gate值

2. **encoder.py**: 
   - 添加了`return_gates`参数支持
   - 将gate信息向上传递

3. **autoencoder.py**: 
   - 添加了`return_gates`参数支持

4. **train_node.py**: 
   - 添加了模型checkpoint保存功能

## 注意事项

1. 确保训练时使用的数据集和配置与可视化时一致
2. Gate值是在推理模式下提取的，反映训练后模型的最终状态
3. 如果使用大量三角形，可视化时可能会限制数量以提高速度

## 扩展功能建议

可以进一步添加：

1. **注意力权重可视化**: 展示模型关注的边和节点
2. **嵌入空间可视化**: 使用t-SNE或UMAP可视化节点嵌入
3. **模体重要性分析**: 分析不同类型模体的贡献
4. **训练过程gate变化**: 跟踪训练过程中gate值的变化趋势

