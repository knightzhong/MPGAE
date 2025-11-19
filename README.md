# MPAE: Motif-Prototype Aware Graph Autoencoder

> **Paper**: *Decoupling Graph Autoencoders for Heterophily: A Motif-Prototype Aware Approach*


## Highlights
- **Objective Decoupling**: replace link prediction with spectral distance reconstruction to remove the ``connection-is-similarity" bias.
- **Motif-Prototype Attention**: inject non-local structural priors through motif participation matrices and learnable prototypes.
- **Plug-and-Play**: implemented on top of PyTorch Geometric with standard configs for 13 benchmarks.
- **Visualization Toolkit**: scripts for motif fusion analysis and parameter inspection are preserved under `visualize_*.py`.

## Project Layout
```
MPAE_release/
├── autoencoder.py              # GraphAutoEncoder definition
├── conv.py                     # Encoder building blocks
├── encoder.py                  # Motif-Prototype encoder
├── triangle_motif_manager.py   # Motif mining utilities
├── train_node.py               # Main training/evaluation script
├── utils.py / evaluation.py    # Helper functions
├── preprocess*.py              # Data preprocessing pipelines
├── config/                     # Dataset-specific hyperparameters
├── data/README.md              # Instructions for placing datasets
├── visualize_*.py              # Optional visualization utilities
├── requirements.txt
├── run_graphpae_nc.sh          # Example training launcher
└── README.md
```

## Quick Start
```bash
# 1. 安装依赖
conda create -n mpae python=3.10 -y
conda activate mpae
pip install -r requirements.txt

# 2. 准备数据 (详见 data/README.md)
#    将 PyG 格式的数据放入 data/<dataset_name>/

# 3. 运行训练 / 评估
python train_node.py --dataset cora 
```
- 所有超参数均由 `config/<dataset>.yaml` 控制，可直接编辑对应文件。
- 模体搜索相关选项位于 `triangle_motif_manager.py` 中，可根据图规模调整 `searchn` 与 `num_random`。

## Visualization & Analysis
- `visualize_model_params.py`: 追踪注意力权重、谱特征等训练曲线。
- `visualize_node_motif_fusion.py`: 展示节点在不同 motif prototype 上的分布。
- `VISUALIZATION_README.md`: 更详细的可视化配置说明。

## Reproducing Reported Results
1. 根据 `config/<dataset>.yaml` 设置 `epochs`, `lr`, `motif_num`, `spectral_weight` 等参数。
2. 运行 `bash run_graphpae_nc.sh` 以批量执行多数据集实验；脚本内包含示例命令。
3. 训练结束后将自动触发 `evaluation.py` 中的节点分类与谱重构指标统计。

## Citation
如果本项目对您的研究有帮助，请引用：
```

```
