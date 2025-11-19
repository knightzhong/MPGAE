from argparse import ArgumentParser
import numpy as np

import torch

from utils import set_random_seed, split, load_config, create_optimizer
from evaluation import node_evaluation

from encoder import GraphEncoder
from autoencoder import GraphAutoEncoder
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)
import scipy.sparse as sp
from collections import defaultdict
from itertools import combinations, chain
import torch
import torch.nn as nn
import torch.nn.functional as F
# import scipy.sparse as sp # 在此代码段中未直接使用
import numpy as np
from sklearn.decomposition import TruncatedSVD
import igraph as ig
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import tqdm
from triangle_motif_manager import TriangleMotifManager, TriangleAngleLoss, TriangleMotifLoss, sample_triangle_batch, TriangleDataset
from torch.utils.data import Dataset,DataLoader
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import os
from torch.cuda.amp import autocast, GradScaler
import gc


# 2. 计算显著模体
def compute_significant_motifs(G, searchn=3, num_random=10, z_threshold=2.0):
    from collections import defaultdict
    import numpy as np
    print('searchn',searchn)
    # 1. 定义关心的 motif ID 和名称W
    if searchn == 3:
        # igraph: 0=空, 1=单边, 2=链, 3=三角形
        motif_id_name = {
            2: 'U_Path',      # 三节点链
            3: 'U_Triangle'   # 三角形
        }
    elif searchn == 4:
        # 
        motif_id_name = {
            
            # 3条边
            4: 'U4_Star',          # ID 4: 星型 (3条边)
            5: 'U4_Path_P4',       # ID 5: 链型 (3条边)
            
            # 4条边
            7: 'U4_TailedTriangle',# ID 7: 带尾三角形/爪子图 (4条边)
            8: 'U4_Cycle_C4',      # ID 8: 环型/正方形 (4条边)
            
            # 5条边
            9: 'U4_Diamond',       # ID 9: 菱形图 (5条边)
            
            # 6条边
            10: 'U4_Clique_K4'      # ID 10: 完全图 (6条边)
        }
    else:
        raise ValueError("只支持3或4节点模体")

    # 2. 真实图 motif 计数
    if searchn == 3:
        cut_prob = [0.5, 0.5, 0.5]
    elif searchn == 4:
        cut_prob = [0.5, 0.5, 0.5, 0.5]
    else:
        raise ValueError("只支持3或4节点模体")
    motif_counts_real = G.motifs_randesu(size=searchn,cut_prob=cut_prob)
    print('motif_counts_real',motif_counts_real)
    if motif_counts_real is None:
        print("错误: G.motifs_randesu 为真实图返回了 None。无法计算 Z-scores。")
        return []

    # 3. 随机图 motif 计数
    random_counts_collection = defaultdict(list)
    degrees = G.degree()
    import tqdm
    print(f"为 {num_random} 个随机网络计算模体计数 (无向)...")
    for i_random in tqdm.tqdm(range(num_random)):
        rand_G = ig.Graph.Degree_Sequence(degrees, method="configuration")
        # if searchn == 3:
        #     cut_prob = [0.01, 0.01, 0.01]
        # elif searchn == 4:
        #     cut_prob = [0.01, 0.01, 0.01, 0.01]
        current_rand_counts = rand_G.motifs_randesu(size=searchn,cut_prob=cut_prob)
        if current_rand_counts:
            for motif_id in motif_id_name:
                count = current_rand_counts[motif_id] if motif_id < len(current_rand_counts) else 0
                random_counts_collection[motif_id].append(count)

    # 4. 计算 Z-score
    significant_motifs = []
    for motif_id, motif_name in motif_id_name.items():
        real_count_val = motif_counts_real[motif_id] if motif_id < len(motif_counts_real) else 0
        if real_count_val == 0:
            continue
        counts_for_motif_in_random = random_counts_collection.get(motif_id, [])
        if not counts_for_motif_in_random:
            continue
        rand_mean = np.mean(counts_for_motif_in_random)
        rand_std = np.std(counts_for_motif_in_random)
        if rand_std > 0:
            z_score = (real_count_val - rand_mean) / rand_std
            if z_score > z_threshold:
                significant_motifs.append((motif_name, real_count_val, z_score, motif_id,searchn))
        elif real_count_val > rand_mean:
            significant_motifs.append((motif_name, real_count_val, float('inf'), motif_id,searchn))

    return significant_motifs
def load_igraph(data):
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes
    edges = list(zip(edge_index[0], edge_index[1]))
    G = ig.Graph(directed=False)
    G.add_vertices(num_nodes)
    G.add_edges(edges)
    G.simplify()
    return G
def compute_motif_link_matrix(G, target_motif_id_igraph,searchn=3):
    n = G.vcount()
    # 使用字典来存储边权重，避免频繁的矩阵更新
    edge_weights = defaultdict(int)
    
    def _motif_link_callback(subgraph_obj, node_list_igraph, motif_id_found_igraph):
        if motif_id_found_igraph == target_motif_id_igraph:
            nodes = list(node_list_igraph)
            # 使用itertools.combinations来生成节点对
            for i, j in combinations(nodes, 2):
                edge_weights[(i, j)] += 1
                edge_weights[(j, i)] += 1
        return None 
    if searchn == 3:    
        cut_prob = [0.1, 0.1, 0.1]
    elif searchn == 4:
        cut_prob = [0.3, 0.3, 0.3, 0.3]
    else:
        raise ValueError("只支持3或4节点模体")
    # 计算模体
    G.motifs_randesu(size=searchn,cut_prob = cut_prob, callback=_motif_link_callback)
    
    # 一次性构建稀疏矩阵
    if edge_weights:
        rows, cols = zip(*edge_weights.keys())
        data = list(edge_weights.values())
        link_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        link_matrix = sp.csr_matrix((n, n))
    
    return link_matrix#min_max_normalize_sparse_matrix(link_matrix)
def build_motif_participation_matrix(G, motif_ids=[2,3,4,5,6, 7, 8, 9], searchn_list=[3,3,4,4,4, 4, 4, 4]):
    """
    G: igraph 图
    motif_ids: igraph 中的 motif 类型 id（如三角形是3）
    searchn_list: 每个 motif_id 对应的 motif 大小
    返回:
        M: numpy array (N, K)
    """
    N = G.vcount()
    # motif_nodetype_len = sum(searchn_list)
    # motif_nodetype_len_dic = {}
    # for i in range(len(motif_ids)):
    #     motif_nodetype_len_dic[motif_ids[i]] = sum(searchn_list[:i])
    M = np.zeros((N), dtype=np.float32)

    for motif_idx, (motif_id, size) in tqdm.tqdm(enumerate(zip(motif_ids, searchn_list)),total=len(motif_ids)):
        # igraph的motifs_randesu函数在较新版本中可能不接受callback，
        # 这里的示例代码遵循了旧版API的思想。
        # 如果使用新版igraph，可能需要用motifs()后手动统计。
        # 此处我们假设它能按预期工作或用其他方式得到了M矩阵。
        
        # 为了让代码可运行，此处使用一个简化的模拟计数过程
        # 在实际应用中，您应该使用真实的motif计数函数
        try:
            # 这是一个简化的示例回调，实际的igraph API可能有所不同
            def _callback(subgraph_obj, node_list_igraph, motif_id_found_igraph):
                if motif_id_found_igraph == motif_id:
                    # for v in node_list_igraph:
                    for index, v in enumerate(node_list_igraph):
                        M[v] += 1
                return None
            # cut_prob = [0.1] * size # cut_prob是可选参数，用于近似计算
            G.motifs_randesu(size=size, callback=_callback)#, cut_prob=cut_prob)
        except Exception as e:
            # 如果 `motifs_randesu` with callback 不可用，我们用一个随机方法模拟M
            print(f"Warning: igraph motif counting failed with '{e}'. Using random M for demonstration.")
            # num_motifs_found = N * 2 # 模拟找到的模体数量
            # nodes_participating = np.random.randint(0, N, size=num_motifs_found)
            # for node_idx in nodes_participating:
            #     M[node_idx, motif_idx] += 1
                
    return M



def train_mae_epoch(graph_auto_encoder, x, edge_index, edge_index_pe, u, PE, optimizer, 
                   triangle_manager=None, triangle_angle_loss=None, triangle_motif_loss=None,triangle_counts_tensor=None,data_iterator=None,n_iter=None,argst=None,writer=None,scaler=None):
    graph_auto_encoder.train()
    triangles, negative_triplets = next(data_iterator)
    device = x.device # 获取模型/数据的设备
    triangles = triangles.to(device, non_blocking=True)
    # triangles = None
    # negative_triplets = negative_triplets.to(device, non_blocking=True)
    # 主损失（开启混合精度，已配合 GradScaler 使用）
    # with autocast(enabled=True, dtype=torch.float16):
    main_loss,u_for_angle_loss = graph_auto_encoder(x, edge_index, u, PE, edge_index_pe, triangles=triangles)
    # u_for_angle_loss = graph_auto_encoder.encoder.embed(x, edge_index,PE)[0]
    # 三角形模体损失
    
    # # node_embeddings = graph_auto_encoder.U_hat
    # angle_loss = triangle_angle_loss(u_for_angle_loss, triangles,u)
    # motif_loss = triangle_motif_loss(u_for_angle_loss, triangles,negative_triplets)
    angle_loss = torch.tensor(0.0, device=main_loss.device)
    # # 检查3: 偏离后，angle_loss 的原始值是多少？
    # print(f"[Check 3] RAW angle_loss value: {angle_loss.item():.8f}")
    # if triangle_manager is not None and triangle_manager.get_triangle_count() > 0:
    #     # 采样三角形批次
    #     batch_size = 64
    #     triangles, negative_triplets, labels = sample_triangle_batch(triangle_manager, batch_size)
        
    #     if len(triangles) >0:
    #         # 获取节点嵌入
    #         # with torch.no_grad():
    #         node_embeddings = graph_auto_encoder.U_hat#graph_auto_encoder.encoder.embed(x, edge_index, PE=PE)[1]
            
    #         # 角度重构损失
    #         if triangle_angle_loss is not None:
    #             angle_loss = triangle_angle_loss(node_embeddings, triangles,u)
    #             triangle_loss += 0.1 * angle_loss
            
    #         # 模体预测损失
    #         # if triangle_motif_loss is not None:
    #         #     motif_loss = triangle_motif_loss(node_embeddings, triangles, negative_triplets)
    #         #     triangle_loss += 0.1 * motif_loss
    #         motif_loss = torch.tensor(0.0, device=main_loss.device)
    motif_loss = torch.tensor(0.0, device=main_loss.device)#triangle_motif_loss(node_embeddings, triangles, negative_triplets)
    triangle_loss = torch.tensor(0.0, device=main_loss.device)
    # triangle_loss +=  0.1 * motif_loss
    # triangle_loss += argst.angle * angle_loss
    # 总损失
    total_loss = main_loss #+ triangle_loss
    
    # 检查loss是否为NaN，如果是则跳过该step（在打印和记录之前检查）
    if torch.isnan(main_loss) or torch.isnan(total_loss):
        print(f"⚠️ Warning: NaN detected at epoch {n_iter}, skipping this step")
        return float('nan')
    
    writer.add_scalar('Loss/main_loss', main_loss.item(), n_iter)
    # writer.add_scalar('Loss/triangle_loss', triangle_loss.item(), n_iter)
    # writer.add_scalar('Loss/angle_loss', angle_loss.item(), n_iter)
    # writer.add_scalar('Loss/motif_loss', motif_loss.item(), n_iter)
    writer.add_scalar('Loss/total_loss', total_loss.item(), n_iter)
    
    print(f"main_loss: {main_loss.item():.4f}, total_loss: {total_loss.item():.4f}", end='\t')
    
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    
    # 梯度裁剪防止梯度爆炸 - 必须在backward之后、step之前执行
    scaler.unscale_(optimizer)  # 在混合精度训练中，需要先unscale才能裁剪梯度
    torch.nn.utils.clip_grad_norm_(graph_auto_encoder.parameters(), max_norm=1.0)
    
    # 检查梯度是否包含NaN
    has_nan_grad = False
    for param in graph_auto_encoder.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            has_nan_grad = True
            break
    
    if has_nan_grad:
        print(f"⚠️ Warning: NaN gradient detected at epoch {n_iter}, skipping this step")
        scaler.update()  # 即使跳过step，也要更新scaler
        return float('nan')
    
    scaler.step(optimizer)
    scaler.update()
    
    
    return total_loss.item()  # 返回loss值供外部监控
import random
def worker_init_fn(worker_id):
    """
    为每个 DataLoader worker 设置独立的随机种子。
    这对于可复现的数据加载和增强至关重要。
    """
    # 获取在主进程中设置的全局种子
    # 注意：这里我们不能直接用 ep_num，因为这个函数是在 DataLoader 内部调用的
    # 一个常用的方法是使用 torch.initial_seed()，它会返回主进程为当前 worker 设置的初始种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed) # <--- 添加这一行

if __name__ == '__main__':

    # 必须将这行代码放在 __main__ 的最开始！
    try:
        mp.set_start_method('spawn', force=True)
        print("Spawn Cuda Process")
    except RuntimeError:
        pass
    
    parser = ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--root', type=str, default="./dataset")
    parser.add_argument('--dataset', type=str, default="cora")  # [pubmed citeseer minesweeper"blog", "chameleon", "squirrel", "actor","cornell","wisconsin","deezereurope","flickr","texas"]
    args = parser.parse_args()
    
    config = load_config(f"./config/{args.dataset}.yaml")
    for key, value in config.items():
        setattr(args, key, value)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # for motif_num in [1024 ,2048,4096,8192]: #1024 ,2048,4096,
        
    #     args.epochs = 1200
    #     if args.dataset == "chameleon":
    #         args.angle = 0.1
    #         args.epochs = 1000
    #     else:
    #         args.angle = 0.01
    #     if args.dataset == "actor":
    #         args.angle = 1.0
    #         args.epochs = 1200
    #     elif args.dataset == "citeseer":#or args.dataset == "pubmed":
    #         args.epochs = 500
    #     elif args.dataset == "cora":
    #         args.epochs = 300
    #     elif args.dataset == "texas":
    #         args.epochs = 800
    #     elif args.dataset == "citeseer":
    #         args.epochs = 1000
    #         # args.masked_pe_loss= 0.001
    #     # if args.dataset == "cornell":
    #     #     args.epochs = 1000
    #     #     args.angle = 1.0
    #     # args.angle = 0.01

    #     print(args)
        
    #     data = torch.load('../dataset/{}.pt'.format(args.dataset))
    #     print(data)
    #     x = data.x.float().to(args.device)
    #     if x.shape[1] > 4096:
    #         svd = TruncatedSVD(n_components=4096, random_state=0)
    #         x_cpu = x.detach().cpu().numpy()
    #         x_reduced = svd.fit_transform(x_cpu)
    #         x = torch.from_numpy(x_reduced).float().to(args.device)
    #         args.feat_dim = x.shape[1]
        
    #     # 在创建模型之前添加
    #     print('实际的特征维度')
    #     print(x.shape)
    #     args.feat_dim = x.shape[1]  # 使用实际数据的特征维度
    #     args.num_node = x.shape[0]
    #     edge = data.edge_index.long().to(args.device)
    #     e = data.e[:args.max_freqs].float().to(args.device)
    #     u = data.u[:, :args.max_freqs].float().to(args.device)

    #     y = data.y.to(args.device)
    #     print(y.min().item(), y.max().item())
    #     nclass = y.max().item() + 1

    #     edge_index_pe, _ = remove_self_loops(edge, None)
    #     edge_index_pe, _ = add_self_loops(edge_index_pe, fill_value='mean', num_nodes=u.shape[0])
    #     PE = torch.linalg.norm(u[edge_index_pe[0]] - u[edge_index_pe[1]], dim=-1)  # [e_sum, 1]

    #     # 在数据加载后添加三角形模体管理器初始化
    #     print("🔍 初始化三角形模体管理器...")
    #     triangle_manager = TriangleMotifManager(data.edge_index.long(), x.shape[0], args.device)
    #     print(f"✅ 找到 {triangle_manager.get_triangle_count()} 个三角形")
    #     # ### ===========================尝试用新的思路===========================
    #     # edge_index_np = edge.cpu().numpy()
    #     # edges = list(zip(edge_index_np[0], edge_index_np[1]))

    #     # G = ig.Graph(directed=False)
    #     # G.add_vertices(x.shape[0])
    #     # G.add_edges(edges)
    #     # G.simplify()
    #     # # 3. 调用核心函数 count_triangles()
    #     # # 这个函数返回一个列表，列表的长度是节点数，值是每个节点参与的三角形数
    #     # triangle_counts_list = build_motif_participation_matrix(G,[3],[3])
        
    #     # # 4. 将结果转换为 PyTorch Tensor
    #     # triangle_counts_tensor = torch.tensor(triangle_counts_list, dtype=torch.float)
        
    #     # if args.dataset == "actor":
    #     #     batch_size = 7121
    #     # elif args.dataset == "cornell":
    #     #     batch_size = 59
    #     if triangle_manager.get_triangle_count() < motif_num:
    #         batch_size = triangle_manager.get_triangle_count()
    #     else:
    #         batch_size = motif_num
        
        
    #     import time
    #     import datetime
    #     nowtime_step = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     time1 = time.time()
    #     args.num_exp = 10
    #     final_results = []
    #     for ep_num in range(0,args.num_exp):
    #         scaler = GradScaler() # 在训练循环外初始化
    #         path_write = './logdata/{}{}{}{}exp{}encoderUhat11'.format(nowtime_step,args.dataset,args.epochs,args.angle,ep_num)
    #         os.makedirs(path_write, exist_ok=True)
    #         writer = SummaryWriter(path_write)
    #         # 1. 创建迭代器对象
            
    #         args.seed = ep_num
    #         set_random_seed(ep_num)
    #         dataset = TriangleDataset(triangle_manager, epoch_size=args.epochs+1, batch_size=batch_size, negative_ratio=2.0)
    #         if args.dataset == "flickr":
    #             dataloader = DataLoader(
    #                 dataset,
    #                 batch_size=None,  # 因为我们的 __getitem__ 已经返回了批次
    #                 num_workers=1,    # 使用4个子进程在后台加载数据，这个值可以根据你的CPU核数调整
    #                 pin_memory=True,   # 如果使用GPU，可以加速数据从CPU到GPU的传输
    #                 worker_init_fn=worker_init_fn,
    #                 # persistent_workers=True, # 建议保留这些优化参数
    #                 # prefetch_factor=4
    #             )
    #         else:
    #             dataloader = DataLoader(
    #                 dataset,
    #                 batch_size=None,  # 因为我们的 __getitem__ 已经返回了批次
    #                 num_workers=8,    # 使用4个子进程在后台加载数据，这个值可以根据你的CPU核数调整
    #                 pin_memory=True,   # 如果使用GPU，可以加速数据从CPU到GPU的传输
    #                 worker_init_fn=worker_init_fn,
    #                 # persistent_workers=True, # 建议保留这些优化参数
    #                 # prefetch_factor=4
    #             )
    #         data_iterator = iter(dataloader)
    #         print('Checking data attributes')
    #         if hasattr(data, 'train_mask'):
    #             if len(data.train_mask.size()) > 1:
    #                 train_idx = torch.where(data.train_mask[:, args.seed])[0]
    #                 val_idx = torch.where(data.val_mask[:, args.seed])[0]
    #                 test_idx = torch.where(data.test_mask[:, args.seed])[0]
    #             else:
    #                 train_idx = torch.where(data.train_mask)[0]
    #                 val_idx = torch.where(data.val_mask)[0]
    #                 test_idx = torch.where(data.test_mask)[0]
    #         else:
    #             train_idx, val_idx, test_idx = split(y)

    #         # 创建三角形相关的损失函数
    #         triangle_angle_loss = TriangleAngleLoss().to(args.device)
    #         triangle_motif_loss = TriangleMotifLoss(u.shape[1]).to(args.device)
    #         encoder = GraphEncoder(out_dim=args.embed_dim, args=args).to(args.device)
    #         model = GraphAutoEncoder(encoder=encoder, num_atom_type=args.feat_dim, args=args).to(args.device)

    #         parameters = model.parameters()#chain(model.parameters(), triangle_motif_loss.parameters())
    #         # --- 请修改成下面这样 ---

    #         # 1. 首先，把模型的参数分成两组

    #         # 第1组：只包含 U_hat。我们为它专门指定一个高学习率。
    #         # 这里的 '* 100' 是一个例子，您可以根据需要调整这个系数 (比如 50, 200)。
    #         # u_hat_param_group = {
    #         #     'params': model.U_hat, 
    #         #     'lr': args.init_lr * 100  
    #         # }

    #         # # 第2组：包含模型中除了 U_hat 以外的所有其他参数。
    #         # # 我们不在这里为它指定 'lr'，这样它就会自动使用函数调用时的默认学习率。
    #         # other_params_group = {
    #         #     'params': [p for n, p in model.named_parameters() if 'U_hat' not in n]
    #         # }

    #         # # 2. 将这两组参数打包成一个列表
    #         # # 这就是我们要传递给优化器的新 `parameters`
    #         # parameters = [u_hat_param_group, other_params_group]

    #         if args.optim == "sgd":
    #             pass
    #         else:
    #             args.momentum = None
    #         optimizer = create_optimizer(opt=args.optim, parameters=parameters, lr=args.init_lr, weight_decay=float(args.weight_decay), momentum=args.momentum)

    #         if args.use_schedule:
    #             scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
    #             scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    #         else:
    #             scheduler = None

    #         for epoch in range(1, args.epochs+1):
    #             train_mae_epoch(graph_auto_encoder=model, x=x, edge_index=edge, u=u, PE=PE, edge_index_pe=edge_index_pe, optimizer=optimizer, 
    #                     triangle_manager=triangle_manager, triangle_angle_loss=triangle_angle_loss, triangle_motif_loss=triangle_motif_loss,data_iterator=data_iterator,n_iter=epoch,argst=args,writer=writer,scaler=scaler)
    #             print('epoch: ',epoch)
    #             # 在每个epoch结束后添加
    #             # if epoch % 100 == 0:  # 每100个epoch检查一次
    #             #     with torch.no_grad():
    #             #         u_hat_norm = torch.norm(model.U_hat, dim=1).mean().item()
    #             #         u_hat_deviation = torch.mean((model.U_hat - torch.eye(model.U_hat.size(0), device=model.U_hat.device)).abs()).item()
    #             #         print(f"[Epoch {epoch}] U_hat norm: {u_hat_norm:.4f}, deviation: {u_hat_deviation:.6f}")
    #             if scheduler:
    #                 scheduler.step()
    #         time2 = time.time()
    #         print('consume time {}'.format(time2-time1))
    #         # print(model.weight_motif)
    #         model.eval()
    #         triangles, negative_triplets = next(data_iterator)
    #         device = x.device # 获取模型/数据的设备
    #         triangles = triangles.to(device, non_blocking=True)
    #         # triangles = None
    #         with torch.no_grad():
    #             embed = model.embed(x, edge, u, edge_index_pe, triangles=triangles)
    #         acc, pred = node_evaluation(emb=embed, y=y, train_idx=train_idx, valid_idx=val_idx, test_idx=test_idx, epochs=args.epochs_eval, lr=args.lr_eval, weight_decay=args.wd_eval)
    #         print(f"Epoch {epoch}, ACC: {acc.item()}")
    #         writer.add_scalar('EXP/ACC', acc.item(), ep_num)
    #         final_results.append(acc.item())
            
    #         # 保存模型checkpoint（用于后续可视化）
    #         # checkpoint_path = os.path.join(path_write, f'model_checkpoint_exp{ep_num}.pt')
    #         # torch.save({
    #         #     'model_state_dict': model.state_dict(),
    #         #     'encoder_state_dict': encoder.state_dict(),
    #         #     'acc': acc.item(),
    #         #     'epoch': epoch,
    #         #     'args': args,
    #         # }, checkpoint_path)
    #         # print(f"模型已保存至: {checkpoint_path}")

    #         # ==== 资源清理，避免多次实验累积显存占用 ====
    #         try:
    #             writer.close()
    #         except Exception:
    #             pass
    #         # 删除迭代器/数据加载器/数据集
    #         try:
    #             del data_iterator
    #         except Exception:
    #             pass
    #         try:
    #             del dataloader
    #         except Exception:
    #             pass
    #         try:
    #             del dataset
    #         except Exception:
    #             pass
    #         # 删除模型与优化相关对象
    #         try:
    #             del model
    #         except Exception:
    #             pass
    #         try:
    #             del encoder
    #         except Exception:
    #             pass
    #         try:
    #             del optimizer
    #         except Exception:
    #             pass
    #         try:
    #             del scheduler
    #         except Exception:
    #             pass
    #         try:
    #             del triangles
    #         except Exception:
    #             pass
    #         try:
    #             del triangle_motif_loss
    #         except Exception:
    #             pass
    #         try:
    #             del scaler
    #         except Exception:
    #             pass
    #         # 强制回收并清空 CUDA 缓存
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    #     mean_final_result = np.mean(final_results)
    #     std_final_result = np.std(final_results)
    #     print(f"{final_results}")
    #     print(f"final result: {mean_final_result:.5f}±{std_final_result:.5}")
    #     print(f"final result: {mean_final_result*100:.2f}±{std_final_result*100:.2f}")
    for pe_loss_lameda in [args.masked_pe_loss]:# #[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        args.masked_pe_loss = pe_loss_lameda
        motif_num = 8192
        args.epochs = 1200
        if args.dataset == "chameleon":
            args.angle = 0.1
            args.epochs = 1000
        else:
            args.angle = 0.01
        if args.dataset == "actor":
            args.angle = 1.0
            args.epochs = 1200
        elif args.dataset == "citeseer":#or args.dataset == "pubmed":
            args.epochs = 500
        elif args.dataset == "cora":
            args.epochs = 300
        elif args.dataset == "texas":
            args.epochs = 800
        elif args.dataset == "citeseer":
            args.epochs = 1000
            # args.masked_pe_loss= 0.001
        # if args.dataset == "cornell":
        #     args.epochs = 1000
        #     args.angle = 1.0
        # args.angle = 0.01

        print(args)
        
        data = torch.load('../dataset/{}.pt'.format(args.dataset))
        print(data)
        x = data.x.float().to(args.device)
        if x.shape[1] > 4096:
            svd = TruncatedSVD(n_components=4096, random_state=0)
            x_cpu = x.detach().cpu().numpy()
            x_reduced = svd.fit_transform(x_cpu)
            x = torch.from_numpy(x_reduced).float().to(args.device)
            args.feat_dim = x.shape[1]
        
        # 在创建模型之前添加
        print('实际的特征维度')
        print(x.shape)
        args.feat_dim = x.shape[1]  # 使用实际数据的特征维度
        args.num_node = x.shape[0]
        edge = data.edge_index.long().to(args.device)
        e = data.e[:args.max_freqs].float().to(args.device)
        u = data.u[:, :args.max_freqs].float().to(args.device)

        y = data.y.to(args.device)
        print(y.min().item(), y.max().item())
        nclass = y.max().item() + 1

        edge_index_pe, _ = remove_self_loops(edge, None)
        edge_index_pe, _ = add_self_loops(edge_index_pe, fill_value='mean', num_nodes=u.shape[0])
        PE = torch.linalg.norm(u[edge_index_pe[0]] - u[edge_index_pe[1]], dim=-1)  # [e_sum, 1]

        # 在数据加载后添加三角形模体管理器初始化
        print("🔍 初始化三角形模体管理器...")
        triangle_manager = TriangleMotifManager(data.edge_index.long(), x.shape[0], args.device)
        print(f"✅ 找到 {triangle_manager.get_triangle_count()} 个三角形")
        # ### ===========================尝试用新的思路===========================
        # edge_index_np = edge.cpu().numpy()
        # edges = list(zip(edge_index_np[0], edge_index_np[1]))

        # G = ig.Graph(directed=False)
        # G.add_vertices(x.shape[0])
        # G.add_edges(edges)
        # G.simplify()
        # # 3. 调用核心函数 count_triangles()
        # # 这个函数返回一个列表，列表的长度是节点数，值是每个节点参与的三角形数
        # triangle_counts_list = build_motif_participation_matrix(G,[3],[3])
        
        # # 4. 将结果转换为 PyTorch Tensor
        # triangle_counts_tensor = torch.tensor(triangle_counts_list, dtype=torch.float)
        
        # if args.dataset == "actor":
        #     batch_size = 7121
        # elif args.dataset == "cornell":
        #     batch_size = 59
        if triangle_manager.get_triangle_count() < motif_num:
            batch_size = triangle_manager.get_triangle_count()
        else:
            batch_size = motif_num
        
        
        import time
        import datetime
        nowtime_step = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        time1 = time.time()
        args.num_exp = 10
        final_results = []
        for ep_num in range(0,args.num_exp):
            scaler = GradScaler() # 在训练循环外初始化
            path_write = './logdata/{}{}{}{}exp{}encodermotifself'.format(nowtime_step,args.dataset,args.epochs,args.angle,ep_num)
            os.makedirs(path_write, exist_ok=True)
            writer = SummaryWriter(path_write)
            # 1. 创建迭代器对象
            
            args.seed = ep_num
            set_random_seed(ep_num)
            dataset = TriangleDataset(triangle_manager, epoch_size=args.epochs+1, batch_size=batch_size, negative_ratio=2.0)
            if args.dataset == "flickr":
                dataloader = DataLoader(
                    dataset,
                    batch_size=None,  # 因为我们的 __getitem__ 已经返回了批次
                    num_workers=1,    # 使用4个子进程在后台加载数据，这个值可以根据你的CPU核数调整
                    pin_memory=True,   # 如果使用GPU，可以加速数据从CPU到GPU的传输
                    worker_init_fn=worker_init_fn,
                    # persistent_workers=True, # 建议保留这些优化参数
                    # prefetch_factor=4
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=None,  # 因为我们的 __getitem__ 已经返回了批次
                    num_workers=8,    # 使用4个子进程在后台加载数据，这个值可以根据你的CPU核数调整
                    pin_memory=True,   # 如果使用GPU，可以加速数据从CPU到GPU的传输
                    worker_init_fn=worker_init_fn,
                    # persistent_workers=True, # 建议保留这些优化参数
                    # prefetch_factor=4
                )
            data_iterator = iter(dataloader)
            print('Checking data attributes')
            if hasattr(data, 'train_mask'):
                if len(data.train_mask.size()) > 1:
                    train_idx = torch.where(data.train_mask[:, args.seed])[0]
                    val_idx = torch.where(data.val_mask[:, args.seed])[0]
                    test_idx = torch.where(data.test_mask[:, args.seed])[0]
                else:
                    train_idx = torch.where(data.train_mask)[0]
                    val_idx = torch.where(data.val_mask)[0]
                    test_idx = torch.where(data.test_mask)[0]
            else:
                train_idx, val_idx, test_idx = split(y)

            # 创建三角形相关的损失函数
            triangle_angle_loss = TriangleAngleLoss().to(args.device)
            triangle_motif_loss = TriangleMotifLoss(u.shape[1]).to(args.device)
            encoder = GraphEncoder(out_dim=args.embed_dim, args=args).to(args.device)
            model = GraphAutoEncoder(encoder=encoder, num_atom_type=args.feat_dim, args=args).to(args.device)

            parameters = model.parameters()#chain(model.parameters(), triangle_motif_loss.parameters())
            # --- 请修改成下面这样 ---

            # 1. 首先，把模型的参数分成两组

            # 第1组：只包含 U_hat。我们为它专门指定一个高学习率。
            # 这里的 '* 100' 是一个例子，您可以根据需要调整这个系数 (比如 50, 200)。
            # u_hat_param_group = {
            #     'params': model.U_hat, 
            #     'lr': args.init_lr * 100  
            # }

            # # 第2组：包含模型中除了 U_hat 以外的所有其他参数。
            # # 我们不在这里为它指定 'lr'，这样它就会自动使用函数调用时的默认学习率。
            # other_params_group = {
            #     'params': [p for n, p in model.named_parameters() if 'U_hat' not in n]
            # }

            # # 2. 将这两组参数打包成一个列表
            # # 这就是我们要传递给优化器的新 `parameters`
            # parameters = [u_hat_param_group, other_params_group]

            if args.optim == "sgd":
                pass
            else:
                args.momentum = None
            optimizer = create_optimizer(opt=args.optim, parameters=parameters, lr=args.init_lr, weight_decay=float(args.weight_decay), momentum=args.momentum)

            if args.use_schedule:
                scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            else:
                scheduler = None

            for epoch in range(1, args.epochs+1):
                train_mae_epoch(graph_auto_encoder=model, x=x, edge_index=edge, u=u, PE=PE, edge_index_pe=edge_index_pe, optimizer=optimizer, 
                        triangle_manager=triangle_manager, triangle_angle_loss=triangle_angle_loss, triangle_motif_loss=triangle_motif_loss,data_iterator=data_iterator,n_iter=epoch,argst=args,writer=writer,scaler=scaler)
                print('epoch: ',epoch)
                # 在每个epoch结束后添加
                # if epoch % 100 == 0:  # 每100个epoch检查一次
                #     with torch.no_grad():
                #         u_hat_norm = torch.norm(model.U_hat, dim=1).mean().item()
                #         u_hat_deviation = torch.mean((model.U_hat - torch.eye(model.U_hat.size(0), device=model.U_hat.device)).abs()).item()
                #         print(f"[Epoch {epoch}] U_hat norm: {u_hat_norm:.4f}, deviation: {u_hat_deviation:.6f}")
                if scheduler:
                    scheduler.step()
            time2 = time.time()
            print('consume time {}'.format(time2-time1))
            # print(model.weight_motif)
            model.eval()
            triangles, negative_triplets = next(data_iterator)
            device = x.device # 获取模型/数据的设备
            triangles = triangles.to(device, non_blocking=True)
            # triangles = None
            with torch.no_grad():
                embed = model.embed(x, edge, u, edge_index_pe, triangles=triangles)
            acc, pred = node_evaluation(emb=embed, y=y, train_idx=train_idx, valid_idx=val_idx, test_idx=test_idx, epochs=args.epochs_eval, lr=args.lr_eval, weight_decay=args.wd_eval)
            print(f"Epoch {epoch}, ACC: {acc.item()}")
            writer.add_scalar('EXP/ACC', acc.item(), ep_num)
            final_results.append(acc.item())
            
            # 保存模型checkpoint（用于后续可视化）
            checkpoint_path = os.path.join(path_write, f'model_checkpoint_exp{ep_num}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'acc': acc.item(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
            print(f"模型已保存至: {checkpoint_path}")

            # ==== 资源清理，避免多次实验累积显存占用 ====
            try:
                writer.close()
            except Exception:
                pass
            # 删除迭代器/数据加载器/数据集
            try:
                del data_iterator
            except Exception:
                pass
            try:
                del dataloader
            except Exception:
                pass
            try:
                del dataset
            except Exception:
                pass
            # 删除模型与优化相关对象
            try:
                del model
            except Exception:
                pass
            try:
                del encoder
            except Exception:
                pass
            try:
                del optimizer
            except Exception:
                pass
            try:
                del scheduler
            except Exception:
                pass
            try:
                del triangle_angle_loss
            except Exception:
                pass
            try:
                del triangle_motif_loss
            except Exception:
                pass
            try:
                del scaler
            except Exception:
                pass
            # 强制回收并清空 CUDA 缓存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_final_result = np.mean(final_results)
        std_final_result = np.std(final_results)
        print(f"{final_results}")
        print(f"pe_loss_lameda: {pe_loss_lameda}")
        print(f"final result: {mean_final_result:.5f}±{std_final_result:.5}")
        print(f"final result: {mean_final_result*100:.2f}±{std_final_result*100:.2f}")