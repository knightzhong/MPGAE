import torch
import torch.nn as nn

from utils import get_activation
from conv import GATConv     
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, encoder, out_dim, args=None):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(args.embed_dim, out_dim)
    
    def forward(self, x, edge_index, eigvals=None, eigvecs=None, mask_tokens=None, PE=None):
        h, pe = self.encoder.embed(x, edge_index, eigvals, eigvecs, mask_tokens, PE)
        pred = self.classifier(h)
        return pred, pe



class GraphEncoder(torch.nn.Module):
    def __init__(self, out_dim, args=None):
        super().__init__()
        self.gnn_dropout = args.gnn_dropout

        self.num_layer = args.enc_gnn_layer
        emb_dim = args.embed_dim
        hid_dim = emb_dim // args.heads

        self.gnns = nn.ModuleList()
        if self.num_layer == 1:
            self.gnns.append(GATConv(in_channels=args.feat_dim, out_channels=out_dim, heads=args.heads, args=args))
        else:
            self.gnns.append(GATConv(in_channels=args.feat_dim, out_channels=hid_dim, heads=args.heads, args=args))
            self.activations = nn.ModuleList()
            for layer in range(self.num_layer - 1):
                self.gnns.append(GATConv(in_channels=emb_dim, out_channels=hid_dim, heads=args.heads, args=args))        
                self.activations.append(get_activation(args.gnn_activation))

    def forward(self, x, x_masked, edge_index, PE=None, PE_noise=None, motif_tensor=None, return_gates=False):
        if motif_tensor is not None and not return_gates:
            print('==========================================================')
        pe = None
        pe_noise = None
        h_list = [x]
        h_masked_list = [x_masked]
        pe_list = [pe]
        pe_noise_list = [pe_noise]
        all_gates = []  # 存储所有层的gate信息
        
        for layer in range(self.num_layer):
            h = F.dropout(h_list[layer], p=self.gnn_dropout, training=self.training)
            result_noise = self.gnns[layer](h, pe_noise_list[layer], edge_index, motif_tensor=motif_tensor, return_gates=return_gates)
            if return_gates:
                h, pe_noise, gate_info_noise = result_noise
                all_gates.append({f'noise_{k}': v for k, v in gate_info_noise.items()})
            else:
                h, pe_noise = result_noise

            h_masked = F.dropout(h_masked_list[layer], p=self.gnn_dropout, training=self.training)
            result = self.gnns[layer](h_masked, pe_list[layer], edge_index, motif_tensor=motif_tensor, return_gates=return_gates)
            if return_gates:
                h_masked, pe, gate_info = result
                all_gates.append({f'clean_{k}': v for k, v in gate_info.items()})
            else:
                h_masked, pe = result

            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
                h_masked = self.activations[layer](h_masked)
            
            h_masked_list.append(h_masked)
            h_list.append(h)
            pe_noise_list.append(pe_noise)
            pe_list.append(pe)
        Z_node = h_masked_list[-1]
        Z_node_pe = h_list[-1]

        if return_gates:
            return Z_node, Z_node_pe, all_gates
        return Z_node, Z_node_pe

    def embed(self, x, edge_index, PE=None, motif_tensor=None, return_gates=False):
        
        pe = None
        h_list = [x]
        pe_list = [pe]
        all_gates = []  # 存储所有层的gate信息
        
        for layer in range(self.num_layer):
            h = F.dropout(h_list[layer], p=self.gnn_dropout, training=self.training)
            result = self.gnns[layer](h, pe_list[layer], edge_index, motif_tensor=motif_tensor, return_gates=return_gates)
            if return_gates:
                h, pe, gate_info = result
                all_gates.append(gate_info)
            else:
                h, pe = result

            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
            
            h_list.append(h)
            pe_list.append(pe)
        
        Z_node = h_list[-1]
        
        if return_gates:
            return Z_node, pe_list[-1], all_gates
        return Z_node, pe_list[-1]