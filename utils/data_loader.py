# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:46:17 2025

@author: ryuse
"""
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric.loader import NeighborLoader
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph
import random

def load_cora_batch(batch_size=32, neighbor_k=10, seed=42):
    dataset = Planetoid(root='data/', name='Cora')
    data = dataset[0]
    torch.manual_seed(seed)

    # ノード特徴とラベル
    x_all = data.x  # [N, D]
    y_all = data.y  # [N]
    edge_index = data.edge_index  # [2, E]

    node_indices = data.train_mask.nonzero(as_tuple=True)[0]
    selected_nodes = node_indices[torch.randperm(len(node_indices))[:batch_size]]

    h_static = x_all[selected_nodes]  # [B, D]
    labels = (y_all[selected_nodes] == 0).float()  # ✅ 0 vs 他の2値分類に変換

    # 擬似的な取引履歴（近傍情報）を構成
    h_neighbors = []
    t = []
    t_prime = []
    semantic_feat = []

    for node_id in selected_nodes:
        # k-hop近傍を取得（固定数じゃないのでランダムにk個選ぶ）
        _, neighbors, _, _ = k_hop_subgraph(node_id.item(), 1, edge_index, num_nodes=x_all.size(0))
        neighbors = neighbors[neighbors != node_id][:neighbor_k]
        if len(neighbors) < neighbor_k:
            pad = neighbors[torch.randint(0, len(neighbors), (neighbor_k - len(neighbors),))]
            neighbors = torch.cat([neighbors, pad], dim=0)
        else:
            neighbors = neighbors[:neighbor_k]

        h_neigh = x_all[neighbors]  # [k, D]
        h_neighbors.append(h_neigh)

        # 擬似的な時刻（0〜1000）を生成
        center_time = random.randint(500, 1000)
        t.append(center_time)
        t_prime.append(torch.tensor([center_time - random.randint(1, 500) for _ in range(neighbor_k)], dtype=torch.float))

        # 擬似的な時間セマンティクス（祝日・週末フラグなどの1-hot）
        semantic_feat.append(torch.randint(0, 2, (neighbor_k, 8)).float())

    h_neighbors = torch.stack(h_neighbors)        # [B, K, D]
    t = torch.tensor(t, dtype=torch.float)        # [B]
    t_prime = torch.stack(t_prime)                # [B, K]
    semantic_feat = torch.stack(semantic_feat)    # [B, K, 8]

    return h_static, h_neighbors, t, t_prime, semantic_feat, labels


def load_static_graph(path='data/transfers.csv', num_users=100):
    df = pd.read_csv(path)
    edge_index = torch.tensor(df.values.T, dtype=torch.long)  # [2, num_edges]
    x = torch.randn(num_users, 128)  # ダミー特徴
    data = Data(x=x, edge_index=edge_index)
    return data

def load_transaction_batch(path='data/transactions.csv', batch_size=32):
    df = pd.read_csv(path)
    df = df.sample(n=batch_size)  # ランダムバッチ

    user_ids = df['user_id'].tolist()
    merchant_ids = df['merchant_id'].tolist()
    timestamps = torch.tensor(df['timestamp'].values, dtype=torch.float)
    labels = torch.tensor(df['label'].values, dtype=torch.float)

    # 各ユーザーに10件の近傍を割り当てる（ここは本来時系列順処理）
    h_neighbors = torch.randn(batch_size, 10, 128)
    t_prime = torch.clamp(timestamps.unsqueeze(1) - torch.randint(1000, 10000, (batch_size, 10), dtype=torch.float), min=0)
    semantic_feat = torch.randint(0, 2, (batch_size, 10, 8)).float()  # 例：祝日/週末など

    h_static = torch.randn(batch_size, 128)

    return h_static, h_neighbors, timestamps, t_prime, semantic_feat, labels
