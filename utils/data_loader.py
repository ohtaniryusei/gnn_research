# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:46:17 2025

@author: ryuse
"""

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

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
