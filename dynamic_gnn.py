# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:34:34 2025

@author: ryuse
"""

import torch
import torch.nn as nn
from temporal_attention import TemporalAttention
from time_encoder import MultiViewTimeEncoder

class DynamicGraphEncoder(nn.Module):
    def __init__(self, node_dim, time_config, out_dim):
        super().__init__()
        d_re, d_ab, d_se, d_time, d_hidden = time_config

        self.mte = MultiViewTimeEncoder(d_re, d_ab, d_se, d_time)
        self.temporal_attn = TemporalAttention(node_dim, d_time)

        self.linear = nn.Linear(node_dim, out_dim)

    def forward(self, h_neighbors, t, t_prime, semantic_feat):
        """
        h_neighbors: [B, N, D] - 隣接ノードの特徴
        t: [B] - 中心ノードの時刻
        t_prime: [B, N] - 各隣接ノードの時刻
        semantic_feat: [B, N, d_se] - 各取引に対する時間意味特徴（祝日など）

        Returns:
            [B, D_out] - 中心ノードの出力表現
        """
        B, N, D = h_neighbors.shape
        t = t.unsqueeze(1).expand(-1, N)  # [B, N]
        t_flat = t.reshape(-1)
        t_prime_flat = t_prime.reshape(-1)
        sem_flat = semantic_feat.reshape(-1, semantic_feat.shape[-1])

        time_embed = self.mte(t_flat, t_prime_flat, sem_flat)  # [B*N, d_time]
        time_embed = time_embed.reshape(B, N, -1)

        out, _ = self.temporal_attn(h_neighbors, time_embed)  # [B, N, D]
        out = out.mean(dim=1)  # [B, D]
        out = self.linear(out)  # [B, D_out]
        return out
