import torch
import torch.nn as nn
from time_encoder import MultiViewTimeEncoder

class GraphMixerEncoder(nn.Module):
    def __init__(self, node_dim, time_config, out_dim):
        super().__init__()
        d_re, d_ab, d_se, d_time, _ = time_config
        self.mte = MultiViewTimeEncoder(d_re, d_ab, d_se, d_time)

        self.token_mixer = nn.Sequential(
            nn.Linear(10, 10),  # 近傍数固定（仮に10件）に対応
            nn.GELU(),
            nn.Linear(10, 10)
        )

        self.feature_mixer = nn.Sequential(
            nn.Linear(node_dim + d_time, node_dim + d_time),
            nn.GELU(),
            nn.Linear(node_dim + d_time, out_dim)
        )

    def forward(self, h_neighbors, t, t_prime, semantic_feat):
        """
        h_neighbors: [B, N, D]
        t: [B]
        t_prime: [B, N]
        semantic_feat: [B, N, d_se]
        """
        B, N, D = h_neighbors.shape

        # MTEエンコーディング
        t = t.unsqueeze(1).expand(-1, N).reshape(-1)
        t_prime_flat = t_prime.reshape(-1)
        sem_feat_flat = semantic_feat.reshape(-1, semantic_feat.shape[-1])
        time_embed = self.mte(t, t_prime_flat, sem_feat_flat)  # [B*N, d_time]
        time_embed = time_embed.reshape(B, N, -1)

        x = torch.cat([h_neighbors, time_embed], dim=-1)  # [B, N, D + d_time]

        # トークンミキサー（時間方向）
        x_t = x.permute(0, 2, 1)  # [B, D+d_time, N]
        x_t = self.token_mixer(x_t)
        x_t = x_t.permute(0, 2, 1)  # [B, N, D+d_time]

        # 特徴ミキサー（次元方向）
        x_f = self.feature_mixer(x_t)  # [B, N, out_dim]

        # 平均プーリング
        out = x_f.mean(dim=1)  # [B, out_dim]
        return out
